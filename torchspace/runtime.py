"""Runtime collector: forward / backward capture on *plain* tensors.

Structure comes from the torchview-extended trace (tracer.py); runtime
behaviour comes from ordinary PyTorch hooks on the user's real forward and
backward passes. The two are joined by the stable key (module fqn,
call_index within step) — the same key the structural pass stamps on every
ModuleNode. This keeps runtime capture usable inside real training loops,
with no RecorderTensor anywhere near the hot path.

Capture policy (spec §13): statistics only, computed on-device, detached
immediately; backward capture is opt-in (`arm_backward=True`).
"""
from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Any, Iterator, Optional

import torch
from torch import nn

from .stats import tensor_stats


class RuntimeSession:
    #: hard cap on retained frames — a forgotten detach() inside a long
    #: training loop must degrade into a warning, not memory exhaustion
    MAX_FRAMES = 200_000

    def __init__(self, model: nn.Module, capture_activations: bool = True,
                 arm_backward: bool = True,
                 max_frames: int | None = None):
        self.model = model
        self.capture_activations = capture_activations
        self.arm_backward = arm_backward
        self.max_frames = self.MAX_FRAMES if max_frames is None else max_frames
        self._overflowed = False
        self.frames: list[dict[str, Any]] = []
        self.step = -1
        self.seq = 0
        self._call_counter: dict[str, int] = defaultdict(int)
        self._handles: list[Any] = []
        self._installed = False
        self._suspended = False

    # ------------------------------------------------------------------ #
    def install(self) -> "RuntimeSession":
        if self._installed:
            return self
        root = self.model
        self._handles.append(
            root.register_forward_pre_hook(self._on_step_begin))
        for fqn, mod in self.model.named_modules():
            self._handles.append(
                mod.register_forward_hook(self._make_hook(fqn)))
        self._installed = True
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._installed = False

    def _record(self, frame: dict[str, Any]) -> None:
        if len(self.frames) >= self.max_frames:
            if not self._overflowed:
                self._overflowed = True
                import warnings
                warnings.warn(
                    f"torchspace RuntimeSession reached max_frames="
                    f"{self.max_frames}; further frames are dropped. "
                    "Call run.detach() when done capturing, or raise "
                    "max_frames.", RuntimeWarning)
            return
        self.frames.append(frame)

    @contextlib.contextmanager
    def suspended(self) -> Iterator[None]:
        """Temporarily mutes all capture (used by auxiliary forward passes
        such as SpaceRun.draw_graph that must not pollute the frame log)."""
        prev = self._suspended
        self._suspended = True
        try:
            yield
        finally:
            self._suspended = prev

    # ------------------------------------------------------------------ #
    def _on_step_begin(self, mod: nn.Module, args: Any) -> None:
        if self._suspended:
            return
        self.step += 1
        self._call_counter.clear()

    def _make_hook(self, fqn: str):
        def hook(mod: nn.Module, args: Any, out: Any) -> None:
            if self._suspended:
                return
            k = self._call_counter[fqn]
            self._call_counter[fqn] += 1
            for j, t in enumerate(_tensors_of(out)):
                if self.capture_activations:
                    s = tensor_stats(t)
                    if s is not None:
                        self._record(dict(
                            step=self.step, seq=self._tick(), phase="forward",
                            fqn=fqn, call_index=k, out_index=j,
                            kind="activation", stats=s))
                if (self.arm_backward and t.requires_grad
                        and torch.is_grad_enabled()):
                    t.register_hook(self._make_grad_hook(fqn, k, j, self.step))
        return hook

    def _make_grad_hook(self, fqn: str, k: int, j: int, step: int):
        def grad_hook(grad: torch.Tensor) -> None:
            # armed on tensors during forward: an in-flight backward may run
            # after detach(), and must not keep appending frames then
            if not self._installed or self._suspended:
                return
            s = tensor_stats(grad)
            if s is not None:
                self._record(dict(
                    step=step, seq=self._tick(), phase="backward",
                    fqn=fqn, call_index=k, out_index=j,
                    kind="gradient", stats=s))
        return grad_hook

    def _tick(self) -> int:
        self.seq += 1
        return self.seq - 1

    # ------------------------------------------------------------------ #
    def capture_backward(self, loss: torch.Tensor, **backward_kwargs: Any) -> None:
        """Runs loss.backward() and collects per-parameter gradient stats."""
        loss.backward(**backward_kwargs)
        self.collect_param_grads()

    def collect_param_grads(self) -> None:
        for fqn, mod in self.model.named_modules():
            for pname, p in mod.named_parameters(recurse=False):
                if p.grad is not None:
                    s = tensor_stats(p.grad)
                    if s is not None:
                        self._record(dict(
                            step=self.step, seq=self._tick(), phase="backward",
                            fqn=fqn, call_index=0, out_index=0,
                            kind="param_grad", param=pname, stats=s))


def _tensors_of(out: Any) -> list[torch.Tensor]:
    found: list[torch.Tensor] = []

    def visit(o: Any) -> None:
        if isinstance(o, torch.Tensor):
            found.append(o)
        elif isinstance(o, (list, tuple)):
            for i in o:
                visit(i)
        elif isinstance(o, dict):
            for v in o.values():
                visit(v)
    visit(out)
    return found
