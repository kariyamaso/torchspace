"""Structure collector: a non-invasive extension of torchview's tracer.

torchview already solves structural tracing (modules, functions, tensors,
hierarchy, branches, recursion) via RecorderTensor + a patched
nn.Module.__call__. TorchSpace needs four things torchview does not record:

  1. a stable identity for every module call  -> module FQN + call index
  2. execution order & timing                 -> seq counter, perf_counter_ns
  3. tensor runtime metadata                  -> dtype / device / nbytes / grad
  4. parameter counts per module              -> own + total

All four are obtained WITHOUT copying or forking torchview code, through two
scoped extension points, active only inside the trace context:

  A. `spatial_forward_wrapper` wraps (composes around) torchview's
     `module_forward_wrapper`. After the wrapped call returns, the ModuleNode
     torchview just created is recovered from the context list torchview
     appends it to, and enriched via a side attribute (`ts_meta`).

  B. `RecorderTensor.__torch_function__` is temporarily composed with an
     after-hook that stamps sequence numbers on new FunctionNodes and dtype /
     nbytes metadata on new TensorNodes. Restored on exit, even on error.

Metadata reads on RecorderTensor are guarded with
`torch._C.DisableTorchFunctionSubclass()` so that enrichment itself never
creates graph nodes.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Optional

import graphviz
import torch
from torch import nn

from ._tv_compat import (
    ComputationGraph,
    FunctionNode,
    NodeContainer,
    Recorder,
    RecorderTensor,
    TensorNode,
    _orig_module_forward,
    collect_tensor_node,
    module_forward_wrapper,
    process_input,
    reduce_data_info,
)
from .stats import tensor_stats

FULL_DETAIL = dict(
    show_shapes=True,
    expand_nested=True,
    hide_inner_tensors=False,   # capture every tensor: views filter later
    hide_module_functions=False,  # capture every op: views filter later
    roll=False,                 # IR keeps the unrolled truth; rolling is a view
)

# Tracing swaps RecorderTensor.__torch_function__ and (via torchview's
# Recorder) nn.Module.__call__ process-wide. Interleaved traces would capture
# each other's wrappers as "original" and leave stale hooks installed after
# both finish, so the whole patched region is serialised.
_TRACE_LOCK = threading.Lock()


class TraceResult:
    def __init__(self, model: nn.Module, model_graph: ComputationGraph,
                 fqn_of: dict[int, str], aliases: dict[int, list[str]],
                 input_meta: list[dict[str, Any]],
                 trace_kwargs: Optional[dict[str, Any]] = None):
        self.model = model
        self.model_graph = model_graph
        self.fqn_of = fqn_of          # id(nn.Module) -> canonical fqn
        self.aliases = aliases        # id(nn.Module) -> extra fqns (shared modules)
        self.input_meta = input_meta
        # exact inputs used for the structural pass; lets SpaceRun.draw_graph /
        # export_dot re-render through torchview's own graphviz pipeline
        self.trace_kwargs = trace_kwargs or {}


class StructureTracer:
    """Runs one recorded forward pass at full detail and returns an enriched
    torchview ComputationGraph ready for IR building."""

    def __init__(self, model: nn.Module, capture_activations: bool = False,
                 collect_attributes: bool = True):
        self.model = model
        self.capture_activations = capture_activations
        self.collect_attributes = collect_attributes
        self.seq = 0
        self.call_counter: dict[str, int] = defaultdict(int)
        # canonical fqn = first binding; extra bindings recorded as aliases
        self.fqn_of: dict[int, str] = {}
        self.aliases: dict[int, list[str]] = defaultdict(list)
        for name, mod in model.named_modules(remove_duplicate=False):
            if id(mod) in self.fqn_of:
                self.aliases[id(mod)].append(name)
            else:
                self.fqn_of[id(mod)] = name

    # ------------------------------------------------------------------ #
    def trace(self, input_data: Any = None, input_size: Any = None,
              device: str | torch.device = "cpu", mode: str = "eval",
              dtypes: Optional[list[torch.dtype]] = None,
              **forward_kwargs: Any) -> TraceResult:
        mg = ComputationGraph(
            graphviz.Digraph("torchspace", strict=True),
            NodeContainer(),  # replaced below by process_input result
            depth=float("inf"),
            collect_attributes=self.collect_attributes,
            **FULL_DETAIL,
        )
        x, kw_rec, input_nodes = process_input(
            input_data, input_size, forward_kwargs, device, dtypes,
            self.collect_attributes)
        # torchview only threads collect_attributes through the input_size
        # path; the flag propagates from input TensorNodes, so stamp them
        # here to cover the input_data path too.
        if self.collect_attributes:
            for node in input_nodes:
                node.collect_attributes = True
        mg.root_container = input_nodes
        mg.reset_graph_history()

        input_meta = []
        for i, t in enumerate(x if isinstance(x, (list, tuple)) else [x]):
            if isinstance(t, RecorderTensor):
                with torch._C.DisableTorchFunctionSubclass():
                    meta = _tensor_meta(t)
                for node in t.tensor_nodes:
                    node.ts_meta = dict(meta)
                input_meta.append(meta)

        base_forward = module_forward_wrapper(mg)
        spatial_forward = self._make_forward_wrapper(base_forward, mg)
        orig_tf = RecorderTensor.__torch_function__.__func__
        spatial_tf = self._make_torch_function(orig_tf)

        saved_training = self.model.training
        with _TRACE_LOCK:
            RecorderTensor.__torch_function__ = classmethod(spatial_tf)
            try:
                self.model.train(mode == "train")
                with Recorder(_orig_module_forward, spatial_forward, mg):
                    with torch.no_grad():
                        m = self.model.to(device)
                        if isinstance(x, (list, tuple)):
                            m(*x, **kw_rec)
                        else:
                            m(**x, **kw_rec)
            finally:
                RecorderTensor.__torch_function__ = classmethod(orig_tf)
                self.model.train(saved_training)

        mg.fill_visual_graph()  # populates mg.edge_list (full-detail edges)
        return TraceResult(self.model, mg, self.fqn_of, dict(self.aliases),
                           input_meta,
                           trace_kwargs=dict(
                               input_data=input_data, input_size=input_size,
                               device=device, mode=mode, dtypes=dtypes,
                               forward_kwargs=forward_kwargs))

    # ------------------------------------------------------------------ #
    def _make_forward_wrapper(self, base_forward, mg):
        tracer = self

        def spatial_forward(mod: nn.Module, *args: Any, **kwargs: Any) -> Any:
            # Recover the context list torchview will append the new
            # ModuleNode's {node: [...]} entry to: the context of the first
            # input TensorNode (same lookup torchview itself performs).
            in_nodes = reduce_data_info(
                [args, kwargs], collect_tensor_node, NodeContainer())
            ctx = next(iter(in_nodes)).context if in_nodes else None
            n0 = len(ctx) if ctx is not None else 0

            seq_in = tracer.seq
            t0 = time.perf_counter_ns()
            out = base_forward(mod, *args, **kwargs)
            elapsed = time.perf_counter_ns() - t0

            if ctx is not None:
                entry = next(
                    (e for e in ctx[n0:] if isinstance(e, dict)), None)
                if entry is not None:
                    mnode = next(iter(entry))
                    tracer._enrich_module_node(mnode, mod, seq_in, elapsed, out)
            return out

        return spatial_forward

    def _enrich_module_node(self, mnode, mod, seq_in, elapsed_ns, out):
        fqn = self.fqn_of.get(id(mod), f"<external:{type(mod).__name__}>")
        call_index = self.call_counter[fqn]
        self.call_counter[fqn] += 1
        meta = dict(
            fqn=fqn,
            call_index=call_index,
            seq_in=seq_in,
            seq_out=self.seq,
            time_ns=elapsed_ns,
            params_own=sum(p.numel() for p in mod.parameters(recurse=False)),
            params_total=sum(p.numel() for p in mod.parameters()),
            trainable=sum(p.numel() for p in mod.parameters() if p.requires_grad),
            is_leaf_module=not any(mod.children()),
            aliases=self.aliases.get(id(mod), []),
        )
        if self.capture_activations:
            outs = reduce_data_info(out, _collect_recorder, [])
            # Guard: stats math on RecorderTensors must NOT dispatch through
            # __torch_function__, or the stats themselves would add graph nodes.
            with torch._C.DisableTorchFunctionSubclass():
                meta["activation"] = [tensor_stats(t) for t in outs]
        mnode.ts_meta = meta

    # ------------------------------------------------------------------ #
    def _make_torch_function(self, orig_tf):
        tracer = self

        def spatial_tf(cls, func, types, args=(), kwargs=None):
            out = orig_tf(cls, func, types, args, kwargs)
            tracer._enrich_from_output(out)
            return out

        return spatial_tf

    def _enrich_from_output(self, out: Any) -> None:
        for rt in reduce_data_info(out, _collect_recorder, []):
            nodes = getattr(rt, "tensor_nodes", None)
            if not nodes:
                continue
            tnode = nodes[-1]
            if hasattr(tnode, "ts_meta"):
                continue  # pre-existing node (op returned input unchanged)
            with torch._C.DisableTorchFunctionSubclass():
                tnode.ts_meta = _tensor_meta(rt)
            for p in tnode.parents:
                if isinstance(p, FunctionNode) and not hasattr(p, "ts_meta"):
                    p.ts_meta = dict(seq=self.seq)
                    self.seq += 1


def _tensor_meta(t: torch.Tensor) -> dict[str, Any]:
    return dict(
        dtype=str(t.dtype).replace("torch.", ""),
        device=str(t.device),
        nbytes=t.element_size() * t.nelement(),
        requires_grad=bool(t.requires_grad),
    )


def _collect_recorder(data, collected: list) -> None:
    if isinstance(data, RecorderTensor):
        collected.append(data)


def _collect_plain(data, collected: list) -> None:
    if isinstance(data, torch.Tensor):
        collected.append(data)
