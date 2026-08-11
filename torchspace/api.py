"""Public API (spec §5):

    from torchspace import view, trace

    view(model, input_size=(1, 3, 224, 224))          # structure only
    run = trace(model, input_data=x)                  # structure + hooks
    loss = criterion(model(x), y)                     # captured forward
    run.capture_backward(loss)                        # captured backward
    run.export_html("model.torchspace.html")          # self-contained scene
"""
from __future__ import annotations

import contextlib
from typing import Any, Optional

import torch
from torch import nn

from .diagnostics import run_rules
from .ir import build_ir, to_json
from .runtime import RuntimeSession
from .tracer import StructureTracer, TraceResult


class SpaceRun:
    def __init__(self, model: nn.Module, trace_result: TraceResult,
                 runtime: Optional[RuntimeSession], model_name: str):
        self.model = model
        self.trace_result = trace_result
        self.runtime = runtime
        self.model_name = model_name
        self._ir: Optional[dict[str, Any]] = None

    # -- capture ------------------------------------------------------- #
    def capture_backward(self, loss: torch.Tensor, **kw: Any) -> None:
        if self.runtime is None:
            raise RuntimeError("trace() this model with capture enabled first")
        self.runtime.capture_backward(loss, **kw)
        self._ir = None

    def detach(self) -> None:
        """Remove all hooks from the model."""
        if self.runtime is not None:
            self.runtime.remove()

    # -- results ------------------------------------------------------- #
    @property
    def ir(self) -> dict[str, Any]:
        if self._ir is None:
            self._ir = build_ir(self.trace_result, self.runtime,
                                self.model_name)
            run_rules(self._ir)
        return self._ir

    def save_ir(self, path: str, indent: int | None = None) -> str:
        with open(path, "w") as f:
            f.write(to_json(self.ir, indent=indent))
        return path

    def export_html(self, path: str, title: str | None = None) -> str:
        from .export import export_html
        return export_html([self.ir], path, title or self.model_name)

    # -- torchview parity ----------------------------------------------- #
    def draw_graph(self, **kw: Any):
        """Renders this run's model through torchview's own graphviz
        pipeline, using the exact inputs of the structural trace.

        Accepts every `torchview.draw_graph` option (`depth`,
        `hide_inner_tensors`, `hide_module_functions`, `roll`, `graph_dir`,
        `expand_nested`, `show_shapes`, `strict`, `collect_attributes`,
        `save_graph`, `filename`, `directory`, ...) and returns torchview's
        ComputationGraph — `.visual_graph` is the graphviz Digraph.
        Runtime capture hooks are suspended for this auxiliary pass, so it
        never pollutes the frame log."""
        import torchview as _tv
        tk = self.trace_result.trace_kwargs
        args: dict[str, Any] = dict(
            model=self.model,
            input_data=tk.get("input_data"),
            input_size=tk.get("input_size"),
            device=tk.get("device"),
            mode=tk.get("mode"),
            dtypes=tk.get("dtypes"),
            graph_name=self.model_name,
            **(tk.get("forward_kwargs") or {}),
        )
        args.update(kw)
        ctx = self.runtime.suspended() if self.runtime is not None \
            else contextlib.nullcontext()
        with ctx:
            return _tv.draw_graph(**args)

    def export_dot(self, path: str | None = None, **kw: Any) -> str:
        """torchview-style graphviz output. Returns the DOT source (no
        graphviz binary needed); writes it to `path` if given. Pass
        `save_graph=True, filename=..., format=...` to also render an
        image via the graphviz binary (torchview's own save path)."""
        src = self.draw_graph(**kw).visual_graph.source
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)
        return src

    # -- notebooks (Jupyter / Colab / VS Code) -------------------------- #
    def show(self, height: int = 650, width: str = "100%"):
        """Displays the interactive 3D scene inline in a notebook.

        Self-contained (three.js inlined, IR embedded, sandboxed iframe):
        works in Colab and offline Jupyter with no server and no network.
        """
        from IPython.display import HTML, display  # lazy: notebook-only
        display(HTML(self._iframe_html(height=height, width=width)))

    def _iframe_html(self, height: int = 650, width: str = "100%") -> str:
        from .export import iframe_html
        return iframe_html([self.ir], self.model_name,
                           height=height, width=width)

    def _repr_html_(self) -> str:
        """A SpaceRun at the end of a notebook cell renders the scene."""
        return self._iframe_html()


def view(model: nn.Module, input_data: Any = None, input_size: Any = None,
         device: str = "cpu", mode: str = "eval",
         model_name: Optional[str] = None,
         collect_attributes: bool = True, **kw: Any) -> SpaceRun:
    """Architecture-only inspection (torchview-class capability)."""
    tracer = StructureTracer(model, collect_attributes=collect_attributes)
    tr = tracer.trace(input_data=input_data, input_size=input_size,
                      device=device, mode=mode, **kw)
    return SpaceRun(model, tr, None, model_name or type(model).__name__)


def trace(model: nn.Module, input_data: Any = None, input_size: Any = None,
          capture_activations: bool = True, arm_backward: bool = True,
          device: str = "cpu", mode: str = "eval",
          model_name: Optional[str] = None,
          collect_attributes: bool = True, **kw: Any) -> SpaceRun:
    """Structure trace + runtime capture hooks (activations, gradients).

    After this call the model is instrumented: the next forward pass is
    recorded as frames; if arm_backward, gradients of every module output
    are recorded when backward runs. Call run.detach() to de-instrument.
    """
    tracer = StructureTracer(model, collect_attributes=collect_attributes)
    tr = tracer.trace(input_data=input_data, input_size=input_size,
                      device=device, mode=mode, **kw)
    rt = RuntimeSession(model, capture_activations=capture_activations,
                        arm_backward=arm_backward).install()
    return SpaceRun(model, tr, rt, model_name or type(model).__name__)
