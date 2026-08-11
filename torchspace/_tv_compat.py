"""Single import surface for every torchview symbol TorchSpace touches.

TorchSpace extends torchview *non-invasively*: no torchview code is copied or
forked. Everything we rely on is imported here, in one place, so that a
torchview version bump is audited against exactly this file. If torchview's
internals move, only this module (and the two wrappers in tracer.py) need
attention.

Compatibility gate: we pin a tested range and fail loudly otherwise.
"""
from __future__ import annotations

import warnings

import torchview

TESTED_VERSIONS = ("0.2.6", "0.2.7")

if torchview.__version__ not in TESTED_VERSIONS:
    warnings.warn(
        f"torchspace was validated against torchview {TESTED_VERSIONS}; "
        f"found {torchview.__version__}. Tracing may still work, but run the "
        "parity test-suite before trusting results.",
        RuntimeWarning,
    )

# --- public / stable-ish torchview API ------------------------------------
from torchview import (  # noqa: E402,F401
    ComputationGraph,
    FunctionNode,
    ModuleNode,
    RecorderTensor,
    TensorNode,
)

# --- internals we deliberately depend on (audited per release) ------------
from torchview.torchview import process_input  # noqa: E402,F401
from torchview.recorder_tensor import (  # noqa: E402,F401
    Recorder,
    _orig_module_forward,
    collect_tensor_node,
    module_forward_wrapper,
    reduce_data_info,
)
from torchview.computation_node import NodeContainer  # noqa: E402,F401

TORCHVIEW_VERSION = torchview.__version__
