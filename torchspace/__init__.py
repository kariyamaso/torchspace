"""TorchSpace — a spatial debugger for PyTorch, built as a non-invasive
extension of torchview. PoC v0.1 (validates the Design Specification).
"""
from .api import SpaceRun, trace, view  # noqa: F401
from .ir import SCHEMA_VERSION  # noqa: F401

__version__ = "0.1.0a1"
__all__ = ["view", "trace", "SpaceRun", "SCHEMA_VERSION", "__version__"]
