from __future__ import annotations

import importlib.util
from typing import Literal

from cv_preprocess.compute.polars_backend import PolarsComputeBackend
from cv_preprocess.compute.protocol import ComputeBackend
from cv_preprocess.compute.python_backend import PythonComputeBackend

BackendName = Literal["auto", "polars", "python"]


def _polars_importable() -> bool:
    return importlib.util.find_spec("polars") is not None


def resolve_compute_backend(requested: BackendName | str = "auto") -> ComputeBackend:
    """Resolve ``auto`` to Polars when importable, otherwise Python."""
    if requested == "python":
        return PythonComputeBackend()
    if requested == "polars":
        if not _polars_importable():
            raise ImportError("polars backend requested but polars is not installed")
        return PolarsComputeBackend()
    if requested == "auto":
        if _polars_importable():
            return PolarsComputeBackend()
        return PythonComputeBackend()
    raise ValueError(f"unsupported compute backend: {requested!r}")
