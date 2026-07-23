from __future__ import annotations

from cv_preprocess.compute.loader import resolve_compute_backend
from cv_preprocess.compute.polars_backend import PolarsComputeBackend
from cv_preprocess.compute.protocol import ComputeBackend, SelectionState
from cv_preprocess.compute.profiling import ResourceSnapshot, measure_callable, resource_snapshot
from cv_preprocess.compute.python_backend import PythonComputeBackend

__all__ = [
    "ComputeBackend",
    "PolarsComputeBackend",
    "PythonComputeBackend",
    "SelectionState",
    "ResourceSnapshot",
    "measure_callable",
    "resolve_compute_backend",
    "resource_snapshot",
]
