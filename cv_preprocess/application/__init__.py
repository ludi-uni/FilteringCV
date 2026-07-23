from __future__ import annotations

from typing import Any

__all__ = [
    "AnalyzeResult",
    "AuditReport",
    "CancellationToken",
    "MaterializeResult",
    "ProgressEvent",
    "ProgressSink",
    "ScanResult",
    "SelectionPlan",
    "SplitPlan",
    "analyze_project",
    "audit_dataset",
    "build_dataset",
    "materialize_dataset",
    "plan_dataset_split",
    "scan_project",
    "select_dataset",
]

_COMMON = {
    "AnalyzeResult",
    "AuditReport",
    "CancellationToken",
    "MaterializeResult",
    "ProgressEvent",
    "ProgressSink",
    "ScanResult",
    "SelectionPlan",
    "SplitPlan",
}

_LAZY_FUNCS = {
    "analyze_project": ("cv_preprocess.application.analyze", "analyze_project"),
    "audit_dataset": ("cv_preprocess.application.audit", "audit_dataset"),
    "build_dataset": ("cv_preprocess.application.build", "build_dataset"),
    "materialize_dataset": ("cv_preprocess.application.materialize", "materialize_dataset"),
    "plan_dataset_split": ("cv_preprocess.application.split", "plan_dataset_split"),
    "scan_project": ("cv_preprocess.application.scan", "scan_project"),
    "select_dataset": ("cv_preprocess.application.select", "select_dataset"),
}


def __getattr__(name: str) -> Any:
    if name in _COMMON:
        from cv_preprocess.application import common as common_mod

        return getattr(common_mod, name)
    if name in _LAZY_FUNCS:
        module_name, attr = _LAZY_FUNCS[name]
        import importlib

        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
