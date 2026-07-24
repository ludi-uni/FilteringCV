"""Rare phoneme / feature coverage automation."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_clip_index",
    "generate_report_from_run_dir",
    "load_index_jsonl",
    "load_run_state",
    "plan_coverage",
    "run_coverage",
    "save_run_state",
    "write_coverage_reports",
]

_LAZY = {
    "build_clip_index": ("cv_preprocess.coverage.indexer", "build_clip_index"),
    "load_index_jsonl": ("cv_preprocess.coverage.indexer", "load_index_jsonl"),
    "plan_coverage": ("cv_preprocess.coverage.planner", "plan_coverage"),
    "run_coverage": ("cv_preprocess.coverage.runner", "run_coverage"),
    "load_run_state": ("cv_preprocess.coverage.state", "load_run_state"),
    "save_run_state": ("cv_preprocess.coverage.state", "save_run_state"),
    "write_coverage_reports": ("cv_preprocess.coverage.report", "write_coverage_reports"),
    "generate_report_from_run_dir": ("cv_preprocess.coverage.report", "generate_report_from_run_dir"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
