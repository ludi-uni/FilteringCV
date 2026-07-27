"""Trainer-format export package (piper_plus / Style-Bert-VITS2)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ExportResult",
    "run_trainer_exports",
    "export_from_materialize_root",
]


def __getattr__(name: str) -> Any:
    if name in {"ExportResult", "run_trainer_exports", "export_from_materialize_root"}:
        from cv_preprocess.export import runner as _runner

        return getattr(_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
