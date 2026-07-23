from __future__ import annotations

from typing import Any

__all__ = [
    "SelectionBackend",
    "PythonSelectionBackend",
    "greedy_local_search",
]


def __getattr__(name: str) -> Any:
    if name == "SelectionBackend":
        from cv_preprocess.selection.protocol import SelectionBackend

        return SelectionBackend
    if name in {"PythonSelectionBackend", "greedy_local_search"}:
        from cv_preprocess.selection import python_backend as pb

        return getattr(pb, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
