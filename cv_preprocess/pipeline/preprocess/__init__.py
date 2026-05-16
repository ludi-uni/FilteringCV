from __future__ import annotations

from typing import Any

from cv_preprocess.pipeline.preprocess.helpers import (
    _compute_clip_mora_count_once,
    _merged_quality_gate_for_mfa_prefilter,
    _merged_quality_gate_for_nfa_prefilter,
    _maybe_prefilter_final_gate_reuse_pair,
    _mora_gates_needed,
    effective_final_quality_gate,
)

__all__ = [
    "run_preprocess",
    "_compute_clip_mora_count_once",
    "_merged_quality_gate_for_mfa_prefilter",
    "_merged_quality_gate_for_nfa_prefilter",
    "_maybe_prefilter_final_gate_reuse_pair",
    "_mora_gates_needed",
    "effective_final_quality_gate",
]


def __getattr__(name: str) -> Any:
    """``run_preprocess`` だけ遅延 import（``session`` ↔ ``asr_gate_apply`` の循環を避ける）。"""
    if name == "run_preprocess":
        from cv_preprocess.pipeline.preprocess.run import run_preprocess as _rp

        return _rp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
