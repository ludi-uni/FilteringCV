"""NFA 有効時の最終品質ゲート（``nfa_gate.quality_gate_overrides``）。"""

from pathlib import Path

from cv_preprocess.config import PipelineConfig
from cv_preprocess.pipeline.preprocess import effective_final_quality_gate


def test_effective_final_quality_gate_without_nfa_is_root() -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": Path("CommonVoice")},
            "quality_gate": {"min_estimated_snr_db": 10.0},
            "nfa_gate": {"enabled": False, "quality_gate_overrides": {"min_estimated_snr_db": None}},
        }
    )
    qg = effective_final_quality_gate(cfg)
    assert qg.min_estimated_snr_db == 10.0


def test_effective_final_quality_gate_nfa_overrides_merge() -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": Path("CommonVoice")},
            "quality_gate": {"min_estimated_snr_db": 10.0, "quality_tier_b_min_snr_db": 8.5},
            "nfa_gate": {
                "enabled": True,
                "quality_gate_overrides": {
                    "min_estimated_snr_db": None,
                    "quality_tier_b_min_snr_db": 6.0,
                },
            },
        }
    )
    qg = effective_final_quality_gate(cfg)
    assert qg.min_estimated_snr_db is None
    assert qg.quality_tier_b_min_snr_db == 6.0


def test_effective_final_quality_gate_nfa_profile_then_overrides() -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": Path("CommonVoice")},
            "quality_gate_profiles": {
                "relaxed_snr": {"min_estimated_snr_db": 7.0},
            },
            "nfa_gate": {
                "enabled": True,
                "quality_gate_profile": "relaxed_snr",
                "quality_gate_overrides": {"min_estimated_snr_db": None},
            },
        }
    )
    qg = effective_final_quality_gate(cfg)
    assert qg.min_estimated_snr_db is None
