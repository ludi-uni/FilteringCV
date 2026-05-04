from pathlib import Path

import pytest

from cv_preprocess.config import PipelineConfig


def test_quality_gate_profile_applies_base() -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": Path("CommonVoice")},
            "quality_gate_profiles": {"strict": {"min_estimated_snr_db": 22.0}},
            "quality_gate_profile": "strict",
            "quality_gate": {},
        }
    )
    assert cfg.quality_gate.min_estimated_snr_db == 22.0


def test_quality_gate_yaml_overrides_profile() -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": Path("CommonVoice")},
            "quality_gate_profiles": {"strict": {"min_estimated_snr_db": 22.0}},
            "quality_gate_profile": "strict",
            "quality_gate": {"min_estimated_snr_db": 10.0},
        }
    )
    assert cfg.quality_gate.min_estimated_snr_db == 10.0


def test_quality_gate_profile_missing_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        PipelineConfig.model_validate(
            {
                "input": {"corpus_root": Path("CommonVoice")},
                "quality_gate_profiles": {},
                "quality_gate_profile": "nope",
                "quality_gate": {},
            }
        )
