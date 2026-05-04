"""mfa_gate.prefilter の設定検証とマージ。"""

from pathlib import Path

import pytest
import yaml

from cv_preprocess.config import PipelineConfig
from cv_preprocess.pipeline.preprocess import _merged_quality_gate_for_mfa_prefilter


def test_prefilter_requires_mfa_enabled(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "mfa_gate": {"enabled": False, "prefilter": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prefilter"):
        PipelineConfig.from_yaml(p)


def test_merged_prefilter_overrides_snr() -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": Path("CommonVoice")},
            "quality_gate": {"min_estimated_snr_db": 10.0},
            "mfa_gate": {
                "enabled": True,
                "prefilter": {
                    "enabled": True,
                    "quality_gate_overrides": {"min_estimated_snr_db": None},
                },
            },
        }
    )
    qg = _merged_quality_gate_for_mfa_prefilter(cfg)
    assert qg.min_estimated_snr_db is None
