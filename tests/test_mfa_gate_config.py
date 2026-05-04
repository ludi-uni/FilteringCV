from pathlib import Path

import pytest
import yaml

from cv_preprocess.config import PipelineConfig


def test_mfa_compare_requires_phonemize(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "text": {"phonemize": False},
                "mfa_gate": {"enabled": True, "compare_phones_to_g2p": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="phonemize"):
        PipelineConfig.from_yaml(p)


def test_mfa_disabled_without_phonemize_ok(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "text": {"phonemize": False},
                "mfa_gate": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.mfa_gate.enabled is False
