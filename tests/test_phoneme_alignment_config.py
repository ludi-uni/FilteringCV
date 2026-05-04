from pathlib import Path

import pytest
import yaml

from cv_preprocess.config import PipelineConfig


def test_alignment_check_requires_phonemize(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "text": {
                    "phonemize": False,
                    "phoneme_alignment_check": {
                        "enabled": True,
                        "manifest_path": str(tmp_path / "m.jsonl"),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "m.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="phonemize"):
        PipelineConfig.from_yaml(p)


def test_alignment_check_requires_manifest_when_enabled(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "text": {
                    "phoneme_alignment_check": {
                        "enabled": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest_path"):
        PipelineConfig.from_yaml(p)
