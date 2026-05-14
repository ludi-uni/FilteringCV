"""asr_gate の設定検証。"""

from pathlib import Path

import pytest
import yaml

from cv_preprocess.config import PipelineConfig


def _minimal_input(tmp_path: Path) -> dict:
    return {"input": {"corpus_root": str(tmp_path / "ja")}}


def test_asr_compare_phonemes_requires_phonemize(tmp_path: Path) -> None:
    (tmp_path / "ja").mkdir()
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                **_minimal_input(tmp_path),
                "text": {"phonemize": False},
                "asr_gate": {
                    "enabled": True,
                    "backend": "mock",
                    "compare_phonemes": True,
                    "compare_text": False,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="phonemize"):
        PipelineConfig.from_yaml(p)


def test_asr_compare_both_false_invalid(tmp_path: Path) -> None:
    (tmp_path / "ja").mkdir()
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                **_minimal_input(tmp_path),
                "text": {"phonemize": True},
                "asr_gate": {
                    "enabled": True,
                    "backend": "mock",
                    "compare_phonemes": False,
                    "compare_text": False,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="compare_text"):
        PipelineConfig.from_yaml(p)


def test_asr_nemo_requires_xor_model_source(tmp_path: Path) -> None:
    (tmp_path / "ja").mkdir()
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                **_minimal_input(tmp_path),
                "asr_gate": {
                    "enabled": True,
                    "backend": "nemo_transcribe",
                    "pretrained_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
                    "model_path": str(tmp_path / "x.nemo"),
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pretrained_name"):
        PipelineConfig.from_yaml(p)


def test_asr_disabled_allows_invalid_nemo_combo(tmp_path: Path) -> None:
    (tmp_path / "ja").mkdir()
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                **_minimal_input(tmp_path),
                "asr_gate": {
                    "enabled": False,
                    "backend": "nemo_transcribe",
                    "pretrained_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
                    "model_path": str(tmp_path / "x.nemo"),
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.asr_gate.enabled is False


def test_asr_yaml_aliases_parakeet_and_model_name(tmp_path: Path) -> None:
    (tmp_path / "ja").mkdir()
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                **_minimal_input(tmp_path),
                "text": {"phonemize": True},
                "asr_gate": {
                    "enabled": False,
                    "backend": "parakeet",
                    "text_normalizer": "openjtalk_like",
                    "model_name": "nvidia/parakeet-tdt-0.6b-v3",
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.asr_gate.backend == "nemo_transcribe"
    assert cfg.asr_gate.text_normalizer == "normalize_for_tts"
    assert cfg.asr_gate.pretrained_name == "nvidia/parakeet-tdt-0.6b-v3"
