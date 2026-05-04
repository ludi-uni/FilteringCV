"""nfa_gate の設定検証。"""

from pathlib import Path

import pytest
import yaml

from cv_preprocess.config import PipelineConfig
from cv_preprocess.pipeline.preprocess import _merged_quality_gate_for_nfa_prefilter


def test_nfa_compare_requires_phonemize(tmp_path: Path) -> None:
    (tmp_path / "m.yaml").write_text("{}", encoding="utf-8")
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "text": {"phonemize": False},
                "nfa_gate": {
                    "enabled": True,
                    "compare_tokens_to_g2p": True,
                    "pretrained_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
                    "nfa_to_g2p_token_map_path": str(tmp_path / "m.yaml"),
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="phonemize"):
        PipelineConfig.from_yaml(p)


def test_nfa_compare_requires_token_map_path(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "text": {"phonemize": True},
                "nfa_gate": {
                    "enabled": True,
                    "compare_tokens_to_g2p": True,
                    "pretrained_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nfa_to_g2p_token_map_path"):
        PipelineConfig.from_yaml(p)


def test_mfa_and_nfa_cannot_both_enable(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "mfa_gate": {"enabled": True},
                "nfa_gate": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="同時"):
        PipelineConfig.from_yaml(p)


def test_nfa_enabled_requires_xor_model_source(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "nfa_gate": {
                    "enabled": True,
                    "pretrained_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
                    "model_path": str(tmp_path / "x.nemo"),
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pretrained_name"):
        PipelineConfig.from_yaml(p)


def test_nfa_prefilter_requires_nfa_enabled(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "nfa_gate": {"enabled": False, "prefilter": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prefilter"):
        PipelineConfig.from_yaml(p)


def test_nfa_compare_pred_requires_phonemize(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "text": {"phonemize": False},
                "nfa_gate": {
                    "enabled": True,
                    "pretrained_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
                    "align_using_pred_text": True,
                    "compare_pred_text_to_norm": True,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="phonemize"):
        PipelineConfig.from_yaml(p)


def test_nfa_pred_text_requires_align_using_pred(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "nfa_gate": {
                    "enabled": True,
                    "pretrained_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
                    "compare_pred_text_to_norm": True,
                    "align_using_pred_text": False,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="align_using_pred_text"):
        PipelineConfig.from_yaml(p)


def test_nfa_phoneme_and_transcript_compare_exclusive(tmp_path: Path) -> None:
    (tmp_path / "m.yaml").write_text("{}", encoding="utf-8")
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "text": {"phonemize": True},
                "nfa_gate": {
                    "enabled": True,
                    "pretrained_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
                    "align_using_pred_text": True,
                    "compare_pred_text_to_norm": True,
                    "compare_tokens_to_g2p": True,
                    "nfa_to_g2p_token_map_path": str(tmp_path / "m.yaml"),
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="compare_pred_text_to_norm"):
        PipelineConfig.from_yaml(p)


def test_merged_nfa_prefilter_overrides() -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": Path("CommonVoice")},
            "quality_gate": {"min_estimated_snr_db": 10.0},
            "nfa_gate": {
                "enabled": True,
                "prefilter": {
                    "enabled": True,
                    "quality_gate_overrides": {"min_estimated_snr_db": None},
                },
            },
        }
    )
    qg = _merged_quality_gate_for_nfa_prefilter(cfg)
    assert qg.min_estimated_snr_db is None
