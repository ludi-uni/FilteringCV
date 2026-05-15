from pathlib import Path

from cv_preprocess.config import PipelineConfig, load_config


def test_example_yaml_loads() -> None:
    p = Path(__file__).resolve().parents[1] / "config" / "example.yaml"
    cfg = load_config(p)
    assert cfg.input.locale_expected == "ja"
    assert cfg.audio_pipeline_align is not None
    assert cfg.audio_pipeline_align.steps[0].type == "decode"
    assert cfg.audio_pipeline_enhance is not None
    enh_types = [s.type for s in cfg.audio_pipeline_enhance.steps]
    assert enh_types[0] == "decode"
    assert enh_types[1] == "sidon_restore"


def test_include_client_ids_scalar_coerced_to_list() -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": Path("CommonVoice")},
            "speakers": {"include_client_ids": "abc123hash"},
        }
    )
    assert cfg.speakers.include_client_ids == ["abc123hash"]


def test_input_max_clips_optional() -> None:
    cfg = PipelineConfig.model_validate({"input": {"corpus_root": Path("CommonVoice")}})
    assert cfg.input.max_clips is None
    cfg2 = PipelineConfig.model_validate(
        {"input": {"corpus_root": Path("CommonVoice"), "max_clips": 20}}
    )
    assert cfg2.input.max_clips == 20


def test_default_yaml_loads() -> None:
    p = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    cfg = load_config(p)
    assert cfg.input.max_clips is None or cfg.input.max_clips >= 1
    assert cfg.audio_pipeline_enhance is not None
    enh_types = [s.type for s in cfg.audio_pipeline_enhance.steps]
    assert enh_types[0] == "decode"
    assert enh_types[1] == "sidon_restore"
    assert cfg.quality_gate.sidon_after_enhance_split.enabled is True
    # 既定は two_pass: 無音トリムは align 側（enhance は decode 注入後に Sidon 等のみのことが多い）
    assert cfg.audio_pipeline_align is not None
    align_types = [s.type for s in cfg.audio_pipeline_align.steps]
    assert align_types.count("trim_silence") >= 1
