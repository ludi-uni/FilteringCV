from pathlib import Path

import pytest

from cv_preprocess.config import PipelineConfig


def _base() -> dict:
    return {
        "input": {"corpus_root": Path("/tmp")},
        "output": {"root": Path("/tmp/out")},
        "two_pass_denoise": {"enabled": True},
        "audio_pipeline": {
            "target_sample_rate": 48000,
            "audio_pipeline_id": "legacy",
            "steps": [{"type": "resample", "sr": 48000}],
        },
    }


def test_split_requires_both_align_and_enhance() -> None:
    d = _base()
    d["audio_pipeline_align"] = {
        "target_sample_rate": 48000,
        "audio_pipeline_id": "a",
        "steps": [{"type": "resample", "sr": 48000}],
    }
    with pytest.raises(ValueError, match="片方だけ"):
        PipelineConfig.model_validate(d)


def test_split_requires_two_pass_enabled() -> None:
    d = _base()
    d["two_pass_denoise"] = {"enabled": False}
    d["audio_pipeline_align"] = {
        "target_sample_rate": 48000,
        "audio_pipeline_id": "a",
        "steps": [{"type": "resample", "sr": 48000}],
    }
    d["audio_pipeline_enhance"] = {
        "target_sample_rate": 48000,
        "audio_pipeline_id": "e",
        "steps": [{"type": "resample", "sr": 48000}],
    }
    with pytest.raises(ValueError, match="two_pass_denoise.enabled"):
        PipelineConfig.model_validate(d)


def test_split_requires_matching_target_sr() -> None:
    d = _base()
    d["audio_pipeline_align"] = {
        "target_sample_rate": 48000,
        "audio_pipeline_id": "a",
        "steps": [{"type": "resample", "sr": 48000}],
    }
    d["audio_pipeline_enhance"] = {
        "target_sample_rate": 22050,
        "audio_pipeline_id": "e",
        "steps": [{"type": "resample", "sr": 22050}],
    }
    with pytest.raises(ValueError, match="target_sample_rate"):
        PipelineConfig.model_validate(d)


def test_enhance_phase_after_align_requires_two_pass() -> None:
    d = _base()
    d["two_pass_denoise"] = {"enabled": False, "enhance_phase": "after_align_complete"}
    with pytest.raises(ValueError, match="after_align_complete requires"):
        PipelineConfig.model_validate(d)


def test_enhance_phase_after_align_ok_with_two_pass() -> None:
    d = _base()
    d["two_pass_denoise"] = {"enabled": True, "enhance_phase": "after_align_complete"}
    d["audio_pipeline_align"] = {
        "target_sample_rate": 48000,
        "audio_pipeline_id": "a",
        "steps": [{"type": "resample", "sr": 48000}],
    }
    d["audio_pipeline_enhance"] = {
        "target_sample_rate": 48000,
        "audio_pipeline_id": "e",
        "steps": [{"type": "resample", "sr": 48000}],
    }
    cfg = PipelineConfig.model_validate(d)
    assert cfg.two_pass_denoise.enhance_phase == "after_align_complete"


def test_split_ok_when_two_pass_and_both_pipelines() -> None:
    d = _base()
    d["audio_pipeline_align"] = {
        "target_sample_rate": 48000,
        "audio_pipeline_id": "a",
        "steps": [{"type": "resample", "sr": 48000}],
    }
    d["audio_pipeline_enhance"] = {
        "target_sample_rate": 48000,
        "audio_pipeline_id": "e",
        "steps": [{"type": "resample", "sr": 48000}],
    }
    cfg = PipelineConfig.model_validate(d)
    assert cfg.audio_pipeline_align is not None
    assert cfg.audio_pipeline_enhance is not None
