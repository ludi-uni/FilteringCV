from __future__ import annotations

from pathlib import Path

from cv_preprocess.catalog.cache import pipeline_cache_key
from cv_preprocess.config import PipelineConfig


def _base_config(tmp_path: Path, **overrides: object) -> PipelineConfig:
    payload: dict[str, object] = {
        "input": {"corpus_root": tmp_path},
        "dataset_builder": {"enabled": True, "work_dir": tmp_path / "work"},
    }
    payload.update(overrides)
    return PipelineConfig.model_validate(payload)


def test_pipeline_cache_key_is_stable_for_same_config(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    first = pipeline_cache_key(cfg)
    second = pipeline_cache_key(cfg)
    assert first == second
    assert len(first) == 64


def test_pipeline_cache_key_changes_when_sample_rate_changes(tmp_path: Path) -> None:
    baseline = _base_config(
        tmp_path,
        audio_pipeline={"target_sample_rate": 22050, "steps": [{"type": "resample", "sr": 22050}]},
    )
    changed = _base_config(
        tmp_path,
        audio_pipeline={"target_sample_rate": 16000, "steps": [{"type": "resample", "sr": 16000}]},
    )
    assert pipeline_cache_key(baseline) != pipeline_cache_key(changed)
