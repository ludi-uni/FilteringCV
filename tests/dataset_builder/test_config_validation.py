from __future__ import annotations

from pathlib import Path

import pytest

from cv_preprocess.config import PipelineConfig, load_config
from cv_preprocess.config.dataset_builder import DatasetBuilderConfig, DatasetBuilderSplitConfig


def test_default_pipeline_config_has_dataset_builder_disabled() -> None:
    cfg = PipelineConfig.model_validate({"input": {"corpus_root": Path("CommonVoice")}})
    assert cfg.schema_version == 1
    assert cfg.dataset_builder.enabled is False
    assert cfg.dataset_builder.work_dir == Path("work")
    assert cfg.compute.backend == "auto"


def test_example_yaml_is_schema_v2_builder_template() -> None:
    path = Path(__file__).resolve().parents[2] / "config" / "example.yaml"
    cfg = load_config(path)
    assert cfg.schema_version == 2
    assert cfg.dataset_builder.enabled is True
    assert cfg.two_pass_denoise.enabled is True
    assert cfg.audio_pipeline_enhance is not None


def test_negative_selection_weight_fails() -> None:
    with pytest.raises(ValueError, match="weight"):
        DatasetBuilderConfig.model_validate(
            {
                "selection": {
                    "feature_weights": {"phone": -0.1},
                }
            }
        )


def test_dataset_builder_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must equal 1.0"):
        DatasetBuilderSplitConfig.model_validate(
            {
                "train": 0.8,
                "val": 0.1,
                "test": 0.05,
            }
        )


def test_negative_target_duration_hours_fails() -> None:
    with pytest.raises(ValueError, match="target_duration_hours"):
        DatasetBuilderConfig.model_validate({"target_duration_hours": -1.0})
