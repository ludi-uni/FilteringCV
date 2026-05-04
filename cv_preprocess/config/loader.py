from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from cv_preprocess.config.pipeline import PipelineConfig


class CLISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CV_", extra="ignore")


def load_config(path: Path) -> PipelineConfig:
    return PipelineConfig.from_yaml(path)
