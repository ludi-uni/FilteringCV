from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ComputeConfig(BaseModel):
    backend: Literal["auto", "polars", "python"] = "auto"


class DatasetBuilderSplitConfig(BaseModel):
    protocol: Literal["unseen_speaker", "seen_speaker", "single_speaker"] = "unseen_speaker"
    train: float = 0.9
    val: float = 0.05
    test: float = 0.05
    seed: int = 42
    preserve_train: bool = True
    leakage_policy: Literal["strict", "warn", "off"] = "strict"
    optimizer: Literal["auto", "greedy_local_search", "ortools"] = "auto"

    @field_validator("train", "val", "test")
    @classmethod
    def non_negative_ratios(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("split ratios must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> DatasetBuilderSplitConfig:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split.train + val + test must equal 1.0, got {total}")
        return self


class LocalSearchConfig(BaseModel):
    enabled: bool = True
    max_iterations: int = 500
    max_wall_sec: float = 120.0
    swap_patterns: list[Literal["1v1", "1v2", "2v1"]] = Field(
        default_factory=lambda: ["1v1", "1v2", "2v1"]
    )

    @field_validator("max_iterations")
    @classmethod
    def non_negative_iterations(cls, value: int) -> int:
        if value < 0:
            raise ValueError("local_search.max_iterations must be >= 0")
        return value

    @field_validator("max_wall_sec")
    @classmethod
    def non_negative_wall_sec(cls, value: float) -> float:
        if value < 0:
            raise ValueError("local_search.max_wall_sec must be >= 0")
        return value


class SelectionConfig(BaseModel):
    reserve_ratio: float = 0.1
    feature_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "phone": 1.0,
            "biphone": 0.8,
            "triphone": 0.6,
            "mora": 1.0,
            "mora_bigram": 0.8,
            "full_context": 0.5,
            "accent_nucleus": 0.4,
            "accent_phrase_length": 0.3,
            "pause_boundary": 0.3,
            "sentence_length_band": 0.2,
            "speaking_rate_band": 0.2,
            "interrogative_declarative": 0.2,
        }
    )
    diminishing_return_tau: dict[str, float] = Field(default_factory=dict)
    local_search: LocalSearchConfig = Field(default_factory=LocalSearchConfig)

    @field_validator("reserve_ratio")
    @classmethod
    def valid_reserve_ratio(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("selection.reserve_ratio must be in [0, 1]")
        return value

    @field_validator("feature_weights", "diminishing_return_tau")
    @classmethod
    def non_negative_weights(cls, value: dict[str, float]) -> dict[str, float]:
        for key, weight in value.items():
            if weight < 0:
                raise ValueError(f"selection weight for {key!r} must be >= 0, got {weight}")
        return value


class SpeakerConstraintsConfig(BaseModel):
    max_clips_per_speaker: int | None = None
    max_duration_sec_per_speaker: float | None = None
    prefer_duration_over_clip_count: bool = True

    @field_validator("max_clips_per_speaker")
    @classmethod
    def non_negative_clip_cap(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("speaker_constraints.max_clips_per_speaker must be >= 0")
        return value

    @field_validator("max_duration_sec_per_speaker")
    @classmethod
    def non_negative_duration_cap(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("speaker_constraints.max_duration_sec_per_speaker must be >= 0")
        return value


class DuplicateGroupPolicyConfig(BaseModel):
    enabled: bool = True
    max_selected: int = 1
    penalty: float = 0.0

    @field_validator("max_selected")
    @classmethod
    def non_negative_max_selected(cls, value: int) -> int:
        if value < 0:
            raise ValueError("duplicates max_selected must be >= 0")
        return value

    @field_validator("penalty")
    @classmethod
    def non_negative_penalty(cls, value: float) -> float:
        if value < 0:
            raise ValueError("duplicates penalty must be >= 0")
        return value


class DuplicatesConfig(BaseModel):
    exact_audio: DuplicateGroupPolicyConfig = Field(default_factory=DuplicateGroupPolicyConfig)
    same_source_path: DuplicateGroupPolicyConfig = Field(
        default_factory=lambda: DuplicateGroupPolicyConfig(max_selected=1, penalty=1.0)
    )
    same_sentence_id: DuplicateGroupPolicyConfig = Field(default_factory=DuplicateGroupPolicyConfig)
    same_normalized_text: DuplicateGroupPolicyConfig = Field(default_factory=DuplicateGroupPolicyConfig)
    same_speaker_same_text: DuplicateGroupPolicyConfig = Field(default_factory=DuplicateGroupPolicyConfig)
    near_duplicate_text: DuplicateGroupPolicyConfig = Field(
        default_factory=lambda: DuplicateGroupPolicyConfig(enabled=False)
    )


class DistributionTemperatureConfig(BaseModel):
    phone: float = 1.0
    biphone: float = 1.0
    triphone: float = 1.0
    mora: float = 1.0
    mora_bigram: float = 1.0
    full_context: float = 1.0
    accent_nucleus: float = 1.0
    accent_phrase_length: float = 1.0
    pause_boundary: float = 1.0
    sentence_length_band: float = 1.0
    speaking_rate_band: float = 1.0
    interrogative_declarative: float = 1.0

    @model_validator(mode="after")
    def non_negative_temperatures(self) -> DistributionTemperatureConfig:
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"distribution_temperature.{field_name} must be >= 0")
        return self


class FeatureSupportConfig(BaseModel):
    min_pool_count: int = 2
    exclude_tokens: list[str] = Field(default_factory=lambda: ["sil", "pau"])
    down_weight_tokens: list[str] = Field(default_factory=list)

    @field_validator("min_pool_count")
    @classmethod
    def non_negative_min_pool_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("feature_support.min_pool_count must be >= 0")
        return value


class MaterializeConfig(BaseModel):
    mode: Literal["copy", "hardlink", "symlink"] = "copy"
    output_root: Path | None = None
    atomic_rename: bool = True
    emit_run_manifest: bool = True


class DatasetBuilderConfig(BaseModel):
    enabled: bool = False
    work_dir: Path = Path("work")
    target_duration_hours: float | None = None
    random_seed: int = 42
    split: DatasetBuilderSplitConfig = Field(default_factory=DatasetBuilderSplitConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    speaker_constraints: SpeakerConstraintsConfig = Field(default_factory=SpeakerConstraintsConfig)
    duplicates: DuplicatesConfig = Field(default_factory=DuplicatesConfig)
    distribution_temperature: DistributionTemperatureConfig = Field(
        default_factory=DistributionTemperatureConfig
    )
    feature_support: FeatureSupportConfig = Field(default_factory=FeatureSupportConfig)
    materialize: MaterializeConfig = Field(default_factory=MaterializeConfig)

    @field_validator("target_duration_hours")
    @classmethod
    def non_negative_target_duration(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("target_duration_hours must be >= 0")
        return value
