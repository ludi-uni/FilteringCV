from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ComputeConfig(BaseModel):
    backend: Literal["auto", "polars", "python"] = "auto"


class SplitRatiosConfig(BaseModel):
    train: float = 0.9
    validation: float | None = None
    val: float = 0.05
    test: float = 0.05

    @field_validator("train", "validation", "val", "test")
    @classmethod
    def non_negative_ratios(cls, value: float | None) -> float | None:
        if value is not None and (value < 0 or value > 1):
            raise ValueError("split ratios must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def resolve_validation_alias(self) -> SplitRatiosConfig:
        if self.validation is not None:
            object.__setattr__(self, "val", self.validation)
        return self

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> SplitRatiosConfig:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must equal 1.0, got {total}")
        return self

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}


class PreserveTrainConfig(BaseModel):
    enabled: bool = True
    critical_feature_max_speakers: int = 2
    min_train_occurrences: int = 1

    @field_validator("critical_feature_max_speakers", "min_train_occurrences")
    @classmethod
    def non_negative_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("preserve_train counts must be >= 0")
        return value


class LeakagePolicyConfig(BaseModel):
    speaker: Literal["forbid", "forbid_for_test", "allow"] = "forbid"
    audio_hash: Literal["forbid", "forbid_for_test", "allow"] = "forbid"
    sentence_id: Literal["forbid", "forbid_for_test", "allow"] = "forbid_for_test"
    normalized_text: Literal["forbid", "forbid_for_test", "allow"] = "forbid_for_test"


class SplitOptimizerConfig(BaseModel):
    backend: Literal["auto", "greedy_local_search", "ortools"] = "auto"
    time_limit_sec: float = 120.0
    fallback: Literal["greedy_local_search"] = "greedy_local_search"

    @field_validator("time_limit_sec")
    @classmethod
    def non_negative_time_limit(cls, value: float) -> float:
        if value < 0:
            raise ValueError("split.optimizer.time_limit_sec must be >= 0")
        return value


class DatasetBuilderSplitConfig(BaseModel):
    protocol: Literal["unseen_speaker", "seen_speaker", "single_speaker"] = "unseen_speaker"
    ratios: SplitRatiosConfig | None = None
    train: float = 0.9
    val: float = 0.05
    test: float = 0.05
    seed: int = 42
    preserve_train: PreserveTrainConfig | bool = Field(default_factory=PreserveTrainConfig)
    objectives: dict[str, float] = Field(default_factory=dict)
    leakage_policy: LeakagePolicyConfig | Literal["strict", "warn", "off"] = Field(
        default_factory=LeakagePolicyConfig
    )
    optimizer: SplitOptimizerConfig | Literal["auto", "greedy_local_search", "ortools"] = Field(
        default_factory=SplitOptimizerConfig
    )

    @field_validator("train", "val", "test")
    @classmethod
    def non_negative_ratios(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("split ratios must be in [0, 1]")
        return value

    @model_validator(mode="before")
    @classmethod
    def normalize_nested_aliases(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if "preserve_train" in data and isinstance(data["preserve_train"], bool):
            data["preserve_train"] = {"enabled": data["preserve_train"]}
        if "leakage_policy" in data and isinstance(data["leakage_policy"], str):
            legacy = data["leakage_policy"]
            if legacy == "strict":
                data["leakage_policy"] = LeakagePolicyConfig().model_dump()
            elif legacy == "warn":
                data["leakage_policy"] = LeakagePolicyConfig().model_dump()
            elif legacy == "off":
                data["leakage_policy"] = LeakagePolicyConfig(
                    speaker="allow",
                    audio_hash="allow",
                    sentence_id="allow",
                    normalized_text="allow",
                ).model_dump()
        if "optimizer" in data and isinstance(data["optimizer"], str):
            data["optimizer"] = {"backend": data["optimizer"]}
        return data

    @model_validator(mode="after")
    def resolve_ratio_aliases(self) -> DatasetBuilderSplitConfig:
        if self.ratios is not None:
            object.__setattr__(self, "train", self.ratios.train)
            object.__setattr__(self, "val", self.ratios.val)
            object.__setattr__(self, "test", self.ratios.test)
        return self

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> DatasetBuilderSplitConfig:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split.train + val + test must equal 1.0, got {total}")
        return self

    def resolved_ratios(self) -> dict[str, float]:
        if self.ratios is not None:
            return self.ratios.as_dict()
        return {"train": self.train, "val": self.val, "test": self.test}

    def resolved_preserve_train(self) -> PreserveTrainConfig:
        if isinstance(self.preserve_train, PreserveTrainConfig):
            return self.preserve_train
        return PreserveTrainConfig(enabled=bool(self.preserve_train))

    def resolved_optimizer(self) -> SplitOptimizerConfig:
        if isinstance(self.optimizer, SplitOptimizerConfig):
            return self.optimizer
        return SplitOptimizerConfig(backend=self.optimizer)

class LocalSearchConfig(BaseModel):
    enabled: bool = True
    max_iterations: int = 500
    max_wall_sec: float = 120.0
    max_seconds: float | None = None
    # 1v1 is the useful default; 1v2/2v1 are combinatorial and wall-clock limited.
    swap_patterns: list[Literal["1v1", "1v2", "2v1"]] = Field(
        default_factory=lambda: ["1v1"]
    )

    @field_validator("max_iterations")
    @classmethod
    def non_negative_iterations(cls, value: int) -> int:
        if value < 0:
            raise ValueError("local_search.max_iterations must be >= 0")
        return value

    @field_validator("max_wall_sec", "max_seconds")
    @classmethod
    def non_negative_wall_sec(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("local_search wall-time limits must be >= 0")
        return value

    @model_validator(mode="after")
    def resolve_max_seconds_alias(self) -> LocalSearchConfig:
        if self.max_seconds is not None:
            object.__setattr__(self, "max_wall_sec", self.max_seconds)
        return self


class QualitySelectionConfig(BaseModel):
    hard_min_score: float | None = None
    preferred_score: float | None = None
    max_low_quality_ratio: float | None = None


class DurationSelectionConfig(BaseModel):
    target_hours: float | None = None
    tolerance_ratio: float = 0.1
    cost_exponent: float = 1.0

    @field_validator("tolerance_ratio")
    @classmethod
    def valid_tolerance_ratio(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("selection.duration.tolerance_ratio must be in [0, 1]")
        return value


class CoverageConstraintsConfig(BaseModel):
    """Connect Force Build coverage targets to final select guarantees."""

    #: On by default: select reserves clips for ``coverage.features`` minima.
    enabled: bool = True
    #: Alias accepted in YAML; mirrored onto ``violation_policy``.
    policy: Literal["fail", "warn", "best_effort"] | None = None
    violation_policy: Literal["fail", "warn", "best_effort"] = "best_effort"
    use_coverage_targets: bool = True
    required_families: list[str] = Field(default_factory=lambda: ["phoneme", "mora"])
    optional_families: list[str] = Field(default_factory=lambda: ["biphone"])
    preserve_during_local_search: bool = True
    quality_weight: float = 0.1
    diversity_weight: float = 0.05
    duration_penalty_weight: float = 0.01
    required_weight_default: float = 1.0
    optional_weight_default: float = 0.35

    @model_validator(mode="after")
    def resolve_policy_alias(self) -> CoverageConstraintsConfig:
        if self.policy is not None:
            object.__setattr__(self, "violation_policy", self.policy)
        return self


class AcousticDiversityConfig(BaseModel):
    """Lightweight acoustic redundancy control (no heavy embedding downloads)."""

    #: On by default; set ``enabled: false`` or use ``--disable-acoustic-diversity`` to turn off.
    enabled: bool = True
    backend: Literal["lightweight", "disabled", "wavlm", "hubert", "wav2vec2", "external"] = (
        "lightweight"
    )
    weight: float = 0.15
    features: list[str] = Field(
        default_factory=lambda: [
            "duration",
            "rms",
            "snr",
            "silence_ratio",
            "quality_score",
        ]
    )
    missing_value_policy: Literal["ignore", "zero"] = "ignore"

    @field_validator("weight")
    @classmethod
    def non_negative_weight(cls, value: float) -> float:
        if value < 0:
            raise ValueError("acoustic_diversity.weight must be >= 0")
        return value

    @model_validator(mode="after")
    def backend_available(self) -> AcousticDiversityConfig:
        heavy = {"wavlm", "hubert", "wav2vec2", "external"}
        if self.backend in heavy:
            raise ValueError(
                f"acoustic_diversity.backend={self.backend!r} is reserved for future use; "
                "use 'lightweight' or 'disabled' in this release"
            )
        return self


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
            "speaker_diversity": 0.3,
            "acoustic_diversity": 0.0,
            "quality": 0.2,
        }
    )
    weights: dict[str, float] = Field(default_factory=dict)
    diminishing_return_tau: dict[str, float] = Field(default_factory=dict)
    quality: QualitySelectionConfig = Field(default_factory=QualitySelectionConfig)
    duration: DurationSelectionConfig = Field(default_factory=DurationSelectionConfig)
    local_search: LocalSearchConfig = Field(default_factory=LocalSearchConfig)
    coverage_constraints: CoverageConstraintsConfig = Field(
        default_factory=CoverageConstraintsConfig
    )
    acoustic_diversity: AcousticDiversityConfig = Field(default_factory=AcousticDiversityConfig)

    @model_validator(mode="before")
    @classmethod
    def merge_weight_aliases(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        weights = data.get("weights") or {}
        feature_weights = dict(data.get("feature_weights") or {})
        if weights:
            feature_weights.update(weights)
            data["feature_weights"] = feature_weights
        return data

    @field_validator("reserve_ratio")
    @classmethod
    def valid_reserve_ratio(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("selection.reserve_ratio must be in [0, 1]")
        return value

    @field_validator("feature_weights", "diminishing_return_tau", "weights")
    @classmethod
    def non_negative_weights(cls, value: dict[str, float]) -> dict[str, float]:
        for key, weight in value.items():
            if weight < 0:
                raise ValueError(f"selection weight for {key!r} must be >= 0, got {weight}")
        return value


class SpeakerConstraintsConfig(BaseModel):
    max_clips_per_speaker: int | None = None
    max_clips: int | None = None
    max_duration_sec_per_speaker: float | None = None
    max_duration_minutes: float | None = None
    min_duration_minutes: float | None = None
    prefer_duration_over_clip_count: bool = True

    @model_validator(mode="after")
    def resolve_aliases(self) -> SpeakerConstraintsConfig:
        if self.max_clips is not None and self.max_clips_per_speaker is None:
            object.__setattr__(self, "max_clips_per_speaker", self.max_clips)
        if self.max_duration_minutes is not None and self.max_duration_sec_per_speaker is None:
            object.__setattr__(
                self, "max_duration_sec_per_speaker", self.max_duration_minutes * 60.0
            )
        return self

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
    phone: float = 0.80
    biphone: float = 0.65
    triphone: float = 0.55
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
    min_utterances: dict[str, int] = Field(default_factory=dict)
    min_speakers: dict[str, int] = Field(default_factory=dict)
    exclude_tokens: list[str] = Field(default_factory=lambda: ["sil", "pau"])
    down_weight_tokens: list[str] = Field(default_factory=list)

    @field_validator("min_pool_count")
    @classmethod
    def non_negative_min_pool_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("feature_support.min_pool_count must be >= 0")
        return value


class PiperPlusExportConfig(BaseModel):
    sample_rate: int = 22050
    wav_dirname: str = "wav"

    @field_validator("sample_rate")
    @classmethod
    def positive_rate(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("piper_plus.sample_rate must be > 0")
        return value


class StyleBertVits2ExportConfig(BaseModel):
    sample_rate: int = 44100
    model_name: str = "filteringcv"
    language: str = "JP"
    packaging: Literal["single_model", "per_speaker"] = "single_model"

    @field_validator("sample_rate")
    @classmethod
    def positive_rate(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("style_bert_vits2.sample_rate must be > 0")
        return value

    @field_validator("model_name")
    @classmethod
    def non_empty_model(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("style_bert_vits2.model_name must be non-empty")
        return str(value).strip()


class TrainerExportsConfig(BaseModel):
    """Write piper_plus / Style-Bert-VITS2 packs under materialize ``exports/``."""

    enabled: bool = True
    formats: list[Literal["piper_plus", "style_bert_vits2"]] = Field(
        default_factory=lambda: ["piper_plus", "style_bert_vits2"]
    )
    text_field: Literal["text_norm", "text_raw"] = "text_norm"
    resample: bool = False
    #: When null, inherit ``materialize.mode``.
    mode: Literal["copy", "hardlink", "symlink"] | None = None
    piper_plus: PiperPlusExportConfig = Field(default_factory=PiperPlusExportConfig)
    style_bert_vits2: StyleBertVits2ExportConfig = Field(default_factory=StyleBertVits2ExportConfig)

    @field_validator("formats")
    @classmethod
    def non_empty_formats(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("trainer_exports.formats must not be empty when used")
        return value


class MaterializeConfig(BaseModel):
    mode: Literal["copy", "hardlink", "symlink"] = "copy"
    output_root: Path | None = None
    atomic_rename: bool = True
    emit_run_manifest: bool = True
    trainer_exports: TrainerExportsConfig = Field(default_factory=TrainerExportsConfig)


class DatasetBuilderConfig(BaseModel):
    enabled: bool = False
    work_dir: Path = Path("work")
    output_dir: Path | None = None
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

    @model_validator(mode="after")
    def resolve_duration_target(self) -> DatasetBuilderConfig:
        if self.target_duration_hours is None and self.selection.duration.target_hours is not None:
            object.__setattr__(self, "target_duration_hours", self.selection.duration.target_hours)
        return self

    @field_validator("target_duration_hours")
    @classmethod
    def non_negative_target_duration(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("target_duration_hours must be >= 0")
        return value
