"""Coverage automation configuration (rare phoneme / feature targeting)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

KNOWN_FEATURE_FAMILIES = frozenset(
    {
        "phoneme",
        "phone",  # alias → normalized to phoneme
        "mora",
        "biphone",
        "positioned_phoneme",
        # reserved for future extension
        "accent_phrase",
        "pitch_accent",
        "triphone",
        "speaker_feature",
        "phoneme_speaker",
    }
)

IMPLEMENTED_FEATURE_FAMILIES = frozenset(
    {"phoneme", "mora", "biphone", "positioned_phoneme"}
)

CountingMode = Literal["per_clip", "occurrence", "per_speaker"]


class FeatureTargetSpec(BaseModel):
    """Per-feature coverage goal.

    Bare integers in YAML (``v: 5``) are normalized to
    ``minimum == desired == 5`` so Force Build targets remain hard requirements
    for coverage-aware select.
    """

    minimum: int = 0
    desired: int = 0

    @model_validator(mode="before")
    @classmethod
    def coerce_int(cls, data: Any) -> Any:
        if isinstance(data, bool):
            raise ValueError("feature target must be an int or mapping, not bool")
        if isinstance(data, int):
            return {"minimum": data, "desired": data}
        if isinstance(data, dict):
            payload = dict(data)
            has_min = "minimum" in payload
            has_des = "desired" in payload
            if has_min and not has_des:
                payload["desired"] = payload["minimum"]
            elif has_des and not has_min:
                payload["minimum"] = payload["desired"]
            return payload
        return data

    @model_validator(mode="after")
    def non_negative_and_ordered(self) -> FeatureTargetSpec:
        if self.minimum < 0 or self.desired < 0:
            raise ValueError("feature target minimum/desired must be >= 0")
        if self.desired < self.minimum:
            object.__setattr__(self, "desired", self.minimum)
        return self


class FeatureFamilyTargetConfig(BaseModel):
    enabled: bool = True
    default_target: int = 0
    targets: dict[str, FeatureTargetSpec] = Field(default_factory=dict)
    required: bool = True

    @field_validator("default_target")
    @classmethod
    def non_negative_default(cls, value: int) -> int:
        if value < 0:
            raise ValueError("default_target must be >= 0")
        return value

    @field_validator("targets", mode="before")
    @classmethod
    def coerce_target_values(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        out: dict[str, Any] = {}
        for key, raw in value.items():
            if isinstance(raw, bool):
                raise ValueError(f"target for {key!r} must be an int or mapping, not bool")
            if isinstance(raw, int):
                out[key] = {"minimum": raw, "desired": raw}
            else:
                out[key] = raw
        return out


class PassProbabilityConfig(BaseModel):
    default: float = 0.5
    prior_strength: float = 10.0
    min_probability: float = 0.05
    max_probability: float = 0.95

    @field_validator("default", "min_probability", "max_probability")
    @classmethod
    def unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("probability values must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def min_le_max(self) -> PassProbabilityConfig:
        if self.min_probability > self.max_probability:
            raise ValueError("min_probability must be <= max_probability")
        if not (self.min_probability <= self.default <= self.max_probability):
            raise ValueError("default pass probability must lie between min and max")
        if self.prior_strength < 0:
            raise ValueError("prior_strength must be >= 0")
        return self


class AnalysisCostConfig(BaseModel):
    base: float = 1.0
    duration_weight: float = 0.1

    @field_validator("base", "duration_weight")
    @classmethod
    def non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("analysis_cost weights must be >= 0")
        return value


class DiversityConfig(BaseModel):
    speaker_penalty: float = 0.5
    duplicate_text_penalty: float = 1.0
    max_per_speaker_per_batch: int = 5

    @field_validator("speaker_penalty", "duplicate_text_penalty")
    @classmethod
    def non_negative_penalty(cls, value: float) -> float:
        if value < 0:
            raise ValueError("diversity penalties must be >= 0")
        return value

    @field_validator("max_per_speaker_per_batch")
    @classmethod
    def positive_max(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_per_speaker_per_batch must be >= 1")
        return value


class BatchConfig(BaseModel):
    min_size: int = 20
    max_size: int = 500
    safety_factor: float = 1.3

    @model_validator(mode="after")
    def sizes_ok(self) -> BatchConfig:
        if self.min_size < 1:
            raise ValueError("batch.min_size must be >= 1")
        if self.max_size < 1:
            raise ValueError("batch.max_size must be >= 1")
        if self.min_size > self.max_size:
            raise ValueError("batch.min_size must be <= batch.max_size")
        if self.safety_factor < 1.0:
            raise ValueError("batch.safety_factor must be >= 1.0")
        return self


class CoverageLimitsConfig(BaseModel):
    max_iterations: int = 50
    max_analyzed_clips: int = 10_000
    max_audio_hours: float = 100.0

    @field_validator("max_iterations", "max_analyzed_clips")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("coverage limits max_iterations/max_analyzed_clips must be >= 1")
        return value

    @field_validator("max_audio_hours")
    @classmethod
    def positive_hours(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("max_audio_hours must be > 0")
        return value


class RareRescueConfig(BaseModel):
    enabled: bool = True
    target_features: list[str] = Field(default_factory=list)
    min_candidate_clips: int = 1
    min_candidate_speakers: int = 1
    stricter_quality_gate: bool = True

    @field_validator("min_candidate_clips", "min_candidate_speakers")
    @classmethod
    def positive_min(cls, value: int) -> int:
        if value < 1:
            raise ValueError("rare_rescue minima must be >= 1")
        return value


class CoverageAutomationConfig(BaseModel):
    """Rare-feature coverage automation. Disabled by default (no effect on existing runs)."""

    enabled: bool = False
    #: When true (default), ``build`` / recommended GUI flow runs coverage **before** full analyze
    #: so only high-utility clips are quality-analyzed first; remaining clips reuse those results.
    insert_before_analyze: bool = True
    #: Directory for index / plan / active run (relative paths resolve from process cwd / project root).
    output_dir: Path = Path("output/coverage")
    #: Fixed run directory name used by GUI jobs (resume-friendly).
    active_run_dirname: str = "active-run"
    counting_mode: CountingMode = "per_clip"
    features: dict[str, FeatureFamilyTargetConfig] = Field(default_factory=dict)
    required_features: list[str] = Field(default_factory=list)
    optional_features: list[str] = Field(default_factory=list)
    pass_probability: PassProbabilityConfig = Field(default_factory=PassProbabilityConfig)
    analysis_cost: AnalysisCostConfig = Field(default_factory=AnalysisCostConfig)
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    limits: CoverageLimitsConfig = Field(default_factory=CoverageLimitsConfig)
    rare_rescue: RareRescueConfig = Field(default_factory=RareRescueConfig)
    target_weight_default: float = 1.0
    required_weight_bonus: float = 1.5

    @field_validator("counting_mode")
    @classmethod
    def counting_mode_ok(cls, value: str) -> str:
        allowed = {"per_clip", "occurrence", "per_speaker"}
        if value not in allowed:
            raise ValueError(f"counting_mode must be one of {sorted(allowed)}")
        if value != "per_clip":
            # Initial implementation supports per_clip only; keep config for future.
            pass
        return value

    @model_validator(mode="before")
    @classmethod
    def normalize_feature_family_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        features = data.get("features")
        if isinstance(features, dict) and "phone" in features and "phoneme" not in features:
            features = dict(features)
            features["phoneme"] = features.pop("phone")
            data = dict(data)
            data["features"] = features
        return data

    @model_validator(mode="after")
    def validate_feature_defs(self) -> CoverageAutomationConfig:
        for family in self.features:
            if family not in KNOWN_FEATURE_FAMILIES:
                raise ValueError(f"unknown feature family: {family!r}")
            if family == "phone":
                raise ValueError("internal error: phone alias should be normalized to phoneme")
        seen: set[str] = set()
        for key in list(self.required_features) + list(self.optional_features):
            if key in seen:
                raise ValueError(f"duplicate feature definition: {key!r}")
            seen.add(key)
            family = key.split(":", 1)[0]
            if family not in KNOWN_FEATURE_FAMILIES:
                raise ValueError(f"unknown feature family in {key!r}")
        for key in self.required_features:
            if self.resolve_target(key) is None:
                raise ValueError(f"required feature {key!r} has no target (set features.*.targets or default_target)")
        return self

    def resolve_target_spec(self, feature_key: str) -> FeatureTargetSpec | None:
        """Return minimum/desired for ``family:token`` (phone → phoneme)."""
        family, _, rest = feature_key.partition(":")
        if family == "phone":
            family = "phoneme"
            rest = feature_key.partition(":")[2]
        cfg = self.features.get(family)
        if cfg is None or not cfg.enabled:
            return None
        if rest in cfg.targets:
            return cfg.targets[rest]
        if cfg.default_target > 0:
            return FeatureTargetSpec(minimum=cfg.default_target, desired=cfg.default_target)
        return None

    def resolve_target(self, feature_key: str) -> int | None:
        """Return desired target count for ``family:token`` (coverage-run compatible)."""
        spec = self.resolve_target_spec(feature_key)
        if spec is None:
            return None
        return int(spec.desired)

    def is_required(self, feature_key: str) -> bool:
        if feature_key in self.required_features:
            return True
        if feature_key in self.optional_features:
            return False
        family = feature_key.split(":", 1)[0]
        if family == "phone":
            family = "phoneme"
        cfg = self.features.get(family)
        if cfg is None:
            return True
        return bool(cfg.required)

    def iter_active_targets(self) -> dict[str, int]:
        """Build ``feature_key -> target`` for enabled families."""
        out: dict[str, int] = {}
        for family, cfg in self.features.items():
            if not cfg.enabled:
                continue
            if family not in IMPLEMENTED_FEATURE_FAMILIES:
                continue
            if cfg.targets:
                for token, spec in cfg.targets.items():
                    desired = int(spec.desired)
                    if desired > 0 or int(spec.minimum) > 0:
                        out[f"{family}:{token}"] = max(desired, int(spec.minimum))
            elif cfg.default_target > 0:
                # default_target alone without explicit targets means "track pool tokens later";
                # planner uses explicit keys only when targets map is empty.
                pass
        for key in self.required_features + self.optional_features:
            target = self.resolve_target(key)
            if target is not None and target > 0:
                out[key if not key.startswith("phone:") else "phoneme:" + key.partition(":")[2]] = int(target)
        return out

    def iter_active_target_specs(self) -> dict[str, FeatureTargetSpec]:
        """Build ``feature_key -> FeatureTargetSpec`` for enabled families."""
        out: dict[str, FeatureTargetSpec] = {}
        for family, cfg in self.features.items():
            if not cfg.enabled:
                continue
            if family not in IMPLEMENTED_FEATURE_FAMILIES:
                continue
            if cfg.targets:
                for token, spec in cfg.targets.items():
                    if int(spec.minimum) > 0 or int(spec.desired) > 0:
                        out[f"{family}:{token}"] = spec
            elif cfg.default_target > 0:
                pass
        for key in self.required_features + self.optional_features:
            normalized = key if not key.startswith("phone:") else "phoneme:" + key.partition(":")[2]
            spec = self.resolve_target_spec(normalized)
            if spec is not None and (spec.minimum > 0 or spec.desired > 0):
                out[normalized] = spec
        return out
