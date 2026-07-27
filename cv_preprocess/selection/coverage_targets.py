"""Coverage target resolution for coverage-aware select."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from cv_preprocess.config.coverage import CoverageAutomationConfig, FeatureTargetSpec
from cv_preprocess.config.dataset_builder import CoverageConstraintsConfig, SelectionConfig
from cv_preprocess.selection.coverage_keys import normalize_feature_key, parse_feature_key
from cv_preprocess.selection.protocol import ClipFeatures

CoverageReachabilityStatus = Literal[
    "configured",
    "corpus_limited",
    "selection_limited",
    "satisfied",
    "unsatisfied",
]

AuditStatus = Literal[
    "satisfied",
    "corpus_limit_satisfied",
    "configured_target_satisfied",
    "minimum_satisfied",
    "desired_satisfied",
    "minimum_unsatisfied",
    "desired_unsatisfied",
    "selection_constraint_conflict",
    "not_present_in_index",
    "not_present_in_eligible",
    "candidate_missing",
]


@dataclass(frozen=True)
class FeatureCoverageTarget:
    feature_key: str
    feature_family: str
    feature_token: str
    configured_minimum: int
    configured_desired: int
    required: bool
    weight: float
    eligible_clip_count: int = 0
    eligible_unique_speaker_count: int = 0
    index_candidate_count: int | None = None
    effective_minimum: int = 0
    effective_desired: int = 0


@dataclass
class SelectionCoverageConstraints:
    enabled: bool
    violation_policy: str
    preserve_during_local_search: bool
    targets: dict[str, FeatureCoverageTarget] = field(default_factory=dict)
    required_keys: set[str] = field(default_factory=set)
    optional_keys: set[str] = field(default_factory=set)
    feature_to_clip_ids: dict[str, set[str]] = field(default_factory=dict)
    clip_coverage_keys: dict[str, set[str]] = field(default_factory=dict)
    quality_weight: float = 0.1
    diversity_weight: float = 0.05
    duration_penalty_weight: float = 0.01

    def effective_required_minimums(self) -> dict[str, int]:
        return {
            key: target.effective_minimum
            for key, target in self.targets.items()
            if key in self.required_keys and target.effective_minimum > 0
        }


def _family_allowed(family: str, families: Sequence[str]) -> bool:
    normalized = {("phoneme" if f == "phone" else f) for f in families}
    return family in normalized or family == "phone" and "phoneme" in normalized


def _clip_keys(clip: ClipFeatures) -> set[str]:
    if clip.coverage_keys:
        return set(clip.coverage_keys)
    return set()


def build_feature_to_clip_index(
    candidates: Sequence[ClipFeatures],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    feature_to_clip_ids: dict[str, set[str]] = defaultdict(set)
    clip_coverage_keys: dict[str, set[str]] = {}
    for clip in candidates:
        keys = _clip_keys(clip)
        clip_coverage_keys[clip.clip_id] = keys
        for key in keys:
            feature_to_clip_ids[normalize_feature_key(key)].add(clip.clip_id)
    return dict(feature_to_clip_ids), clip_coverage_keys


def compute_effective_targets(
    *,
    configured_minimum: int,
    configured_desired: int,
    eligible_clip_count: int,
) -> tuple[int, int]:
    effective_minimum = min(max(0, configured_minimum), max(0, eligible_clip_count))
    effective_desired = min(max(0, configured_desired), max(0, eligible_clip_count))
    if effective_desired < effective_minimum:
        effective_desired = effective_minimum
    return effective_minimum, effective_desired


def build_selection_coverage_constraints(
    coverage_config: CoverageAutomationConfig,
    selection_config: SelectionConfig,
    eligible_catalog: Sequence[ClipFeatures],
    *,
    index_candidate_counts: Mapping[str, int] | None = None,
) -> SelectionCoverageConstraints:
    """Adapter: reuse coverage YAML targets as select-side constraints."""
    cc = selection_config.coverage_constraints
    feature_to_clip_ids, clip_coverage_keys = build_feature_to_clip_index(eligible_catalog)

    speaker_by_feature: dict[str, set[str]] = defaultdict(set)
    clips_by_id = {c.clip_id: c for c in eligible_catalog}
    for feature, clip_ids in feature_to_clip_ids.items():
        for clip_id in clip_ids:
            clip = clips_by_id.get(clip_id)
            if clip is not None:
                speaker_by_feature[feature].add(clip.speaker_id)

    specs = coverage_config.iter_active_target_specs() if cc.use_coverage_targets else {}
    targets: dict[str, FeatureCoverageTarget] = {}
    required_keys: set[str] = set()
    optional_keys: set[str] = set()

    for feature_key, spec in specs.items():
        key = normalize_feature_key(feature_key)
        family, token = parse_feature_key(key)
        is_required_family = _family_allowed(family, cc.required_families)
        is_optional_family = _family_allowed(family, cc.optional_families)
        if not is_required_family and not is_optional_family:
            # Still track if listed in coverage required_features
            if key not in coverage_config.required_features and key not in coverage_config.optional_features:
                continue
            is_required_family = key in coverage_config.required_features or coverage_config.is_required(key)
            is_optional_family = not is_required_family

        # Family-level and coverage-level required flags
        family_cfg = coverage_config.features.get(family)
        coverage_says_required = coverage_config.is_required(key)
        required = bool(is_required_family and coverage_says_required)
        if key in coverage_config.required_features:
            required = True
        if key in coverage_config.optional_features:
            required = False
        if is_optional_family and not is_required_family:
            required = False

        eligible_count = len(feature_to_clip_ids.get(key, set()))
        speaker_count = len(speaker_by_feature.get(key, set()))
        index_count = None
        if index_candidate_counts is not None:
            index_count = int(index_candidate_counts.get(key, 0))

        eff_min, eff_des = compute_effective_targets(
            configured_minimum=int(spec.minimum),
            configured_desired=int(spec.desired),
            eligible_clip_count=eligible_count,
        )
        weight = (
            cc.required_weight_default
            if required
            else cc.optional_weight_default
        )
        if family_cfg is not None and family_cfg.required and required:
            weight = max(weight, cc.required_weight_default)

        targets[key] = FeatureCoverageTarget(
            feature_key=key,
            feature_family=family,
            feature_token=token,
            configured_minimum=int(spec.minimum),
            configured_desired=int(spec.desired),
            required=required,
            weight=weight,
            eligible_clip_count=eligible_count,
            eligible_unique_speaker_count=speaker_count,
            index_candidate_count=index_count,
            effective_minimum=eff_min,
            effective_desired=eff_des,
        )
        if required:
            required_keys.add(key)
        else:
            optional_keys.add(key)

    return SelectionCoverageConstraints(
        enabled=bool(cc.enabled),
        violation_policy=cc.violation_policy,
        preserve_during_local_search=cc.preserve_during_local_search,
        targets=targets,
        required_keys=required_keys,
        optional_keys=optional_keys,
        feature_to_clip_ids=feature_to_clip_ids,
        clip_coverage_keys=clip_coverage_keys,
        quality_weight=cc.quality_weight,
        diversity_weight=cc.diversity_weight,
        duration_penalty_weight=cc.duration_penalty_weight,
    )


def rarity_weight(eligible_clip_count: int) -> float:
    return 1.0 / math.sqrt(max(eligible_clip_count, 0) + 1.0)


def deficit_weight(deficit: int, effective_minimum: int) -> float:
    return float(deficit) / float(max(effective_minimum, 1))
