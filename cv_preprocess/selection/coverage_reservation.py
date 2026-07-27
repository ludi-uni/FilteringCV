"""Phase A: greedy set-cover reservation for required coverage minima."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

from cv_preprocess.selection.constraints import (
    ConstraintConfig,
    ConstraintState,
    add_clip_to_state,
    can_add_clip,
)
from cv_preprocess.selection.coverage_targets import (
    SelectionCoverageConstraints,
    deficit_weight,
    rarity_weight,
)
from cv_preprocess.selection.protocol import ClipFeatures


@dataclass
class CoverageContribution:
    feature: str
    deficit_before: int
    deficit_after: int


@dataclass
class ReservationRecord:
    clip_id: str
    selection_phase: str = "coverage_reservation"
    coverage_contributions: list[CoverageContribution] = field(default_factory=list)
    quality_score: float | None = None
    duration_sec: float = 0.0
    speaker_id: str = ""
    reservation_score: float = 0.0


@dataclass
class ReservationResult:
    reserved_ids: list[str]
    records: list[ReservationRecord]
    selected_counts: dict[str, int]
    conflict_features: set[str] = field(default_factory=set)
    unmet_features: set[str] = field(default_factory=set)


def _current_deficits(
    constraints: SelectionCoverageConstraints,
    selected_counts: Mapping[str, int],
    *,
    required_only: bool = True,
) -> dict[str, int]:
    deficits: dict[str, int] = {}
    keys = constraints.required_keys if required_only else set(constraints.targets)
    for key in keys:
        target = constraints.targets.get(key)
        if target is None or target.effective_minimum <= 0:
            continue
        selected = int(selected_counts.get(key, 0))
        deficit = max(0, target.effective_minimum - selected)
        if deficit > 0:
            deficits[key] = deficit
    return deficits


def _clip_gain(
    clip_id: str,
    constraints: SelectionCoverageConstraints,
    deficits: Mapping[str, int],
) -> tuple[float, list[str]]:
    keys = constraints.clip_coverage_keys.get(clip_id, set())
    gain = 0.0
    contributing: list[str] = []
    for feature in keys:
        deficit = deficits.get(feature)
        if not deficit:
            continue
        target = constraints.targets.get(feature)
        if target is None:
            continue
        feature_gain = (
            target.weight
            * rarity_weight(target.eligible_clip_count)
            * deficit_weight(deficit, target.effective_minimum)
        )
        gain += feature_gain
        contributing.append(feature)
    return gain, contributing


def reserve_coverage_clips(
    candidates: list[ClipFeatures],
    constraints: SelectionCoverageConstraints,
    *,
    constraint_config: ConstraintConfig,
    initial_state: ConstraintState | None = None,
    initial_selected_counts: Mapping[str, int] | None = None,
    max_duration_sec: float | None = None,
    forced_exclude: set[str] | None = None,
) -> ReservationResult:
    """Greedy multi-feature set cover under existing hard constraints.

    Mutates a *copy* of ``initial_state`` for admissibility checks only; the
    caller is responsible for committing returned clip ids into the live state.
    """
    clips_by_id = {c.clip_id: c for c in candidates}
    if initial_state is None:
        state = ConstraintState()
    else:
        state = ConstraintState(
            speaker_duration_sec=Counter(initial_state.speaker_duration_sec),
            speaker_clip_count=Counter(initial_state.speaker_clip_count),
            duplicate_group_selected={
                key: Counter(value)
                for key, value in initial_state.duplicate_group_selected.items()
            },
            selected_count=initial_state.selected_count,
            low_quality_count=initial_state.low_quality_count,
            total_duration_sec=initial_state.total_duration_sec,
            selected_speakers=set(initial_state.selected_speakers),
        )
    selected_counts: dict[str, int] = {key: 0 for key in constraints.targets}
    if initial_selected_counts:
        for key, value in initial_selected_counts.items():
            if key in selected_counts:
                selected_counts[key] = int(value)
    reserved: list[str] = []
    records: list[ReservationRecord] = []
    conflict_features: set[str] = set()
    exclude = set(forced_exclude or ())
    selected_set: set[str] = set(exclude)  # already-selected ids are passed via exclude

    while True:
        deficits = _current_deficits(constraints, selected_counts, required_only=True)
        if not deficits:
            break

        candidate_ids: set[str] = set()
        for feature in deficits:
            candidate_ids |= constraints.feature_to_clip_ids.get(feature, set())
        candidate_ids -= selected_set
        candidate_ids -= exclude

        best_clip: ClipFeatures | None = None
        best_score = float("-inf")
        best_features: list[str] = []
        blocked_by_constraint = False

        for clip_id in candidate_ids:
            clip = clips_by_id.get(clip_id)
            if clip is None:
                continue
            if max_duration_sec is not None and state.total_duration_sec + clip.duration_sec > max_duration_sec:
                continue
            ok, _penalties = can_add_clip(clip, state, constraint_config)
            if not ok:
                blocked_by_constraint = True
                continue
            coverage_gain, contributing = _clip_gain(clip_id, constraints, deficits)
            if coverage_gain <= 0:
                continue
            quality = (clip.quality_score or 0.0) / 100.0
            diversity = 1.0 if clip.speaker_id not in state.selected_speakers else 0.0
            duration_penalty = constraints.duration_penalty_weight * max(clip.duration_sec, 0.0)
            score = (
                coverage_gain
                + quality * constraints.quality_weight
                + diversity * constraints.diversity_weight
                - duration_penalty
            )
            if score > best_score or (
                score == best_score and (best_clip is None or clip.clip_id < best_clip.clip_id)
            ):
                best_score = score
                best_clip = clip
                best_features = contributing

        if best_clip is None:
            # Remaining deficits cannot be filled under constraints.
            for feature, deficit in deficits.items():
                eligible = constraints.feature_to_clip_ids.get(feature, set())
                remaining = eligible - selected_set - exclude
                if not remaining:
                    continue
                if blocked_by_constraint or remaining:
                    # Candidates exist but none were admissible.
                    if remaining:
                        conflict_features.add(feature)
            break

        contributions: list[CoverageContribution] = []
        clip_keys = constraints.clip_coverage_keys.get(best_clip.clip_id, set())
        for feature in sorted(clip_keys):
            target = constraints.targets.get(feature)
            if target is None:
                continue
            before = max(0, target.effective_minimum - int(selected_counts.get(feature, 0)))
            selected_counts[feature] = int(selected_counts.get(feature, 0)) + 1
            after = max(0, target.effective_minimum - selected_counts[feature])
            if before > 0:
                contributions.append(
                    CoverageContribution(
                        feature=feature,
                        deficit_before=before,
                        deficit_after=after,
                    )
                )

        add_clip_to_state(best_clip, state, constraint_config)
        reserved.append(best_clip.clip_id)
        selected_set.add(best_clip.clip_id)
        records.append(
            ReservationRecord(
                clip_id=best_clip.clip_id,
                coverage_contributions=contributions,
                quality_score=best_clip.quality_score,
                duration_sec=best_clip.duration_sec,
                speaker_id=best_clip.speaker_id,
                reservation_score=best_score,
            )
        )

    unmet = {
        key
        for key, target in constraints.targets.items()
        if key in constraints.required_keys
        and target.effective_minimum > int(selected_counts.get(key, 0))
    }
    return ReservationResult(
        reserved_ids=reserved,
        records=records,
        selected_counts=selected_counts,
        conflict_features=conflict_features,
        unmet_features=unmet,
    )
