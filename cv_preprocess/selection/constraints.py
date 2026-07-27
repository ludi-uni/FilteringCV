from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from cv_preprocess.selection.protocol import ClipFeatures


@dataclass
class ConstraintState:
    speaker_duration_sec: Counter[str] = field(default_factory=Counter)
    speaker_clip_count: Counter[str] = field(default_factory=Counter)
    duplicate_group_selected: dict[str, Counter[str]] = field(default_factory=dict)
    selected_count: int = 0
    low_quality_count: int = 0
    total_duration_sec: float = 0.0
    selected_speakers: set[str] = field(default_factory=set)


@dataclass
class ConstraintConfig:
    max_clips_per_speaker: int | None = None
    max_duration_sec_per_speaker: float | None = None
    min_duration_sec_per_speaker: float | None = None
    duplicate_max_selected: dict[str, int] = field(default_factory=dict)
    hard_min_quality: float | None = None
    preferred_quality: float | None = None
    max_low_quality_ratio: float | None = None
    low_quality_threshold: float | None = None


def _is_low_quality(score: float | None, threshold: float | None) -> bool:
    if score is None or threshold is None:
        return False
    return score < threshold


def can_add_clip(
    clip: ClipFeatures,
    state: ConstraintState,
    config: ConstraintConfig,
) -> tuple[bool, dict[str, float]]:
    penalties: dict[str, float] = {}
    if config.hard_min_quality is not None:
        if clip.quality_score is None or clip.quality_score < config.hard_min_quality:
            penalties["quality_hard_min"] = 1.0
            return False, penalties

    speaker = clip.speaker_id
    next_clip_count = state.speaker_clip_count[speaker] + 1
    next_duration = state.speaker_duration_sec[speaker] + clip.duration_sec
    if config.max_clips_per_speaker is not None and next_clip_count > config.max_clips_per_speaker:
        penalties["speaker_max_clips"] = 1.0
        return False, penalties
    if (
        config.max_duration_sec_per_speaker is not None
        and next_duration > config.max_duration_sec_per_speaker
    ):
        penalties["speaker_max_duration"] = 1.0
        return False, penalties

    for group_type, group_id in clip.duplicate_groups.items():
        max_selected = config.duplicate_max_selected.get(group_type)
        if max_selected is None:
            continue
        current = state.duplicate_group_selected.get(group_type, Counter()).get(group_id, 0)
        if current + 1 > max_selected:
            penalties[f"duplicate_{group_type}"] = 1.0
            return False, penalties

    if config.max_low_quality_ratio is not None and config.low_quality_threshold is not None:
        next_selected = state.selected_count + 1
        next_low = state.low_quality_count
        if _is_low_quality(clip.quality_score, config.low_quality_threshold):
            next_low += 1
        ratio = next_low / next_selected
        if ratio > config.max_low_quality_ratio:
            penalties["max_low_quality_ratio"] = 1.0
            return False, penalties

    return True, penalties


def add_clip_to_state(clip: ClipFeatures, state: ConstraintState, config: ConstraintConfig) -> None:
    state.selected_count += 1
    state.total_duration_sec += clip.duration_sec
    state.speaker_clip_count[clip.speaker_id] += 1
    state.speaker_duration_sec[clip.speaker_id] += clip.duration_sec
    state.selected_speakers.add(clip.speaker_id)
    if _is_low_quality(clip.quality_score, config.low_quality_threshold):
        state.low_quality_count += 1
    for group_type, group_id in clip.duplicate_groups.items():
        state.duplicate_group_selected.setdefault(group_type, Counter())[group_id] += 1


def remove_clip_from_state(
    clip: ClipFeatures, state: ConstraintState, config: ConstraintConfig
) -> None:
    state.selected_count -= 1
    state.total_duration_sec -= clip.duration_sec
    state.speaker_clip_count[clip.speaker_id] -= 1
    state.speaker_duration_sec[clip.speaker_id] -= clip.duration_sec
    if state.speaker_clip_count[clip.speaker_id] <= 0:
        state.selected_speakers.discard(clip.speaker_id)
    if _is_low_quality(clip.quality_score, config.low_quality_threshold):
        state.low_quality_count -= 1
    for group_type, group_id in clip.duplicate_groups.items():
        counter = state.duplicate_group_selected.get(group_type)
        if counter is None:
            continue
        counter[group_id] -= 1
        if counter[group_id] <= 0:
            del counter[group_id]


def preserves_required_coverage(
    selected_counts: Mapping[str, int],
    removed_clip: ClipFeatures,
    added_clip: ClipFeatures,
    required_targets: Mapping[str, int],
    *,
    removed_keys: set[str] | None = None,
    added_keys: set[str] | None = None,
) -> bool:
    """Return True if swapping removed→added keeps required effective minima."""
    if not required_targets:
        return True
    removed = set(removed_keys) if removed_keys is not None else set(removed_clip.coverage_keys)
    added = set(added_keys) if added_keys is not None else set(added_clip.coverage_keys)
    for feature, minimum in required_targets.items():
        if minimum <= 0:
            continue
        current = int(selected_counts.get(feature, 0))
        delta = 0
        if feature in removed:
            delta -= 1
        if feature in added:
            delta += 1
        if current + delta < minimum:
            return False
    return True


def coverage_counts_for_selection(
    selected_ids: Sequence[str],
    clips_by_id: Mapping[str, ClipFeatures],
    feature_keys: Iterable[str],
) -> dict[str, int]:
    keys = set(feature_keys)
    counts = {key: 0 for key in keys}
    for clip_id in selected_ids:
        clip = clips_by_id.get(clip_id)
        if clip is None:
            continue
        for key in clip.coverage_keys:
            if key in counts:
                counts[key] += 1
    return counts
