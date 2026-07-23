from __future__ import annotations

import random
from collections import Counter, defaultdict

from cv_preprocess.config.dataset_builder import (
    DatasetBuilderSplitConfig,
    LeakagePolicyConfig,
    PreserveTrainConfig,
)
from cv_preprocess.split.leakage import (
    ClipSplitRecord,
    _is_violation,
    _resolve_leakage_policy,
)
from cv_preprocess.split.protocol import SPLIT_ORDER, SplitProtocol


def _feature_speaker_support(clips: list[ClipSplitRecord]) -> dict[tuple[str, str], int]:
    speakers_by_feature: dict[tuple[str, str], set[str]] = defaultdict(set)
    for clip in clips:
        for family, tokens in clip.features_by_family.items():
            for token in tokens:
                speakers_by_feature[(family, token)].add(clip.speaker_id)
    return {key: len(speakers) for key, speakers in speakers_by_feature.items()}


def _critical_features(
    feature_support: dict[tuple[str, str], int],
    preserve_train: PreserveTrainConfig,
) -> set[tuple[str, str]]:
    if not preserve_train.enabled:
        return set()
    return {
        key
        for key, count in feature_support.items()
        if count <= preserve_train.critical_feature_max_speakers
    }


def _clip_features(clip: ClipSplitRecord) -> set[tuple[str, str]]:
    return {
        (family, token)
        for family, tokens in clip.features_by_family.items()
        for token in tokens
    }


def _dimension_value(clip: ClipSplitRecord, dimension: str) -> str | None:
    if dimension == "speaker":
        return clip.speaker_id or None
    if dimension == "audio_hash":
        return clip.audio_hash or None
    if dimension == "sentence_id":
        return clip.sentence_id or None
    if dimension == "normalized_text":
        return clip.normalized_text or None
    return None


def _leakage_allowed(
    clip: ClipSplitRecord,
    split_name: str,
    assigned_by_split: dict[str, list[ClipSplitRecord]],
    policy: LeakagePolicyConfig,
    *,
    protocol: SplitProtocol,
) -> bool:
    if protocol == SplitProtocol.SINGLE_SPEAKER and split_name != "train":
        pass

    dimensions = ("speaker", "audio_hash", "sentence_id", "normalized_text")
    for dimension in dimensions:
        if protocol == SplitProtocol.SINGLE_SPEAKER and dimension == "speaker":
            continue
        action = getattr(policy, dimension)
        if action == "allow":
            continue
        value = _dimension_value(clip, dimension)
        if not value:
            continue
        occupied_splits: set[str] = set()
        for other_split, other_clips in assigned_by_split.items():
            if other_split == split_name:
                continue
            for other in other_clips:
                if _dimension_value(other, dimension) == value:
                    occupied_splits.add(other_split)
        if _is_violation(occupied_splits | {split_name}, action):
            return False
    return True


def _split_targets(total_duration: float, ratios: dict[str, float]) -> dict[str, float]:
    return {split_name: total_duration * ratios.get(split_name, 0.0) for split_name in SPLIT_ORDER}


def assign_clip_splits(
    clips: list[ClipSplitRecord],
    config: DatasetBuilderSplitConfig,
    *,
    protocol: SplitProtocol = SplitProtocol.SEEN_SPEAKER,
    feature_speaker_support: dict[tuple[str, str], int] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Assign selected clips to splits with leakage constraints and train preservation."""
    warnings: list[str] = []
    if not clips:
        return {}, ["no clips provided for split assignment"]

    ratios = config.resolved_ratios()
    preserve_train = config.resolved_preserve_train()
    policy = _resolve_leakage_policy(config.leakage_policy)
    support = feature_speaker_support or _feature_speaker_support(clips)
    critical = _critical_features(support, preserve_train)

    total_duration = sum(clip.duration_sec for clip in clips)
    targets = _split_targets(total_duration, ratios)
    split_duration: dict[str, float] = {name: 0.0 for name in SPLIT_ORDER}
    assigned_by_split: dict[str, list[ClipSplitRecord]] = {name: [] for name in SPLIT_ORDER}
    assignments: dict[str, str] = {}

    train_feature_counts: Counter[tuple[str, str]] = Counter()
    critical_needs: dict[tuple[str, str], int] = {
        feature: preserve_train.min_train_occurrences for feature in critical
    }

    ordered = sorted(clips, key=lambda clip: clip.clip_id)
    rng = random.Random(config.seed)
    rng.shuffle(ordered)

    for clip in ordered:
        clip_critical = _clip_features(clip) & critical
        candidate_splits = list(SPLIT_ORDER)
        if clip_critical:
            candidate_splits = ["train"] + [name for name in SPLIT_ORDER if name != "train"]

        best_split: str | None = None
        best_score = float("-inf")
        for split_name in candidate_splits:
            if ratios.get(split_name, 0.0) <= 0.0 and split_name != "train":
                continue
            if not _leakage_allowed(
                clip,
                split_name,
                assigned_by_split,
                policy,
                protocol=protocol,
            ):
                continue
            projected = split_duration[split_name] + clip.duration_sec
            deficit = targets[split_name] - projected
            score = deficit
            if clip_critical and split_name == "train":
                score += 1_000_000.0
            if score > best_score or (score == best_score and (best_split is None or split_name < best_split)):
                best_score = score
                best_split = split_name

        if best_split is None:
            warnings.append(f"could not assign clip {clip.clip_id} without leakage; defaulting to train")
            best_split = "train"

        assignments[clip.clip_id] = best_split
        assigned_by_split[best_split].append(clip)
        split_duration[best_split] += clip.duration_sec
        for feature in _clip_features(clip):
            if feature in critical and best_split == "train":
                train_feature_counts[feature] += 1

    for feature, needed in critical_needs.items():
        if train_feature_counts[feature] < needed:
            donor = next(
                (
                    clip
                    for clip in ordered
                    if feature in _clip_features(clip) and assignments.get(clip.clip_id) != "train"
                ),
                None,
            )
            if donor is None:
                warnings.append(
                    f"critical feature {feature!r} has {train_feature_counts[feature]} train occurrences; "
                    f"need {needed}"
                )
                continue
            old_split = assignments[donor.clip_id]
            assigned_by_split[old_split] = [
                clip for clip in assigned_by_split[old_split] if clip.clip_id != donor.clip_id
            ]
            split_duration[old_split] -= donor.duration_sec
            assignments[donor.clip_id] = "train"
            assigned_by_split["train"].append(donor)
            split_duration["train"] += donor.duration_sec
            train_feature_counts[feature] += 1

    return assignments, warnings
