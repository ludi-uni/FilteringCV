from __future__ import annotations

import random
from collections import defaultdict

import polars as pl

from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.config.dataset_builder import DatasetBuilderSplitConfig, PreserveTrainConfig
from cv_preprocess.split.leakage import ClipSplitRecord
from cv_preprocess.split.protocol import SPLIT_ORDER


def _eligible_clips(clips: pl.DataFrame) -> pl.DataFrame:
    eligible_value = ClipDisposition.ELIGIBLE.value
    return clips.filter(
        (pl.col("disposition") == eligible_value)
        | pl.col("disposition").is_null()
        | (pl.col("disposition") == ClipDisposition.SELECTED.value)
    )


def _speaker_durations(clips: pl.DataFrame) -> dict[str, float]:
    durations: dict[str, float] = defaultdict(float)
    for row in clips.iter_rows(named=True):
        speaker = str(row.get("speaker_id") or "")
        if not speaker:
            continue
        durations[speaker] += float(row.get("duration_sec") or 0.0)
    return dict(durations)


def _critical_feature_speakers(
    clips: pl.DataFrame,
    feature_counts: pl.DataFrame | None,
    preserve_train: PreserveTrainConfig,
) -> set[str]:
    if not preserve_train.enabled or feature_counts is None or feature_counts.is_empty():
        return set()

    critical_features: set[tuple[str, str]] = set()
    for row in feature_counts.iter_rows(named=True):
        speaker_count = int(row.get("speaker_count") or 0)
        if speaker_count <= preserve_train.critical_feature_max_speakers:
            critical_features.add((str(row["feature_type"]), str(row["feature"])))

    if not critical_features:
        return set()

    protected: set[str] = set()
    for row in clips.iter_rows(named=True):
        speaker = str(row.get("speaker_id") or "")
        if not speaker:
            continue
        phonemes = str(row.get("phonemes") or "")
        for token in phonemes.replace("\t", " ").split():
            if ("phone", token) in critical_features:
                protected.add(speaker)
                break
    return protected


def _split_targets(
    total_duration: float,
    ratios: dict[str, float],
) -> dict[str, float]:
    return {split_name: total_duration * ratios.get(split_name, 0.0) for split_name in SPLIT_ORDER}


def plan_unseen_speaker_splits(
    clips: pl.DataFrame,
    config: DatasetBuilderSplitConfig,
    *,
    feature_counts: pl.DataFrame | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Assign speakers to splits by duration ratio with train preservation."""
    warnings: list[str] = []
    eligible = _eligible_clips(clips)
    if eligible.is_empty():
        return {}, ["no eligible clips for speaker split planning"]

    ratios = config.resolved_ratios()
    preserve_train = config.resolved_preserve_train()
    speaker_duration = _speaker_durations(eligible)
    if not speaker_duration:
        return {}, ["no speakers found in eligible clips"]

    protected_speakers = _critical_feature_speakers(eligible, feature_counts, preserve_train)
    total_duration = sum(speaker_duration.values())
    targets = _split_targets(total_duration, ratios)

    split_duration: dict[str, float] = {name: 0.0 for name in SPLIT_ORDER}
    assignments: dict[str, str] = {}

    for speaker in protected_speakers:
        if speaker in speaker_duration:
            assignments[speaker] = "train"
            split_duration["train"] += speaker_duration[speaker]

    remaining = [
        speaker
        for speaker in sorted(speaker_duration)
        if speaker not in assignments
    ]
    rng = random.Random(config.seed)
    rng.shuffle(remaining)

    for speaker in remaining:
        duration = speaker_duration[speaker]
        best_split = "train"
        best_deficit = float("-inf")
        for split_name in SPLIT_ORDER:
            if ratios.get(split_name, 0.0) <= 0.0:
                continue
            projected = split_duration[split_name] + duration
            deficit = targets[split_name] - projected
            if deficit > best_deficit or (
                deficit == best_deficit and split_name < best_split
            ):
                best_deficit = deficit
                best_split = split_name
        assignments[speaker] = best_split
        split_duration[best_split] += duration

    train_speakers = {speaker for speaker, split_name in assignments.items() if split_name == "train"}
    if protected_speakers - train_speakers:
        warnings.append(
            "some critical-feature speakers could not be assigned exclusively to train"
        )

    overlap = set()
    for split_name in ("val", "test"):
        split_speakers = {s for s, n in assignments.items() if n == split_name}
        overlap |= train_speakers & split_speakers
    if overlap:
        warnings.append(f"speaker overlap detected across splits: {sorted(overlap)}")

    return assignments, warnings


def clips_by_split_from_speakers(
    clips: pl.DataFrame,
    speaker_assignments: dict[str, str],
) -> dict[str, list[ClipSplitRecord]]:
    grouped: dict[str, list[ClipSplitRecord]] = {name: [] for name in SPLIT_ORDER}
    for row in _eligible_clips(clips).iter_rows(named=True):
        speaker = str(row.get("speaker_id") or "")
        split_name = speaker_assignments.get(speaker)
        if split_name is None:
            continue
        grouped.setdefault(split_name, []).append(
            ClipSplitRecord(
                clip_id=str(row["clip_id"]),
                speaker_id=speaker,
                audio_hash=str(row.get("audio_sha256") or ""),
                sentence_id=str(row.get("sentence_id") or ""),
                normalized_text=str(row.get("text_norm") or ""),
                duration_sec=float(row.get("duration_sec") or 0.0),
            )
        )
    return grouped
