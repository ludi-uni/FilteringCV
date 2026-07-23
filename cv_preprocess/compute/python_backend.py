from __future__ import annotations

from collections import Counter, defaultdict

import polars as pl

from cv_preprocess.catalog.aggregates import (
    FEATURE_SPECS,
    _feature_values,
    _groups_from_key_map,
    normalized_text_sha256,
)
from cv_preprocess.compute.protocol import SelectionState
from cv_preprocess.selection.protocol import ClipFeatures
from cv_preprocess.selection.scoring import total_selection_score

_FEATURE_COUNTS_SCHEMA = {
    "feature_type": pl.Utf8,
    "feature": pl.Utf8,
    "count": pl.Int64,
    "speaker_count": pl.Int64,
    "utterance_count": pl.Int64,
}

_DUPLICATE_GROUPS_SCHEMA = {
    "group_id": pl.Utf8,
    "kind": pl.Utf8,
    "clip_ids": pl.List(pl.Utf8),
    "size": pl.Int64,
}


class PythonComputeBackend:
    """Pure-Python compute path for catalogs and selection scoring."""

    @property
    def name(self) -> str:
        return "python"

    def count_features(self, clips: pl.DataFrame) -> pl.DataFrame:
        if clips.is_empty():
            return pl.DataFrame(schema=_FEATURE_COUNTS_SCHEMA)

        counts: dict[tuple[str, str], int] = defaultdict(int)
        speakers: dict[tuple[str, str], set[str]] = defaultdict(set)
        utterances: dict[tuple[str, str], set[str]] = defaultdict(set)

        columns = [
            "clip_id",
            "speaker_id",
            "phonemes",
            *[col for _, col in FEATURE_SPECS if col != "phonemes"],
        ]
        for row in clips.select(columns).iter_rows(named=True):
            clip_id = str(row["clip_id"])
            speaker_id = str(row.get("speaker_id") or "")
            for feature_type, _ in FEATURE_SPECS:
                for feature in _feature_values(feature_type, row):
                    key = (feature_type, feature)
                    counts[key] += 1
                    speakers[key].add(speaker_id)
                    utterances[key].add(clip_id)

        rows = [
            {
                "feature_type": feature_type,
                "feature": feature,
                "count": count,
                "speaker_count": len(speakers[(feature_type, feature)]),
                "utterance_count": len(utterances[(feature_type, feature)]),
            }
            for (feature_type, feature), count in sorted(counts.items())
        ]
        return pl.DataFrame(rows)

    def build_duplicate_groups(self, clips: pl.DataFrame) -> pl.DataFrame:
        if clips.is_empty():
            return pl.DataFrame(schema=_DUPLICATE_GROUPS_SCHEMA)

        rows: list[dict[str, object]] = []
        exact_audio = self._groups_for_key(clips, kind="exact_audio", key="audio_sha256")
        rows.extend(exact_audio)
        rows.extend(
            self._groups_for_key(
                clips,
                kind="same_source_path",
                key="normalized_relative_source_path",
            )
        )

        sentence_rows = [row for row in clips.iter_rows(named=True) if row.get("sentence_id")]
        if sentence_rows:
            rows.extend(
                self._groups_for_key(
                    pl.DataFrame(sentence_rows),
                    kind="same_sentence_id",
                    key="sentence_id",
                )
            )

        text_rows: list[dict[str, object]] = []
        for row in clips.iter_rows(named=True):
            text_norm = row.get("text_norm")
            if text_norm:
                row_dict = dict(row)
                row_dict["normalized_text_sha256"] = normalized_text_sha256(str(text_norm))
                text_rows.append(row_dict)
        if text_rows:
            rows.extend(
                self._groups_for_key(
                    pl.DataFrame(text_rows),
                    kind="same_normalized_text",
                    key="normalized_text_sha256",
                )
            )

        speaker_text_rows = [
            row
            for row in clips.iter_rows(named=True)
            if row.get("text_norm") and row.get("speaker_id")
        ]
        if speaker_text_rows:
            keyed_rows: list[dict[str, object]] = []
            for row in speaker_text_rows:
                row_dict = dict(row)
                row_dict["speaker_text_key"] = f"{row['speaker_id']}\x00{row['text_norm']}"
                keyed_rows.append(row_dict)
            rows.extend(
                self._groups_for_key(
                    pl.DataFrame(keyed_rows),
                    kind="same_speaker_same_text",
                    key="speaker_text_key",
                )
            )

        if not rows:
            return pl.DataFrame(schema=_DUPLICATE_GROUPS_SCHEMA)
        return pl.DataFrame(rows).sort(["kind", "group_id"])

    def score_candidates(
        self,
        candidates: list[ClipFeatures],
        *,
        feature_weights: dict[str, float],
        diminishing_tau: dict[str, float],
        temperatures: dict[str, float],
        pool_counts_by_family: dict[str, Counter[str]],
        utterance_counts_by_family: dict[str, Counter[str]],
        speaker_sets_by_family: dict[str, dict[str, set[str]]],
        min_utterances_by_family: dict[str, int],
        min_speakers_by_family: dict[str, int],
        state: SelectionState,
        quality_weight: float = 0.0,
        speaker_diversity_weight: float = 0.0,
    ) -> dict[str, tuple[float, dict[str, object], dict[str, float]]]:
        scores: dict[str, tuple[float, dict[str, object], dict[str, float]]] = {}
        for clip in candidates:
            scores[clip.clip_id] = total_selection_score(
                clip,
                feature_weights=feature_weights,
                diminishing_tau=diminishing_tau,
                temperatures=temperatures,
                current_counts_by_family=state.current_counts_by_family,
                pool_counts_by_family=pool_counts_by_family,
                utterance_counts_by_family=utterance_counts_by_family,
                speaker_sets_by_family=speaker_sets_by_family,
                min_utterances_by_family=min_utterances_by_family,
                min_speakers_by_family=min_speakers_by_family,
                selected_speakers=state.selected_speakers,
                quality_weight=quality_weight,
                speaker_diversity_weight=speaker_diversity_weight,
            )
        return scores

    def update_selection_state(self, state: SelectionState, clip: ClipFeatures) -> None:
        if clip.speaker_id:
            state.selected_speakers.add(clip.speaker_id)
        for family, tokens in clip.features_by_family.items():
            counter = state.current_counts_by_family.setdefault(family, Counter())
            for token in tokens:
                counter[token] += 1

    def _groups_for_key(
        self,
        clips: pl.DataFrame,
        *,
        kind: str,
        key: str,
    ) -> list[dict[str, object]]:
        key_to_clip_ids: dict[str, list[str]] = defaultdict(list)
        for row in clips.iter_rows(named=True):
            value = row.get(key)
            if value is None:
                continue
            key_to_clip_ids[str(value)].append(str(row["clip_id"]))
        return _groups_from_key_map(kind=kind, key_to_clip_ids=dict(key_to_clip_ids))
