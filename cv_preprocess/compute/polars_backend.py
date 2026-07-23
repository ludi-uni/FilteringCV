from __future__ import annotations

from collections import Counter

import polars as pl

from cv_preprocess.catalog.aggregates import build_duplicate_groups, build_feature_counts
from cv_preprocess.compute.protocol import SelectionState
from cv_preprocess.selection.protocol import ClipFeatures
from cv_preprocess.selection.scoring import total_selection_score


class PolarsComputeBackend:
    """Polars-oriented compute path delegating catalog aggregates to vectorized helpers."""

    @property
    def name(self) -> str:
        return "polars"

    def count_features(self, clips: pl.DataFrame) -> pl.DataFrame:
        return build_feature_counts(clips)

    def build_duplicate_groups(self, clips: pl.DataFrame) -> pl.DataFrame:
        return build_duplicate_groups(clips)

    def score_candidates(
        self,
        candidates: list[ClipFeatures],
        *,
        feature_weights: dict[str, float],
        diminishing_tau: dict[str, float],
        temperatures: dict[str, float],
        pool_counts_by_family,
        utterance_counts_by_family,
        speaker_sets_by_family,
        min_utterances_by_family: dict[str, int],
        min_speakers_by_family: dict[str, int],
        state: SelectionState,
        quality_weight: float = 0.0,
        speaker_diversity_weight: float = 0.0,
    ):
        scores = {}
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
