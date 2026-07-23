from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import polars as pl

from cv_preprocess.selection.protocol import ClipFeatures


@dataclass
class SelectionState:
    """Mutable selection counters used while scoring greedy candidates."""

    current_counts_by_family: dict[str, Counter[str]] = field(default_factory=dict)
    selected_speakers: set[str] = field(default_factory=set)


@runtime_checkable
class ComputeBackend(Protocol):
    @property
    def name(self) -> str: ...

    def count_features(self, clips: pl.DataFrame) -> pl.DataFrame: ...

    def build_duplicate_groups(self, clips: pl.DataFrame) -> pl.DataFrame: ...

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
    ) -> dict[str, tuple[float, dict[str, Any], dict[str, float]]]: ...

    def update_selection_state(self, state: SelectionState, clip: ClipFeatures) -> None: ...
