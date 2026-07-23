from __future__ import annotations

import time
from collections import Counter
from itertools import combinations
from typing import Callable

from cv_preprocess.selection.constraints import (
    ConstraintConfig,
    ConstraintState,
    add_clip_to_state,
)
from cv_preprocess.selection.protocol import ClipFeatures
from cv_preprocess.selection.scoring import (
    pool_counts_for_family,
    speaker_counts_for_family,
    total_selection_score,
)


ScoreFn = Callable[[ClipFeatures, ConstraintState], tuple[float, dict, dict]]


def _build_score_fn(
    *,
    feature_weights: dict[str, float],
    diminishing_tau: dict[str, float],
    temperatures: dict[str, float],
    candidates: list[ClipFeatures],
    min_utterances_by_family: dict[str, int],
    min_speakers_by_family: dict[str, int],
    current_counts_by_family: dict[str, Counter[str]],
) -> ScoreFn:
    pool_counts_by_family = {
        family: pool_counts_for_family(candidates, family) for family in feature_weights
    }
    utterance_counts_by_family = {
        family: pool_counts_for_family(candidates, family) for family in feature_weights
    }
    speaker_sets_by_family = {
        family: speaker_counts_for_family(candidates, family) for family in feature_weights
    }

    def score(clip: ClipFeatures, state: ConstraintState) -> tuple[float, dict, dict]:
        return total_selection_score(
            clip,
            feature_weights=feature_weights,
            diminishing_tau=diminishing_tau,
            temperatures=temperatures,
            current_counts_by_family=current_counts_by_family,
            pool_counts_by_family=pool_counts_by_family,
            utterance_counts_by_family=utterance_counts_by_family,
            speaker_sets_by_family=speaker_sets_by_family,
            min_utterances_by_family=min_utterances_by_family,
            min_speakers_by_family=min_speakers_by_family,
            selected_speakers=state.selected_speakers,
            quality_weight=feature_weights.get("quality", 0.0),
            speaker_diversity_weight=feature_weights.get("speaker_diversity", 0.0),
        )

    return score


def _evaluate_set(
    selected: list[str],
    clips_by_id: dict[str, ClipFeatures],
    score_fn: ScoreFn,
    constraint_config: ConstraintConfig,
) -> tuple[float, ConstraintState]:
    state = ConstraintState()
    total = 0.0
    for clip_id in selected:
        clip = clips_by_id[clip_id]
        clip_score, _, _ = score_fn(clip, state)
        total += clip_score
        add_clip_to_state(clip, state, constraint_config)
    return total, state


def local_search_improve(
    selected: list[str],
    reserve: list[str],
    clips_by_id: dict[str, ClipFeatures],
    *,
    feature_weights: dict[str, float],
    diminishing_tau: dict[str, float],
    temperatures: dict[str, float],
    candidates: list[ClipFeatures],
    min_utterances_by_family: dict[str, int],
    min_speakers_by_family: dict[str, int],
    constraint_config: ConstraintConfig,
    swap_patterns: list[str],
    max_iterations: int,
    max_wall_sec: float,
) -> tuple[list[str], list[str], int]:
    if not selected or not reserve:
        return selected, reserve, 0

    current_counts_by_family: dict[str, Counter[str]] = {
        family: Counter() for family in feature_weights
    }
    for clip_id in selected:
        clip = clips_by_id[clip_id]
        for family, tokens in clip.features_by_family.items():
            for token in tokens:
                current_counts_by_family.setdefault(family, Counter())[token] += 1

    score_fn = _build_score_fn(
        feature_weights=feature_weights,
        diminishing_tau=diminishing_tau,
        temperatures=temperatures,
        candidates=candidates,
        min_utterances_by_family=min_utterances_by_family,
        min_speakers_by_family=min_speakers_by_family,
        current_counts_by_family=current_counts_by_family,
    )

    selected_set = list(selected)
    reserve_set = list(reserve)
    best_score, _ = _evaluate_set(selected_set, clips_by_id, score_fn, constraint_config)
    iterations = 0
    start = time.monotonic()

    patterns = set(swap_patterns)
    while iterations < max_iterations and (time.monotonic() - start) < max_wall_sec:
        improved = False
        iterations += 1

        if "1v1" in patterns:
            for out_id in selected_set:
                for in_id in reserve_set:
                    if out_id == in_id:
                        continue
                    trial_selected = [cid for cid in selected_set if cid != out_id] + [in_id]
                    trial_score, _ = _evaluate_set(
                        trial_selected, clips_by_id, score_fn, constraint_config
                    )
                    if trial_score > best_score + 1e-12:
                        selected_set = trial_selected
                        reserve_set = [cid for cid in reserve_set if cid != in_id] + [out_id]
                        best_score = trial_score
                        improved = True
                        break
                if improved:
                    break
        if improved:
            continue

        if "1v2" in patterns:
            for out_id in selected_set:
                for in_a, in_b in combinations(reserve_set, 2):
                    trial_selected = [
                        cid for cid in selected_set if cid != out_id
                    ] + [in_a, in_b]
                    trial_score, _ = _evaluate_set(
                        trial_selected, clips_by_id, score_fn, constraint_config
                    )
                    if trial_score > best_score + 1e-12:
                        selected_set = trial_selected
                        reserve_set = [
                            cid for cid in reserve_set if cid not in {in_a, in_b}
                        ] + [out_id]
                        best_score = trial_score
                        improved = True
                        break
                if improved:
                    break
        if improved:
            continue

        if "2v1" in patterns:
            for out_a, out_b in combinations(selected_set, 2):
                for in_id in reserve_set:
                    trial_selected = [
                        cid for cid in selected_set if cid not in {out_a, out_b}
                    ] + [in_id]
                    if len(trial_selected) != len(selected_set) - 1:
                        continue
                    trial_score, _ = _evaluate_set(
                        trial_selected, clips_by_id, score_fn, constraint_config
                    )
                    if trial_score > best_score + 1e-12:
                        selected_set = trial_selected
                        reserve_set = [cid for cid in reserve_set if cid != in_id] + [
                            out_a,
                            out_b,
                        ]
                        best_score = trial_score
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    return selected_set, reserve_set, iterations
