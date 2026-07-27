from __future__ import annotations

import random
import time
from collections import Counter
from itertools import combinations
from typing import Callable

from cv_preprocess.application.common import ProgressEvent, ProgressSink
from cv_preprocess.selection.constraints import (
    ConstraintConfig,
    ConstraintState,
    add_clip_to_state,
    coverage_counts_for_selection,
    preserves_required_coverage,
)
from cv_preprocess.selection.protocol import ClipFeatures
from cv_preprocess.selection.scoring import (
    dedupe_clip_features,
    pool_counts_for_family,
    precompute_family_score_tables,
    score_clip_fast,
    speaker_counts_for_family,
)


ScoreFn = Callable[[ClipFeatures, ConstraintState], float]


def _build_score_fn(
    *,
    feature_weights: dict[str, float],
    diminishing_tau: dict[str, float],
    temperatures: dict[str, float],
    candidates: list[ClipFeatures],
    min_utterances_by_family: dict[str, int],
    min_speakers_by_family: dict[str, int],
    current_counts_by_family: dict[str, Counter[str]],
    target_tables: dict[str, dict[str, float]] | None = None,
    features_cache: dict[str, dict[str, list[str]]] | None = None,
) -> ScoreFn:
    if target_tables is None:
        pool_counts_by_family = {
            family: pool_counts_for_family(candidates, family) for family in feature_weights
        }
        target_tables = precompute_family_score_tables(
            feature_weights=feature_weights,
            temperatures=temperatures,
            pool_counts_by_family=pool_counts_by_family,
            utterance_counts_by_family=pool_counts_by_family,
            speaker_sets_by_family={
                family: speaker_counts_for_family(candidates, family) for family in feature_weights
            },
            min_utterances_by_family=min_utterances_by_family,
            min_speakers_by_family=min_speakers_by_family,
        )

    quality_weight = feature_weights.get("quality", 0.0)
    speaker_diversity_weight = feature_weights.get("speaker_diversity", 0.0)
    if features_cache is None:
        features_cache = {clip.clip_id: dedupe_clip_features(clip) for clip in candidates}

    def score(clip: ClipFeatures, state: ConstraintState) -> float:
        return score_clip_fast(
            clip,
            feature_weights=feature_weights,
            diminishing_tau=diminishing_tau,
            target_tables=target_tables or {},
            current_counts_by_family=current_counts_by_family,
            selected_speakers=state.selected_speakers,
            quality_weight=quality_weight,
            speaker_diversity_weight=speaker_diversity_weight,
            features_by_family=features_cache.get(clip.clip_id),
        )

    return score


def _evaluate_set(
    selected: list[str],
    clips_by_id: dict[str, ClipFeatures],
    score_fn: ScoreFn,
    constraint_config: ConstraintConfig,
    *,
    feature_counts: dict[str, Counter[str]] | None = None,
    features_cache: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[float, ConstraintState]:
    state = ConstraintState()
    if feature_counts is not None:
        for counter in feature_counts.values():
            counter.clear()
    total = 0.0
    for clip_id in selected:
        clip = clips_by_id[clip_id]
        total += score_fn(clip, state)
        add_clip_to_state(clip, state, constraint_config)
        if feature_counts is not None:
            tokens_by_family = (
                features_cache.get(clip_id, clip.features_by_family)
                if features_cache is not None
                else clip.features_by_family
            )
            for family, tokens in tokens_by_family.items():
                bucket = feature_counts.setdefault(family, Counter())
                for token in tokens:
                    bucket[token] += 1
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
    progress: ProgressSink | None = None,
    progress_label: str | None = None,
    target_tables: dict[str, dict[str, float]] | None = None,
    seed: int = 0,
    required_coverage_targets: dict[str, int] | None = None,
) -> tuple[list[str], list[str], int]:
    if not selected or not reserve:
        return selected, reserve, 0

    # Bound combinatorial explosion on large corpora.
    selected_cap = 200
    reserve_cap = 200
    selected_work = list(selected[:selected_cap])
    reserve_work = list(reserve[:reserve_cap])

    features_cache = {clip.clip_id: dedupe_clip_features(clip) for clip in candidates}
    # Mutable feature counts updated inside _evaluate_set so diminishing-returns
    # scoring follows the trial selection order (same as greedy).
    live_counts: dict[str, Counter[str]] = {family: Counter() for family in feature_weights}
    required_targets = dict(required_coverage_targets or {})

    def coverage_ok(trial_selected: list[str], removed: list[str], added: list[str]) -> bool:
        if not required_targets:
            return True
        # Multi-swap: apply removals/additions sequentially against current counts.
        current = coverage_counts_for_selection(selected_set, clips_by_id, required_targets.keys())
        # Simulate full swap on counts
        for out_id in removed:
            out_clip = clips_by_id.get(out_id)
            if out_clip is None:
                continue
            for key in out_clip.coverage_keys:
                if key in current:
                    current[key] -= 1
        for in_id in added:
            in_clip = clips_by_id.get(in_id)
            if in_clip is None:
                continue
            for key in in_clip.coverage_keys:
                if key in current:
                    current[key] += 1
        for feature, minimum in required_targets.items():
            if current.get(feature, 0) < minimum:
                return False
        return True

    score_fn = _build_score_fn(
        feature_weights=feature_weights,
        diminishing_tau=diminishing_tau,
        temperatures=temperatures,
        candidates=candidates,
        min_utterances_by_family=min_utterances_by_family,
        min_speakers_by_family=min_speakers_by_family,
        current_counts_by_family=live_counts,
        target_tables=target_tables,
        features_cache=features_cache,
    )

    selected_set = list(selected_work)
    reserve_set = list(reserve_work)
    best_score, _ = _evaluate_set(
        selected_set,
        clips_by_id,
        score_fn,
        constraint_config,
        feature_counts=live_counts,
        features_cache=features_cache,
    )
    iterations = 0
    improvements = 0
    start = time.monotonic()
    last_progress_at = 0.0
    rng = random.Random(seed)
    # Cap pair evaluations per iteration; full cartesian is O(n^2) and can exceed wall
    # before the outer loop ever checks time.
    max_pair_checks = max(256, min(4000, selected_cap * 8))

    patterns = set(swap_patterns)
    # Expensive multi-swaps only briefly; prefer 1v1.
    allow_expensive = ("1v2" in patterns or "2v1" in patterns) and improvements < 3

    def timed_out() -> bool:
        return (time.monotonic() - start) >= max_wall_sec

    def report(*, force: bool = False) -> None:
        nonlocal last_progress_at
        now = time.monotonic()
        if not force and (now - last_progress_at) < 0.5:
            return
        last_progress_at = now
        if progress is None:
            return
        elapsed = now - start
        iter_frac = iterations / max_iterations if max_iterations > 0 else 0.0
        time_frac = elapsed / max_wall_sec if max_wall_sec > 0 else 0.0
        local_frac = max(iter_frac, time_frac)
        message = (
            f"local_search iter={iterations}/{max_iterations} "
            f"improvements={improvements} elapsed={elapsed:.1f}s/{max_wall_sec:.0f}s"
        )
        if progress_label:
            message = f"[{progress_label}] {message}"
        progress(
            ProgressEvent(
                stage="select",
                message=message,
                current=iterations,
                total=max_iterations,
                fraction=0.82 + 0.16 * min(1.0, local_frac),
                metadata={
                    "phase": "local_search",
                    "iterations": iterations,
                    "improvements": improvements,
                    "elapsed_sec": elapsed,
                    "max_wall_sec": max_wall_sec,
                    "label": progress_label,
                },
            )
        )

    report(force=True)

    while iterations < max_iterations and not timed_out():
        improved = False
        iterations += 1
        checks = 0

        if "1v1" in patterns:
            out_order = list(selected_set)
            in_order = list(reserve_set)
            rng.shuffle(out_order)
            rng.shuffle(in_order)
            for out_id in out_order:
                if timed_out():
                    break
                out_clip = clips_by_id.get(out_id)
                for in_id in in_order:
                    if timed_out() or checks >= max_pair_checks:
                        break
                    if out_id == in_id:
                        continue
                    in_clip = clips_by_id.get(in_id)
                    if out_clip is None or in_clip is None:
                        continue
                    checks += 1
                    if required_targets and not preserves_required_coverage(
                        coverage_counts_for_selection(
                            selected_set, clips_by_id, required_targets.keys()
                        ),
                        out_clip,
                        in_clip,
                        required_targets,
                    ):
                        continue
                    trial_selected = [cid for cid in selected_set if cid != out_id] + [in_id]
                    trial_score, _ = _evaluate_set(
                        trial_selected,
                        clips_by_id,
                        score_fn,
                        constraint_config,
                        feature_counts=live_counts,
                        features_cache=features_cache,
                    )
                    if trial_score > best_score + 1e-12:
                        selected_set = trial_selected
                        reserve_set = [cid for cid in reserve_set if cid != in_id] + [out_id]
                        best_score = trial_score
                        improved = True
                        improvements += 1
                        break
                if improved or timed_out() or checks >= max_pair_checks:
                    break
        if improved:
            report()
            continue
        if timed_out():
            break

        allow_expensive = ("1v2" in patterns or "2v1" in patterns) and improvements < 3
        if allow_expensive and "1v2" in patterns and len(reserve_set) >= 2:
            out_order = list(selected_set)
            rng.shuffle(out_order)
            for out_id in out_order:
                if timed_out() or checks >= max_pair_checks:
                    break
                pairs = list(combinations(reserve_set, 2))
                rng.shuffle(pairs)
                for in_a, in_b in pairs:
                    if timed_out() or checks >= max_pair_checks:
                        break
                    checks += 1
                    if not coverage_ok([cid for cid in selected_set if cid != out_id] + [in_a, in_b], [out_id], [in_a, in_b]):
                        continue
                    trial_selected = [cid for cid in selected_set if cid != out_id] + [in_a, in_b]
                    trial_score, _ = _evaluate_set(
                        trial_selected,
                        clips_by_id,
                        score_fn,
                        constraint_config,
                        feature_counts=live_counts,
                        features_cache=features_cache,
                    )
                    if trial_score > best_score + 1e-12:
                        selected_set = trial_selected
                        reserve_set = [
                            cid for cid in reserve_set if cid not in {in_a, in_b}
                        ] + [out_id]
                        best_score = trial_score
                        improved = True
                        improvements += 1
                        break
                if improved:
                    break
        if improved:
            report()
            continue
        if timed_out():
            break

        if allow_expensive and "2v1" in patterns and len(selected_set) >= 2:
            out_pairs = list(combinations(selected_set, 2))
            rng.shuffle(out_pairs)
            in_order = list(reserve_set)
            rng.shuffle(in_order)
            for out_a, out_b in out_pairs:
                if timed_out() or checks >= max_pair_checks:
                    break
                for in_id in in_order:
                    if timed_out() or checks >= max_pair_checks:
                        break
                    checks += 1
                    trial_selected = [
                        cid for cid in selected_set if cid not in {out_a, out_b}
                    ] + [in_id]
                    if len(trial_selected) != len(selected_set) - 1:
                        continue
                    if not coverage_ok(trial_selected, [out_a, out_b], [in_id]):
                        continue
                    trial_score, _ = _evaluate_set(
                        trial_selected,
                        clips_by_id,
                        score_fn,
                        constraint_config,
                        feature_counts=live_counts,
                        features_cache=features_cache,
                    )
                    if trial_score > best_score + 1e-12:
                        selected_set = trial_selected
                        reserve_set = [cid for cid in reserve_set if cid != in_id] + [
                            out_a,
                            out_b,
                        ]
                        best_score = trial_score
                        improved = True
                        improvements += 1
                        break
                if improved:
                    break

        report()
        if not improved:
            break

    report(force=True)

    # Reconcile with full selected/reserve lists outside the capped working sets.
    selected_tail = [cid for cid in selected if cid not in set(selected_work)]
    reserve_tail = [cid for cid in reserve if cid not in set(reserve_work)]
    final_selected = selected_set + selected_tail
    moved = set(selected_set) | set(reserve_set)
    final_reserve = [cid for cid in reserve_set if cid not in set(selected_set)] + [
        cid for cid in reserve_tail if cid not in moved and cid not in set(selected_set)
    ]
    return final_selected, final_reserve, iterations
