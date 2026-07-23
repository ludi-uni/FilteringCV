from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from cv_preprocess.application.select import _clip_features_from_catalog, _resolved_target_duration_sec
from cv_preprocess.compute.loader import resolve_compute_backend
from cv_preprocess.compute.profiling import measure_callable
from cv_preprocess.compute.protocol import SelectionState
from cv_preprocess.config import PipelineConfig
from cv_preprocess.selection.python_backend import (
    _eligible_candidates,
    _support_thresholds,
    _temperatures,
    greedy_local_search,
)
from cv_preprocess.selection.scoring import pool_counts_for_family, speaker_counts_for_family


def _default_benchmark_config() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "input": {"corpus_root": ".", "clip_tsv": "validated.tsv"},
            "dataset_builder": {
                "enabled": True,
                "work_dir": "work",
                "target_duration_hours": 1.0,
                "selection": {"local_search": {"enabled": False}},
            },
        }
    )


def benchmark_selection(
    catalog_path: Path,
    *,
    config: PipelineConfig | None = None,
    repeat: int = 3,
    backend: str = "auto",
) -> dict[str, Any]:
    """Measure selection scoring and greedy selection timings on a catalog parquet."""
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    cfg = config or _default_benchmark_config()
    clips_df = pl.read_parquet(catalog_path)
    candidates = _clip_features_from_catalog(clips_df, cfg, overrides={}, split_plan=None)
    compute = resolve_compute_backend(backend)

    db = cfg.dataset_builder
    selection = db.selection
    feature_weights = dict(selection.feature_weights)
    diminishing_tau = dict(selection.diminishing_return_tau)
    for family in feature_weights:
        diminishing_tau.setdefault(family, 1.0)
    temperatures = _temperatures(db)
    min_utterances, min_speakers = _support_thresholds(db.feature_support)
    eligible = _eligible_candidates(candidates)
    pool_counts_by_family = {
        family: pool_counts_for_family(eligible, family) for family in feature_weights
    }
    utterance_counts_by_family = pool_counts_by_family
    speaker_sets_by_family = {
        family: speaker_counts_for_family(eligible, family) for family in feature_weights
    }
    state = SelectionState(
        current_counts_by_family={family: Counter() for family in feature_weights},
    )
    target_duration_sec = _resolved_target_duration_sec(cfg)
    tolerance_ratio = selection.duration.tolerance_ratio

    scoring_runs: list[dict[str, Any]] = []
    for _ in range(repeat):
        started = time.perf_counter()
        scores = compute.score_candidates(
            eligible,
            feature_weights=feature_weights,
            diminishing_tau=diminishing_tau,
            temperatures=temperatures,
            pool_counts_by_family=pool_counts_by_family,
            utterance_counts_by_family=utterance_counts_by_family,
            speaker_sets_by_family=speaker_sets_by_family,
            min_utterances_by_family=min_utterances,
            min_speakers_by_family=min_speakers,
            state=state,
            quality_weight=feature_weights.get("quality", 0.0),
            speaker_diversity_weight=feature_weights.get("speaker_diversity", 0.0),
        )
        scoring_runs.append(
            {
                "wall_sec": time.perf_counter() - started,
                "candidate_count": len(eligible),
                "scored_count": len(scores),
            }
        )

    greedy_runs: list[dict[str, Any]] = []
    for index in range(repeat):
        _, timing = measure_callable(
            lambda: greedy_local_search(
                candidates,
                config=db,
                target_duration_sec=target_duration_sec,
                tolerance_ratio=tolerance_ratio,
                seed=db.random_seed + index,
            )
        )
        greedy_runs.append(
            {
                **timing,
                "candidate_count": len(candidates),
            }
        )

    aggregate_runs: list[dict[str, Any]] = []
    for _ in range(repeat):
        _, timing = measure_callable(
            lambda: (
                compute.count_features(clips_df),
                compute.build_duplicate_groups(clips_df),
            )
        )
        aggregate_runs.append(timing)

    return {
        "catalog": str(catalog_path.resolve()),
        "backend": compute.name,
        "repeat": repeat,
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "scoring": scoring_runs,
        "greedy_selection": greedy_runs,
        "catalog_aggregates": aggregate_runs,
    }
