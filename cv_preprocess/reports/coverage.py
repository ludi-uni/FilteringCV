from __future__ import annotations

import math
from collections.abc import Mapping

import polars as pl

from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.reports.models import CoverageReport, FeatureCoverageEntry


def _to_probabilities(values: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(values.values()))
    if total <= 0:
        return {}
    return {key: float(value) / total for key, value in sorted(values.items())}


def js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Deterministic Jensen-Shannon divergence between two discrete distributions."""
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0

    p_prob = _to_probabilities({key: p.get(key, 0.0) for key in keys})
    q_prob = _to_probabilities({key: q.get(key, 0.0) for key in keys})
    m_prob = {key: 0.5 * (p_prob.get(key, 0.0) + q_prob.get(key, 0.0)) for key in keys}

    def _kl(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        total = 0.0
        for key in keys:
            a_val = a.get(key, 0.0)
            if a_val <= 0.0:
                continue
            b_val = b.get(key, 0.0)
            if b_val <= 0.0:
                continue
            total += a_val * math.log(a_val / b_val)
        return total

    return 0.5 * _kl(p_prob, m_prob) + 0.5 * _kl(q_prob, m_prob)


def js_distance(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Square-root of Jensen-Shannon divergence (a metric)."""
    return math.sqrt(max(js_divergence(p, q), 0.0))


def _uniform_distribution(keys: list[str]) -> dict[str, float]:
    if not keys:
        return {}
    weight = 1.0 / len(keys)
    return {key: weight for key in keys}


def compute_coverage_summary(
    clips: pl.DataFrame,
    feature_counts: pl.DataFrame,
) -> CoverageReport:
    """Summarize pool coverage from clips and feature_counts parquet."""
    eligible_value = ClipDisposition.ELIGIBLE.value
    total_clips = clips.height
    eligible_clips = int(clips.filter(pl.col("disposition") == eligible_value).height)

    if feature_counts.is_empty():
        return CoverageReport(
            total_clips=total_clips,
            eligible_clips=eligible_clips,
        )

    entries = [
        FeatureCoverageEntry(
            feature_type=str(row["feature_type"]),
            feature=str(row["feature"]),
            pool_count=int(row["count"]),
            pool_speaker_count=int(row["speaker_count"]),
            pool_utterance_count=int(row["utterance_count"]),
        )
        for row in feature_counts.sort(["feature_type", "feature"]).iter_rows(named=True)
    ]
    feature_types = sorted({entry.feature_type for entry in entries})

    js_distance_to_uniform: dict[str, float] = {}
    for feature_type in feature_types:
        subset = feature_counts.filter(pl.col("feature_type") == feature_type)
        observed = {
            str(row["feature"]): float(row["count"])
            for row in subset.iter_rows(named=True)
        }
        uniform = _uniform_distribution(sorted(observed))
        js_distance_to_uniform[feature_type] = js_distance(observed, uniform)

    return CoverageReport(
        total_clips=total_clips,
        eligible_clips=eligible_clips,
        feature_types=feature_types,
        unique_features=len(entries),
        entries=entries,
        js_distance_to_uniform=js_distance_to_uniform,
    )
