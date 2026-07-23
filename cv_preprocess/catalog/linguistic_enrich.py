from __future__ import annotations

from typing import Any

from cv_preprocess.config import PipelineConfig
from cv_preprocess.linguistic.features import FeatureSource, extract_linguistic_features


def enrich_row_with_linguistic_features(
    row: dict[str, Any],
    *,
    text_norm: str,
    phonemes: str | None,
    duration_sec: float | None,
    config: PipelineConfig,
) -> list[str]:
    """Populate linguistic list columns on a catalog row. Returns per-clip warnings."""
    feature_source = FeatureSource.TEXT_G2P
    features = extract_linguistic_features(
        text_norm,
        phonemes,
        feature_source=feature_source,
        duration_sec=duration_sec,
        exclude_tokens=config.dataset_builder.feature_support.exclude_tokens or None,
        down_weight_tokens=config.dataset_builder.feature_support.down_weight_tokens or None,
    )
    row["biphones"] = features.biphones
    row["triphones"] = features.triphones
    row["moras"] = features.morae
    row["fullcontext_labels"] = features.full_context_labels
    row["feature_source"] = feature_source.value
    warnings: list[str] = []
    if features.full_context_warning:
        warnings.append(features.full_context_warning)
    row["analysis_warnings"] = warnings or None
    return warnings
