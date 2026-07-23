from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from cv_preprocess.application.common import ProgressEvent, ProgressSink, SelectionPlan, SplitPlan
from cv_preprocess.catalog import CatalogRef
from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.catalog.reader import read_clips
from cv_preprocess.config import PipelineConfig
from cv_preprocess.linguistic.features import FeatureSource, extract_linguistic_features
from cv_preprocess.selection.overrides import load_overrides, resolved_overrides_path
from cv_preprocess.selection.protocol import ClipFeatures, SelectionBackend
from cv_preprocess.selection.python_backend import PythonSelectionBackend


def _atomic_write_parquet(path: Path, df: pl.DataFrame) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(partial)
    partial.replace(path)


def _duplicate_groups_from_row(row: dict[str, Any]) -> dict[str, str]:
    groups: dict[str, str] = {}
    if row.get("audio_sha256"):
        groups["exact_audio"] = str(row["audio_sha256"])
    if row.get("normalized_relative_source_path"):
        groups["same_source_path"] = str(row["normalized_relative_source_path"])
    if row.get("sentence_id"):
        groups["same_sentence_id"] = str(row["sentence_id"])
    if row.get("text_norm"):
        groups["same_normalized_text"] = str(row["text_norm"])
    speaker = row.get("speaker_id")
    text_norm = row.get("text_norm")
    if speaker and text_norm:
        groups["same_speaker_same_text"] = f"{speaker}:{text_norm}"
    return groups


def _features_from_row(row: dict[str, Any], config: PipelineConfig) -> dict[str, list[str]]:
    feature_support = config.dataset_builder.feature_support
    source_raw = row.get("feature_source") or "none"
    try:
        source = FeatureSource(source_raw)
    except ValueError:
        source = FeatureSource.TEXT_G2P

    linguistic = extract_linguistic_features(
        str(row.get("text_norm") or row.get("text_raw") or ""),
        row.get("phonemes"),
        feature_source=source,
        duration_sec=row.get("duration_sec"),
        exclude_tokens=feature_support.exclude_tokens,
        down_weight_tokens=feature_support.down_weight_tokens,
    )
    features: dict[str, list[str]] = {
        "phone": linguistic.phones,
        "biphone": linguistic.biphones,
        "triphone": linguistic.triphones,
        "mora": linguistic.morae,
        "mora_bigram": linguistic.mora_bigrams,
    }
    if linguistic.full_context_labels:
        features["full_context"] = linguistic.full_context_labels
    if linguistic.accent_nucleus_features:
        features["accent_nucleus"] = linguistic.accent_nucleus_features
    if linguistic.accent_phrase_length_bands:
        features["accent_phrase_length"] = linguistic.accent_phrase_length_bands
    if linguistic.pause_boundary_markers:
        features["pause_boundary"] = linguistic.pause_boundary_markers
    if linguistic.sentence_length_band:
        features["sentence_length_band"] = [linguistic.sentence_length_band]
    if linguistic.speaking_rate_band:
        features["speaking_rate_band"] = [linguistic.speaking_rate_band]
    if linguistic.utterance_type:
        features["interrogative_declarative"] = [linguistic.utterance_type]
    return features


def _clip_features_from_catalog(
    df: pl.DataFrame,
    config: PipelineConfig,
    overrides: dict[str, Any],
    split_plan: SplitPlan | None,
) -> list[ClipFeatures]:
    assignments = split_plan.assignments if split_plan is not None else {}
    clips: list[ClipFeatures] = []
    for row in df.iter_rows(named=True):
        disposition = str(row.get("disposition") or "")
        if disposition == ClipDisposition.HARD_REJECTED.value:
            continue
        if disposition not in {ClipDisposition.ELIGIBLE.value, ClipDisposition.SELECTED.value, ClipDisposition.RESERVE.value}:
            if disposition:
                continue
        clip_id = str(row["clip_id"])
        override = overrides.get(clip_id)
        clips.append(
            ClipFeatures(
                clip_id=clip_id,
                speaker_id=str(row.get("speaker_id") or ""),
                duration_sec=float(row.get("duration_sec") or 0.0),
                quality_score=row.get("quality_score"),
                audio_sha256=str(row.get("audio_sha256") or ""),
                sentence_id=str(row.get("sentence_id") or ""),
                text_norm=str(row.get("text_norm") or ""),
                features_by_family=_features_from_row(row, config),
                duplicate_groups=_duplicate_groups_from_row(row),
                override_action=override.action if override is not None else None,
                split=assignments.get(clip_id) or row.get("split"),
            )
        )
    return clips


def _resolved_target_duration_sec(config: PipelineConfig) -> float:
    db = config.dataset_builder
    hours = db.target_duration_hours
    if hours is None:
        hours = db.selection.duration.target_hours
    if hours is None:
        hours = 1.0
    return hours * 3600.0


def _resolve_backend(
    config: PipelineConfig,
    backend: str | None,
) -> SelectionBackend:
    requested = backend or config.compute.backend
    if requested in {"auto", "python", "polars"}:
        return PythonSelectionBackend(config.dataset_builder)
    raise ValueError(f"unsupported selection backend: {requested!r}")


def write_selection_plan_parquet(
    path: Path,
    *,
    selected_ids: list[str],
    reserve_ids: list[str],
    explanations: dict[str, Any],
    split_plan: SplitPlan | None,
) -> None:
    assignments = split_plan.assignments if split_plan is not None else {}
    rows: list[dict[str, Any]] = []
    for rank, clip_id in enumerate(selected_ids, start=1):
        explanation = explanations.get(clip_id)
        rows.append(
            {
                "clip_id": clip_id,
                "disposition": ClipDisposition.SELECTED.value,
                "split": assignments.get(clip_id),
                "selection_score": getattr(explanation, "selection_score", None),
                "selection_rank": rank,
                "positive_contributions": json.dumps(
                    getattr(explanation, "positive_contributions", {}), ensure_ascii=False
                ),
                "penalties": json.dumps(getattr(explanation, "penalties", {}), ensure_ascii=False),
                "selected_reason": getattr(explanation, "selected_reason", None),
                "reserve_reason": getattr(explanation, "reserve_reason", None),
            }
        )
    for clip_id in reserve_ids:
        explanation = explanations.get(clip_id)
        rows.append(
            {
                "clip_id": clip_id,
                "disposition": ClipDisposition.RESERVE.value,
                "split": assignments.get(clip_id),
                "selection_score": getattr(explanation, "selection_score", None),
                "selection_rank": getattr(explanation, "rank", None),
                "positive_contributions": json.dumps(
                    getattr(explanation, "positive_contributions", {}), ensure_ascii=False
                ),
                "penalties": json.dumps(getattr(explanation, "penalties", {}), ensure_ascii=False),
                "selected_reason": getattr(explanation, "selected_reason", None),
                "reserve_reason": getattr(explanation, "reserve_reason", None),
            }
        )
    _atomic_write_parquet(path, pl.DataFrame(rows))


def select_dataset(
    config: PipelineConfig,
    catalog: CatalogRef,
    split_plan: SplitPlan | None,
    *,
    backend: str | None = None,
    progress: ProgressSink | None = None,
) -> SelectionPlan:
    if progress is not None:
        progress(
            ProgressEvent(
                stage="select",
                message="loading catalog",
            )
        )

    clips_df = read_clips(catalog.resolved_clips_path())
    overrides = load_overrides(resolved_overrides_path(catalog.work_dir))
    candidates = _clip_features_from_catalog(clips_df, config, overrides, split_plan)

    if progress is not None:
        progress(
            ProgressEvent(
                stage="select",
                message="running selection",
                total=len(candidates),
            )
        )

    selection_backend = _resolve_backend(config, backend)
    target_duration_sec = _resolved_target_duration_sec(config)
    tolerance_ratio = config.dataset_builder.selection.duration.tolerance_ratio
    result = selection_backend.select(
        candidates,
        target_duration_sec=target_duration_sec,
        tolerance_ratio=tolerance_ratio,
        seed=config.dataset_builder.random_seed,
    )

    if split_plan is None or not split_plan.assignments:
        placeholder_split = SplitPlan(
            catalog=catalog,
            protocol=config.dataset_builder.split.protocol,
            assignments={clip_id: "train" for clip_id in result.selected_ids},
        )
    else:
        placeholder_split = split_plan

    plans_dir = Path(catalog.work_dir) / "plans"
    plan_path = plans_dir / "selection_plan.parquet"
    write_selection_plan_parquet(
        plan_path,
        selected_ids=result.selected_ids,
        reserve_ids=result.reserve_ids,
        explanations=result.explanations,
        split_plan=placeholder_split,
    )

    if progress is not None:
        progress(
            ProgressEvent(
                stage="select",
                message="selection complete",
                current=len(result.selected_ids),
                total=len(candidates),
            )
        )

    return SelectionPlan(
        catalog=catalog,
        selected_clip_ids=result.selected_ids,
        reserve_clip_ids=result.reserve_ids,
        plan_path=plan_path,
    )
