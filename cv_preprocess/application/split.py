from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cv_preprocess.application.common import ProgressEvent, ProgressSink, SplitPlan
from cv_preprocess.catalog import CatalogRef
from cv_preprocess.catalog.reader import read_clips
from cv_preprocess.config import PipelineConfig
from cv_preprocess.linguistic.features import FeatureSource, extract_linguistic_features
from cv_preprocess.reports.serializer import write_json_atomic
from cv_preprocess.split.leakage import ClipSplitRecord
from cv_preprocess.split.protocol import SplitProtocol
from cv_preprocess.split.seen_speaker import assign_clip_splits
from cv_preprocess.split.single_speaker import assign_single_speaker_splits
from cv_preprocess.split.unseen_speaker import plan_unseen_speaker_splits


def write_split_plan(path: Path, plan: SplitPlan) -> None:
    write_json_atomic(
        path,
        {
            "protocol": plan.protocol,
            "speaker_assignments": plan.speaker_assignments,
            "clip_assignments": plan.clip_assignments,
            "assignments": plan.assignments,
            "ratios": plan.ratios,
            "warnings": plan.warnings,
        },
    )


def load_split_plan(catalog: CatalogRef, path: Path) -> SplitPlan:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    speaker_assignments = {
        str(k): str(v) for k, v in (data.get("speaker_assignments") or {}).items()
    }
    clip_assignments = {
        str(k): str(v) for k, v in (data.get("clip_assignments") or data.get("assignments") or {}).items()
    }
    return SplitPlan(
        catalog=catalog,
        protocol=str(data.get("protocol") or "unseen_speaker"),
        speaker_assignments=speaker_assignments,
        clip_assignments=clip_assignments,
        assignments=clip_assignments,
        ratios={str(k): float(v) for k, v in (data.get("ratios") or {}).items()},
        warnings=[str(item) for item in data.get("warnings") or []],
        plan_path=path,
    )


def _features_from_row(row: dict, config: PipelineConfig) -> dict[str, list[str]]:
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
    return features


def _clip_records_from_df(
    clips_df: pl.DataFrame,
    clip_ids: list[str],
    config: PipelineConfig,
) -> list[ClipSplitRecord]:
    selected = clips_df.filter(pl.col("clip_id").is_in(clip_ids))
    records: list[ClipSplitRecord] = []
    for row in selected.iter_rows(named=True):
        records.append(
            ClipSplitRecord(
                clip_id=str(row["clip_id"]),
                speaker_id=str(row.get("speaker_id") or ""),
                audio_hash=str(row.get("audio_sha256") or ""),
                sentence_id=str(row.get("sentence_id") or ""),
                normalized_text=str(row.get("text_norm") or ""),
                duration_sec=float(row.get("duration_sec") or 0.0),
                features_by_family=_features_from_row(row, config),
            )
        )
    return records


def finalize_clip_splits(
    config: PipelineConfig,
    split_plan: SplitPlan,
    selected_clip_ids: list[str],
    *,
    clips_df: pl.DataFrame | None = None,
) -> SplitPlan:
    protocol = SplitProtocol(split_plan.protocol)
    split_config = config.dataset_builder.split
    ratios = split_config.resolved_ratios()
    warnings = list(split_plan.warnings)

    if protocol == SplitProtocol.UNSEEN_SPEAKER:
        if clips_df is None:
            clips_df = read_clips(split_plan.catalog.resolved_clips_path())
        clip_assignments: dict[str, str] = {}
        for row in clips_df.filter(pl.col("clip_id").is_in(selected_clip_ids)).iter_rows(named=True):
            speaker = str(row.get("speaker_id") or "")
            split_name = split_plan.speaker_assignments.get(speaker, "train")
            clip_assignments[str(row["clip_id"])] = split_name
        return SplitPlan(
            catalog=split_plan.catalog,
            protocol=split_plan.protocol,
            speaker_assignments=dict(split_plan.speaker_assignments),
            clip_assignments=clip_assignments,
            assignments=clip_assignments,
            ratios=ratios,
            warnings=warnings,
            plan_path=split_plan.plan_path,
        )

    if clips_df is None:
        clips_df = read_clips(split_plan.catalog.resolved_clips_path())
    clip_records = _clip_records_from_df(clips_df, selected_clip_ids, config)
    if protocol == SplitProtocol.SINGLE_SPEAKER:
        clip_assignments, split_warnings = assign_single_speaker_splits(clip_records, split_config)
    else:
        clip_assignments, split_warnings = assign_clip_splits(
            clip_records,
            split_config,
            protocol=protocol,
        )
    warnings.extend(split_warnings)
    return SplitPlan(
        catalog=split_plan.catalog,
        protocol=split_plan.protocol,
        speaker_assignments=dict(split_plan.speaker_assignments),
        clip_assignments=clip_assignments,
        assignments=clip_assignments,
        ratios=ratios,
        warnings=warnings,
        plan_path=split_plan.plan_path,
    )


def plan_dataset_split(
    config: PipelineConfig,
    catalog: CatalogRef,
    *,
    progress: ProgressSink | None = None,
) -> SplitPlan:
    if progress is not None:
        progress(ProgressEvent(stage="plan-split", message="creating split plan"))

    split_config = config.dataset_builder.split
    protocol = SplitProtocol(split_config.protocol)
    ratios = split_config.resolved_ratios()
    clips_df = read_clips(catalog.resolved_clips_path())
    feature_counts_path = catalog.feature_counts_path or (catalog.work_dir / "catalog" / "feature_counts.parquet")
    feature_counts = pl.read_parquet(feature_counts_path) if feature_counts_path.is_file() else None

    warnings: list[str] = []
    speaker_assignments: dict[str, str] = {}
    clip_assignments: dict[str, str] = {}

    if protocol == SplitProtocol.UNSEEN_SPEAKER:
        speaker_assignments, warnings = plan_unseen_speaker_splits(
            clips_df,
            split_config,
            feature_counts=feature_counts,
        )

    plan = SplitPlan(
        catalog=catalog,
        protocol=split_config.protocol,
        speaker_assignments=speaker_assignments,
        clip_assignments=clip_assignments,
        assignments=clip_assignments,
        ratios=ratios,
        warnings=warnings,
    )
    plans_dir = Path(catalog.work_dir) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plans_dir / "split_plan.json"
    write_split_plan(plan_path, plan)
    plan.plan_path = plan_path
    return plan
