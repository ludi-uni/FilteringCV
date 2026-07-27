from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from cv_preprocess.application.common import ProgressEvent, ProgressSink, SelectionPlan, SplitPlan
from cv_preprocess.application.split import finalize_clip_splits
from cv_preprocess.catalog import CatalogRef
from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.catalog.reader import read_clips
from cv_preprocess.config import PipelineConfig
from cv_preprocess.linguistic.features import FeatureSource, extract_linguistic_features
from cv_preprocess.selection.coverage_keys import coverage_keys_from_clip_parts
from cv_preprocess.selection.overrides import load_overrides, resolved_overrides_path
from cv_preprocess.selection.protocol import ClipFeatures, SelectionBackend, SelectionResult
from cv_preprocess.selection.python_backend import PythonSelectionBackend
from cv_preprocess.split.protocol import SPLIT_ORDER, SplitProtocol


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
        phoneme_str = str(row.get("phonemes") or "")
        text_norm = str(row.get("text_norm") or "")
        exclude_tokens = config.dataset_builder.feature_support.exclude_tokens
        coverage_keys: list[str] = []
        if phoneme_str or text_norm:
            coverage_keys = coverage_keys_from_clip_parts(
                normalized_text=text_norm,
                phoneme_str=phoneme_str,
                exclude_tokens=exclude_tokens,
            )
        acoustic_metrics: dict[str, float | None] = {
            "duration": float(row.get("duration_sec") or 0.0),
            "snr": float(row["estimated_snr_db"]) if row.get("estimated_snr_db") is not None else None,
            "estimated_snr_db": (
                float(row["estimated_snr_db"]) if row.get("estimated_snr_db") is not None else None
            ),
            "silence_ratio": (
                float(row["silence_ratio"]) if row.get("silence_ratio") is not None else None
            ),
            "quality_score": (
                float(row["quality_score"]) if row.get("quality_score") is not None else None
            ),
            "rms": float(row["rms"]) if row.get("rms") is not None else None,
            "peak": float(row["peak"]) if row.get("peak") is not None else None,
            "f0_median": float(row["f0_median"]) if row.get("f0_median") is not None else None,
            "f0_range": float(row["f0_range"]) if row.get("f0_range") is not None else None,
            "speech_rate": float(row["speech_rate"]) if row.get("speech_rate") is not None else None,
            "alignment_confidence": (
                float(row["alignment_confidence"])
                if row.get("alignment_confidence") is not None
                else None
            ),
        }
        clips.append(
            ClipFeatures(
                clip_id=clip_id,
                speaker_id=str(row.get("speaker_id") or ""),
                duration_sec=float(row.get("duration_sec") or 0.0),
                quality_score=row.get("quality_score"),
                audio_sha256=str(row.get("audio_sha256") or ""),
                sentence_id=str(row.get("sentence_id") or ""),
                text_norm=text_norm,
                features_by_family=_features_from_row(row, config),
                duplicate_groups=_duplicate_groups_from_row(row),
                override_action=override.action if override is not None else None,
                split=assignments.get(clip_id) or row.get("split"),
                coverage_keys=coverage_keys,
                acoustic_metrics=acoustic_metrics,
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
    *,
    index_candidate_counts: dict[str, int] | None = None,
    coverage_audit_output: Path | None = None,
) -> SelectionBackend:
    requested = backend or config.compute.backend
    if requested in {"auto", "python", "polars"}:
        return PythonSelectionBackend(
            config.dataset_builder,
            coverage_config=config.coverage,
            index_candidate_counts=index_candidate_counts,
            coverage_audit_output=coverage_audit_output,
        )
    raise ValueError(f"unsupported selection backend: {requested!r}")


def _load_index_candidate_counts(config: PipelineConfig) -> dict[str, int] | None:
    """Count feature keys from coverage clip-index.jsonl when available."""
    index_path = Path(config.coverage.output_dir) / "clip-index.jsonl"
    if not index_path.is_file():
        # GUI/active-run layout may keep index beside active-run
        alt = Path(config.coverage.output_dir) / "clip-index.jsonl"
        if not alt.is_file():
            return None
        index_path = alt
    try:
        from cv_preprocess.coverage.indexer import load_index_jsonl
    except Exception:
        return None
    counts: dict[str, int] = {}
    try:
        records = load_index_jsonl(index_path)
    except Exception:
        return None
    for record in records:
        for key in record.feature_key_set():
            counts[key] = counts.get(key, 0) + 1
    return counts


def _run_selection(
    selection_backend: SelectionBackend,
    candidates: list[ClipFeatures],
    *,
    config: PipelineConfig,
    split_plan: SplitPlan | None,
    target_duration_sec: float,
    tolerance_ratio: float,
    progress: ProgressSink | None = None,
) -> SelectionResult:
    protocol = SplitProtocol(config.dataset_builder.split.protocol)
    if (
        protocol == SplitProtocol.UNSEEN_SPEAKER
        and split_plan is not None
        and split_plan.speaker_assignments
    ):
        ratios = config.dataset_builder.split.resolved_ratios()
        by_split: dict[str, list[ClipFeatures]] = {name: [] for name in SPLIT_ORDER}
        for clip in candidates:
            split_name = split_plan.speaker_assignments.get(clip.speaker_id)
            if split_name:
                clip.split = split_name
                by_split.setdefault(split_name, []).append(clip)

        selected_ids: list[str] = []
        reserve_ids: list[str] = []
        explanations: dict[str, Any] = {}
        active_splits = [
            name
            for name in SPLIT_ORDER
            if by_split.get(name) and ratios.get(name, 0.0) > 0.0
        ]
        for split_idx, split_name in enumerate(active_splits, start=1):
            split_candidates = by_split.get(split_name, [])
            split_target = target_duration_sec * ratios.get(split_name, 0.0)
            if progress is not None:
                progress(
                    ProgressEvent(
                        stage="select",
                        message=(
                            f"split {split_name} ({split_idx}/{len(active_splits)}): "
                            f"{len(split_candidates)} candidates"
                        ),
                        current=split_idx - 1,
                        total=len(active_splits),
                        fraction=(split_idx - 1) / max(len(active_splits), 1),
                        metadata={"phase": "split", "split": split_name},
                    )
                )
            split_result = selection_backend.select(
                split_candidates,
                target_duration_sec=split_target,
                tolerance_ratio=tolerance_ratio,
                seed=config.dataset_builder.random_seed,
                progress=progress,
                progress_label=split_name,
            )
            selected_ids.extend(split_result.selected_ids)
            reserve_ids.extend(split_result.reserve_ids)
            explanations.update(split_result.explanations)
        return SelectionResult(
            selected_ids=selected_ids,
            reserve_ids=reserve_ids,
            explanations=explanations,
        )

    return selection_backend.select(
        candidates,
        target_duration_sec=target_duration_sec,
        tolerance_ratio=tolerance_ratio,
        seed=config.dataset_builder.random_seed,
        progress=progress,
    )


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
    # Explicit schema: selected rows often have null reserve_reason for far more than
    # Polars' default infer_schema_length (100), which then rejects later string values.
    schema = {
        "clip_id": pl.Utf8,
        "disposition": pl.Utf8,
        "split": pl.Utf8,
        "selection_score": pl.Float64,
        "selection_rank": pl.Int64,
        "positive_contributions": pl.Utf8,
        "penalties": pl.Utf8,
        "selected_reason": pl.Utf8,
        "reserve_reason": pl.Utf8,
    }
    _atomic_write_parquet(path, pl.DataFrame(rows, schema=schema))


def load_selection_plan(catalog: CatalogRef, plan_path: Path) -> SelectionPlan:
    plan_df = pl.read_parquet(plan_path)
    selected_ids = plan_df.filter(pl.col("disposition") == ClipDisposition.SELECTED.value)[
        "clip_id"
    ].to_list()
    reserve_ids = plan_df.filter(pl.col("disposition") == ClipDisposition.RESERVE.value)[
        "clip_id"
    ].to_list()
    return SelectionPlan(
        catalog=catalog,
        selected_clip_ids=[str(clip_id) for clip_id in selected_ids],
        reserve_clip_ids=[str(clip_id) for clip_id in reserve_ids],
        plan_path=plan_path,
    )


def select_dataset(
    config: PipelineConfig,
    catalog: CatalogRef,
    split_plan: SplitPlan | None,
    *,
    backend: str | None = None,
    progress: ProgressSink | None = None,
    coverage_aware: bool | None = None,
    coverage_policy: str | None = None,
    coverage_audit_output: Path | None = None,
    disable_acoustic_diversity: bool = False,
) -> SelectionPlan:
    if progress is not None:
        progress(
            ProgressEvent(
                stage="select",
                message="loading catalog",
                fraction=0.0,
                metadata={"phase": "load"},
            )
        )

    # CLI / caller overrides for coverage-aware select (mutate a copy of nested config).
    selection = config.dataset_builder.selection
    if coverage_aware is True:
        selection.coverage_constraints.enabled = True
    if coverage_policy is not None:
        selection.coverage_constraints.violation_policy = coverage_policy  # type: ignore[assignment]
        selection.coverage_constraints.policy = coverage_policy  # type: ignore[assignment]
    if disable_acoustic_diversity:
        selection.acoustic_diversity.enabled = False
        selection.acoustic_diversity.backend = "disabled"
        selection.feature_weights["acoustic_diversity"] = 0.0

    clips_df = read_clips(catalog.resolved_clips_path())
    overrides = load_overrides(resolved_overrides_path(catalog.work_dir))
    if progress is not None:
        progress(
            ProgressEvent(
                stage="select",
                message=f"building candidate features from {clips_df.height} clips",
                current=0,
                total=max(clips_df.height, 1),
                fraction=0.02,
                metadata={"phase": "features"},
            )
        )
    candidates = _clip_features_from_catalog(clips_df, config, overrides, split_plan)

    if progress is not None:
        progress(
            ProgressEvent(
                stage="select",
                message=f"running selection on {len(candidates)} candidates",
                current=0,
                total=max(len(candidates), 1),
                fraction=0.05,
                metadata={"phase": "start", "candidates": len(candidates)},
            )
        )

    index_counts = None
    if selection.coverage_constraints.enabled:
        index_counts = _load_index_candidate_counts(config)

    audit_out = coverage_audit_output
    if audit_out is None and selection.coverage_constraints.enabled:
        audit_out = Path(catalog.work_dir) / "reports" / "selection"

    selection_backend = _resolve_backend(
        config,
        backend,
        index_candidate_counts=index_counts,
        coverage_audit_output=audit_out,
    )
    target_duration_sec = _resolved_target_duration_sec(config)
    tolerance_ratio = config.dataset_builder.selection.duration.tolerance_ratio
    result = _run_selection(
        selection_backend,
        candidates,
        config=config,
        split_plan=split_plan,
        target_duration_sec=target_duration_sec,
        tolerance_ratio=tolerance_ratio,
        progress=progress,
    )

    if split_plan is None:
        placeholder_split = SplitPlan(
            catalog=catalog,
            protocol=config.dataset_builder.split.protocol,
            ratios=config.dataset_builder.split.resolved_ratios(),
        )
    else:
        placeholder_split = finalize_clip_splits(
            config,
            split_plan,
            result.selected_ids,
            clips_df=clips_df,
        )

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
                message=(
                    f"wrote selection plan selected={len(result.selected_ids)} "
                    f"reserve={len(result.reserve_ids)}"
                ),
                current=len(result.selected_ids),
                total=max(len(candidates), 1),
                fraction=1.0,
                metadata={
                    "phase": "complete",
                    "selected": len(result.selected_ids),
                    "reserve": len(result.reserve_ids),
                    "coverage_warnings": list(result.warnings),
                    "coverage_report_paths": dict(result.coverage_report_paths),
                },
            )
        )

    return SelectionPlan(
        catalog=catalog,
        selected_clip_ids=result.selected_ids,
        reserve_clip_ids=result.reserve_ids,
        plan_path=plan_path,
    )
