"""Coverage automation iterative runner."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cv_preprocess.application.analyze import ClipInput, analyze_clips
from cv_preprocess.application.common import CancellationToken, ProgressEvent, ProgressSink
from cv_preprocess.config.coverage import CoverageAutomationConfig
from cv_preprocess.config.pipeline import PipelineConfig
from cv_preprocess.coverage.counter import load_accepted_counts
from cv_preprocess.coverage.deficits import compute_deficits, remaining_required_deficits, total_deficit
from cv_preprocess.coverage.indexer import config_hash_for_coverage, load_index_jsonl, meta_path_for_index
from cv_preprocess.coverage.models import (
    AnalysisBatchResult,
    ClipIndexMeta,
    ClipIndexRecord,
    CoverageRunState,
    SpeakerPassStats,
    StopReason,
)
from cv_preprocess.coverage.planner import plan_coverage
from cv_preprocess.coverage.report import write_coverage_reports
from cv_preprocess.coverage.state import append_jsonl, load_run_state, save_run_state
from cv_preprocess.io.tsv_loader import ClipRow, load_clip_rows_for_pipeline
from cv_preprocess.reports.serializer import write_json_atomic

logger = logging.getLogger(__name__)

AnalyzeFn = Callable[..., AnalysisBatchResult]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_relative_path(path: str) -> str:
    return path.replace("\\", "/")


def _load_index_meta(index_path: Path) -> ClipIndexMeta | None:
    meta_path = meta_path_for_index(index_path)
    if not meta_path.is_file():
        return None
    return ClipIndexMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))


def _rows_by_source_index(config: PipelineConfig) -> dict[int, ClipRow]:
    loaded = load_clip_rows_for_pipeline(
        config,
        apply_input_max_clips=False,
        apply_speaker_merge=True,
        sort_by_path=False,
    )
    indexed = list(enumerate(loaded.rows))
    indexed.sort(key=lambda item: (_normalize_relative_path(item[1].path), item[0]))
    return {source_row_index: row for source_row_index, row in indexed}


def _clip_row_from_index_record(record: ClipIndexRecord) -> ClipRow:
    return ClipRow(
        client_id=record.client_id,
        path=record.source_path,
        sentence=record.sentence,
        raw={
            "client_id": record.client_id,
            "path": record.source_path,
            "sentence": record.sentence,
            "up_votes": str(record.up_votes) if record.up_votes is not None else "",
            "down_votes": str(record.down_votes) if record.down_votes is not None else "",
        },
    )


def _clip_inputs_for_batch(
    selected: Sequence[ClipIndexRecord],
    *,
    config: PipelineConfig,
    prefer_corpus_rows: bool,
) -> list[ClipInput]:
    rows_by_index: dict[int, ClipRow] = {}
    if prefer_corpus_rows:
        try:
            rows_by_index = _rows_by_source_index(config)
        except FileNotFoundError:
            rows_by_index = {}
    inputs: list[ClipInput] = []
    for record in selected:
        row = rows_by_index.get(record.source_row_index) or _clip_row_from_index_record(record)
        inputs.append(ClipInput(row=row, source_row_index=record.source_row_index, clip_id=record.clip_id))
    return inputs


def _accepted_counts_for_state(
    *,
    accepted_metadata: Path | None,
    catalog_clips: Path | None,
    index_records: Sequence[ClipIndexRecord],
    state: CoverageRunState,
) -> dict[str, int]:
    base = load_accepted_counts(
        accepted_metadata=accepted_metadata,
        catalog_clips=catalog_clips,
    )
    # Fold in accepted clips from this run that may not yet be in external metadata.
    if state.accepted_clip_ids:
        from cv_preprocess.coverage.counter import count_from_index_records

        run_counts = count_from_index_records(index_records, clip_ids=set(state.accepted_clip_ids))
        merged = dict(base)
        for key, value in run_counts.items():
            merged[key] = max(int(merged.get(key, 0)), int(value))
        # Prefer sum when metadata empty: use union counting via index accepted set + metadata ids
        if not base:
            return run_counts
        # When both exist, recount from union of clip ids if catalog unavailable
        if catalog_clips is None or not Path(catalog_clips).is_file():
            # Approximate: take max per feature (safe lower-bound of deficit reduction)
            return merged
        return base
    return base


def _update_pass_stats(state: CoverageRunState, batch: AnalysisBatchResult, records_by_id: Mapping[str, ClipIndexRecord]) -> None:
    for item in batch.accepted:
        if item.clip_id in {s.clip_id for s in batch.skipped}:
            continue
        state.global_pass_attempts += 1
        state.global_pass_passes += 1
        client_id = item.client_id or records_by_id.get(item.clip_id, ClipIndexRecord(clip_id=item.clip_id, source_path="", client_id="", sentence="", normalized_text="")).client_id
        stats = state.speaker_pass_stats.get(client_id) or SpeakerPassStats()
        stats.attempts += 1
        stats.passes += 1
        state.speaker_pass_stats[client_id] = stats
        if item.duration_sec:
            state.analyzed_audio_sec += float(item.duration_sec)
    for item in batch.rejected:
        if item.clip_id in {s.clip_id for s in batch.skipped}:
            continue
        state.global_pass_attempts += 1
        client_id = item.client_id or ""
        stats = state.speaker_pass_stats.get(client_id) or SpeakerPassStats()
        stats.attempts += 1
        state.speaker_pass_stats[client_id] = stats
        if item.duration_sec:
            state.analyzed_audio_sec += float(item.duration_sec)


def _decide_stop(
    *,
    config: CoverageAutomationConfig,
    state: CoverageRunState,
    required_deficits: Mapping[str, int],
    plan_stop_hints: Sequence[StopReason],
    selected_count: int,
) -> StopReason | None:
    if total_deficit(required_deficits) <= 0:
        return StopReason.COMPLETE
    if state.iteration >= config.limits.max_iterations:
        return StopReason.ITERATION_LIMIT_REACHED
    if len(state.analyzed_clip_ids) >= config.limits.max_analyzed_clips:
        return StopReason.ANALYSIS_BUDGET_EXCEEDED
    if state.analyzed_audio_sec / 3600.0 >= config.limits.max_audio_hours:
        return StopReason.ANALYSIS_BUDGET_EXCEEDED
    if StopReason.CANDIDATE_EXHAUSTED in plan_stop_hints and selected_count == 0:
        return StopReason.CANDIDATE_EXHAUSTED
    if StopReason.UNREACHABLE in plan_stop_hints and selected_count == 0:
        return StopReason.UNREACHABLE
    if selected_count == 0:
        return StopReason.CANDIDATE_EXHAUSTED
    return None


def run_coverage(
    config: PipelineConfig,
    *,
    index_path: Path,
    output_dir: Path,
    accepted_metadata: Path | None = None,
    resume: bool = False,
    dry_run: bool = False,
    max_iterations: int | None = None,
    max_clips: int | None = None,
    batch_size: int | None = None,
    analyze_fn: AnalyzeFn | None = None,
    progress: ProgressSink | None = None,
    cancellation: CancellationToken | None = None,
) -> CoverageRunState:
    coverage = config.coverage
    if not coverage.enabled:
        raise ValueError("coverage.enabled must be true to run coverage automation")

    def _emit(
        message: str,
        *,
        current: int | None = None,
        total: int | None = None,
        fraction: float | None = None,
        **metadata: Any,
    ) -> None:
        if progress is None:
            return
        progress(
            ProgressEvent(
                stage="coverage-run",
                message=message,
                current=current,
                total=total,
                fraction=fraction,
                metadata={"phase": "run", **metadata},
            )
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cancellation is not None:
        cancellation.raise_if_cancelled()
    _emit("loading index", metadata={"index_path": str(index_path)})
    index_records = load_index_jsonl(index_path)
    records_by_id = {record.clip_id: record for record in index_records}
    index_meta = _load_index_meta(index_path)
    cfg_hash = config_hash_for_coverage(config)
    fingerprint = index_meta.source_fingerprint if index_meta else ""

    catalog_clips = config.dataset_builder.work_dir / "catalog" / "clips.parquet"
    if accepted_metadata is not None and accepted_metadata.suffix.lower() == ".parquet":
        catalog_clips = accepted_metadata
        accepted_metadata = None

    if resume:
        state = load_run_state(output_dir)
        if state.config_hash and state.config_hash != cfg_hash:
            raise ValueError(
                f"coverage resume refused: config hash mismatch "
                f"(state={state.config_hash[:12]}… current={cfg_hash[:12]}…)"
            )
        if state.index_fingerprint and fingerprint and state.index_fingerprint != fingerprint:
            raise ValueError(
                f"coverage resume refused: index fingerprint mismatch "
                f"(state={state.index_fingerprint[:12]}… current={fingerprint[:12]}…)"
            )
        state.status = StopReason.RUNNING
    else:
        state = CoverageRunState(
            run_id=str(uuid.uuid4()),
            started_at=_utc_now(),
            updated_at=_utc_now(),
            config_hash=cfg_hash,
            index_fingerprint=fingerprint,
            status=StopReason.RUNNING,
        )

    if max_iterations is not None:
        coverage.limits.max_iterations = max_iterations
    if max_clips is not None:
        coverage.limits.max_analyzed_clips = max_clips

    analyze = analyze_fn or analyze_clips
    prefer_corpus_rows = analyze_fn is None

    max_iters = int(coverage.limits.max_iterations)
    while True:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        accepted_counts = _accepted_counts_for_state(
            accepted_metadata=accepted_metadata,
            catalog_clips=catalog_clips if catalog_clips.is_file() else None,
            index_records=index_records,
            state=state,
        )
        targets = coverage.iter_active_targets()
        deficits = compute_deficits(targets, accepted_counts)
        required = remaining_required_deficits(coverage, deficits)
        state.current_coverage = dict(accepted_counts)
        state.remaining_deficits = dict(required)

        iter_num = state.iteration + 1
        _emit(
            f"planning iteration {iter_num}",
            current=state.iteration,
            total=max_iters,
            fraction=min(1.0, state.iteration / max(max_iters, 1)),
            iteration=iter_num,
            remaining_deficit_total=total_deficit(required),
            analyzed=len(state.analyzed_clip_ids),
            accepted=len(state.accepted_clip_ids),
        )

        plan = plan_coverage(
            config=coverage,
            index_records=index_records,
            accepted_counts=accepted_counts,
            analyzed_clip_ids=set(state.analyzed_clip_ids),
            speaker_stats=state.speaker_pass_stats,
            global_attempts=state.global_pass_attempts,
            global_passes=state.global_pass_passes,
            batch_size_override=batch_size,
        )

        logger.info(
            "coverage iteration=%s required_features=%s satisfied_features=%s "
            "remaining_deficit_total=%s candidate_pool=%s selected_batch=%s estimated_pass_rate=%.2f",
            state.iteration + 1,
            len(required),
            sum(1 for feature, target in targets.items() if accepted_counts.get(feature, 0) >= target),
            total_deficit(required),
            plan.candidate_pool_size,
            plan.batch_size,
            plan.estimated_pass_rate,
        )

        stop = _decide_stop(
            config=coverage,
            state=state,
            required_deficits=required,
            plan_stop_hints=plan.stop_hints,
            selected_count=len(plan.selected),
        )
        if total_deficit(required) <= 0:
            state.status = StopReason.COMPLETE
            state.updated_at = _utc_now()
            save_run_state(output_dir, state)
            write_coverage_reports(output_dir, state=state, plan=plan, config=coverage)
            break
        if stop is not None:
            state.status = stop
            state.updated_at = _utc_now()
            save_run_state(output_dir, state)
            write_coverage_reports(output_dir, state=state, plan=plan, config=coverage)
            if stop == StopReason.CANDIDATE_EXHAUSTED:
                for feature, deficit in required.items():
                    if deficit <= 0:
                        continue
                    logger.info(
                        "coverage stopped reason=candidate_exhausted feature=%s target=%s accepted=%s remaining_candidates=0",
                        feature,
                        targets.get(feature),
                        accepted_counts.get(feature, 0),
                    )
            break

        state.iteration += 1
        append_jsonl(
            output_dir / "selected-batches.jsonl",
            {
                "iteration": state.iteration,
                "selected": plan.to_dict()["selected"],
                "batch_size": plan.batch_size,
            },
        )

        if dry_run:
            state.status = StopReason.DRY_RUN
            state.updated_at = _utc_now()
            save_run_state(output_dir, state)
            write_json_atomic(output_dir / "dry-run-plan.json", plan.to_dict())
            write_coverage_reports(output_dir, state=state, plan=plan, config=coverage)
            break

        clip_inputs = _clip_inputs_for_batch(
            plan.selected,
            config=config,
            prefer_corpus_rows=prefer_corpus_rows and not dry_run,
        )

        before_satisfied = {
            feature for feature, target in targets.items() if accepted_counts.get(feature, 0) >= target
        }
        _emit(
            f"analyzing batch iteration {state.iteration}",
            current=state.iteration,
            total=max_iters,
            fraction=min(1.0, state.iteration / max(max_iters, 1)),
            iteration=state.iteration,
            batch_size=len(clip_inputs),
            candidate_pool=plan.candidate_pool_size,
            remaining_deficit_total=total_deficit(required),
        )
        batch_result = analyze(
            clip_inputs,
            config,
            config.dataset_builder.work_dir,
            reuse_existing=True,
            merge_into_catalog=True,
            progress=progress,
            cancellation=cancellation,
        )

        newly_accepted = [item for item in batch_result.accepted if item.clip_id not in state.accepted_clip_ids]
        newly_rejected = [item for item in batch_result.rejected if item.clip_id not in state.rejected_clip_ids]

        for item in newly_accepted:
            if item.clip_id not in state.analyzed_clip_ids:
                state.analyzed_clip_ids.append(item.clip_id)
            if item.clip_id not in state.accepted_clip_ids:
                state.accepted_clip_ids.append(item.clip_id)
            rescue_targets = coverage.rare_rescue.target_features or list(required.keys())
            record = records_by_id.get(item.clip_id)
            if record is not None and any(feature in rescue_targets for feature in record.feature_key_set()):
                if item.clip_id not in state.rare_rescue_clip_ids:
                    state.rare_rescue_clip_ids.append(item.clip_id)
        for item in newly_rejected:
            if item.clip_id not in state.analyzed_clip_ids:
                state.analyzed_clip_ids.append(item.clip_id)
            if item.clip_id not in state.rejected_clip_ids:
                state.rejected_clip_ids.append(item.clip_id)
        for item in batch_result.errors:
            if item.clip_id not in state.analyzed_clip_ids:
                state.analyzed_clip_ids.append(item.clip_id)

        _update_pass_stats(state, batch_result, records_by_id)

        accepted_counts_after = _accepted_counts_for_state(
            accepted_metadata=accepted_metadata,
            catalog_clips=catalog_clips if catalog_clips.is_file() else None,
            index_records=index_records,
            state=state,
        )
        after_satisfied = {
            feature for feature, target in targets.items() if accepted_counts_after.get(feature, 0) >= target
        }
        newly_satisfied = len(after_satisfied - before_satisfied)
        deficits_after = compute_deficits(targets, accepted_counts_after)
        required_after = remaining_required_deficits(coverage, deficits_after)
        state.current_coverage = dict(accepted_counts_after)
        state.remaining_deficits = dict(required_after)
        state.updated_at = _utc_now()
        save_run_state(output_dir, state)

        append_jsonl(
            output_dir / "iteration-history.jsonl",
            {
                "iteration": state.iteration,
                "selected": len(plan.selected),
                "analyzed": len(batch_result.accepted) + len(batch_result.rejected) + len(batch_result.errors),
                "accepted": len(newly_accepted),
                "rejected": len(newly_rejected),
                "errors": len(batch_result.errors),
                "batch_pass_rate": (
                    len(newly_accepted) / max(len(newly_accepted) + len(newly_rejected), 1)
                ),
                "newly_satisfied_features": newly_satisfied,
                "remaining_deficit_total": total_deficit(required_after),
                "top_score_reasons": [b.model_dump(mode="json") for b in plan.score_breakdowns[:5]],
            },
        )
        if newly_rejected:
            append_jsonl(
                output_dir / "rejected-reasons.jsonl",
                {
                    "iteration": state.iteration,
                    "items": [{"clip_id": r.clip_id, "reason": r.reason} for r in newly_rejected],
                },
            )

        logger.info(
            "coverage iteration=%s complete analyzed=%s accepted=%s rejected=%s errors=%s "
            "newly_satisfied_features=%s remaining_deficit_total=%s",
            state.iteration,
            len(newly_accepted) + len(newly_rejected) + len(batch_result.errors),
            len(newly_accepted),
            len(newly_rejected),
            len(batch_result.errors),
            newly_satisfied,
            total_deficit(required_after),
        )

        write_coverage_reports(
            output_dir,
            state=state,
            plan=plan,
            config=coverage,
            accepted_counts=accepted_counts_after,
        )

        if total_deficit(required_after) <= 0:
            state.status = StopReason.COMPLETE
            state.updated_at = _utc_now()
            save_run_state(output_dir, state)
            break

        if state.iteration >= coverage.limits.max_iterations:
            state.status = StopReason.ITERATION_LIMIT_REACHED
            state.updated_at = _utc_now()
            save_run_state(output_dir, state)
            break
        if len(state.analyzed_clip_ids) >= coverage.limits.max_analyzed_clips:
            state.status = StopReason.ANALYSIS_BUDGET_EXCEEDED
            state.updated_at = _utc_now()
            save_run_state(output_dir, state)
            break

    write_coverage_reports(output_dir, state=state, plan=None, config=coverage)
    return state
