from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from cv_preprocess.application.common import AuditReport, SelectionPlan
from cv_preprocess.catalog import CatalogRef
from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.catalog.reader import read_clips
from cv_preprocess.config import PipelineConfig
from cv_preprocess.reports.serializer import write_json_atomic
from cv_preprocess.split.leakage import ClipSplitRecord, detect_leakage


def _load_selected_ids(selection_plan: SelectionPlan) -> list[str]:
    if selection_plan.selected_clip_ids:
        return list(selection_plan.selected_clip_ids)
    plan_path = selection_plan.plan_path
    if plan_path is None or not Path(plan_path).is_file():
        return []
    plan_df = pl.read_parquet(plan_path)
    return plan_df.filter(pl.col("disposition") == ClipDisposition.SELECTED.value)["clip_id"].to_list()


def _speaker_constraint_issues(
    selected_rows: list[dict[str, Any]],
    config: PipelineConfig,
) -> list[str]:
    issues: list[str] = []
    constraints = config.dataset_builder.speaker_constraints
    clip_counts: Counter[str] = Counter()
    duration_sec: Counter[str] = Counter()
    for row in selected_rows:
        speaker = str(row.get("speaker_id") or "")
        clip_counts[speaker] += 1
        duration_sec[speaker] += float(row.get("duration_sec") or 0.0)

    max_clips = constraints.max_clips_per_speaker
    if max_clips is not None:
        for speaker, count in sorted(clip_counts.items()):
            if count > max_clips:
                issues.append(
                    f"speaker {speaker!r} has {count} selected clips (max {max_clips})"
                )

    max_duration = constraints.max_duration_sec_per_speaker
    if max_duration is not None:
        for speaker, duration in sorted(duration_sec.items()):
            if duration > max_duration + 1e-6:
                issues.append(
                    f"speaker {speaker!r} has {duration:.2f}s selected "
                    f"(max {max_duration:.2f}s)"
                )

    min_duration = constraints.min_duration_minutes
    if min_duration is not None:
        min_sec = min_duration * 60.0
        for speaker, duration in sorted(duration_sec.items()):
            if duration + 1e-6 < min_sec:
                issues.append(
                    f"speaker {speaker!r} has {duration:.2f}s selected "
                    f"(min {min_sec:.2f}s)"
                )

    return issues


def _leakage_issues(
    selected_rows: list[dict[str, Any]],
    split_by_id: dict[str, str | None],
    config: PipelineConfig,
) -> list[str]:
    if not split_by_id:
        return []

    by_split: dict[str, list[ClipSplitRecord]] = {}
    for row in selected_rows:
        clip_id = str(row["clip_id"])
        split_name = split_by_id.get(clip_id) or row.get("split") or "train"
        by_split.setdefault(str(split_name), []).append(
            ClipSplitRecord(
                clip_id=clip_id,
                speaker_id=str(row.get("speaker_id") or ""),
                audio_hash=str(row.get("audio_sha256") or ""),
                sentence_id=str(row.get("sentence_id") or ""),
                normalized_text=str(row.get("text_norm") or ""),
                duration_sec=float(row.get("duration_sec") or 0.0),
            )
        )

    violations = detect_leakage(by_split, config.dataset_builder.split.leakage_policy)
    return [
        (
            f"leakage {violation.dimension}={violation.key!r} across splits "
            f"{violation.splits}"
        )
        for violation in violations
    ]


def audit_dataset(
    config: PipelineConfig,
    catalog: CatalogRef,
    selection_plan: SelectionPlan,
) -> AuditReport:
    issues: list[str] = []
    clips_df = read_clips(catalog.resolved_clips_path())
    selected_ids = _load_selected_ids(selection_plan)
    selected_df = clips_df.filter(pl.col("clip_id").is_in(selected_ids))
    selected_rows = list(selected_df.iter_rows(named=True))

    hard_rejected_selected = selected_df.filter(
        pl.col("disposition") == ClipDisposition.HARD_REJECTED.value
    )
    if hard_rejected_selected.height > 0:
        rejected_ids = hard_rejected_selected["clip_id"].to_list()
        issues.append(
            f"{len(rejected_ids)} HARD_REJECTED clip(s) in selection: {rejected_ids[:5]}"
        )

    issues.extend(_speaker_constraint_issues(selected_rows, config))

    split_by_id: dict[str, str | None] = {}
    plan_path = selection_plan.plan_path
    if plan_path is not None and Path(plan_path).is_file():
        plan_df = pl.read_parquet(plan_path)
        for row in plan_df.filter(pl.col("disposition") == ClipDisposition.SELECTED.value).iter_rows(
            named=True
        ):
            split_by_id[str(row["clip_id"])] = row.get("split")

    issues.extend(_leakage_issues(selected_rows, split_by_id, config))

    report_payload = {
        "passed": len(issues) == 0,
        "issues": issues,
        "selected_count": len(selected_ids),
        "catalog_work_dir": str(catalog.work_dir),
    }
    reports_dir = Path(catalog.work_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "audit_report.json"
    write_json_atomic(report_path, report_payload)

    return AuditReport(
        catalog=catalog,
        passed=len(issues) == 0,
        issues=issues,
    )
