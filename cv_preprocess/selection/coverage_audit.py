"""Final coverage audit and report writers for coverage-aware select."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from cv_preprocess.reports.serializer import write_json_atomic
from cv_preprocess.selection.coverage_keys import parse_feature_key
from cv_preprocess.selection.coverage_reservation import ReservationRecord
from cv_preprocess.selection.coverage_targets import (
    AuditStatus,
    SelectionCoverageConstraints,
)
from cv_preprocess.selection.protocol import ClipFeatures


@dataclass
class FeatureAuditRow:
    feature_family: str
    feature: str
    configured_minimum: int
    configured_desired: int
    index_candidate_count: int
    eligible_candidate_count: int
    eligible_speaker_count: int
    effective_minimum: int
    effective_desired: int
    selected_count: int
    selected_speaker_count: int
    status: str
    failure_reason: str = ""
    required: bool = True


@dataclass
class CoverageAuditReport:
    rows: list[FeatureAuditRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    should_fail: bool = False
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissingFeatureRequest:
    feature_family: str
    feature: str
    required_count: int
    selected_count: int
    reason: str
    configured_target: int | None = None
    index_candidate_count: int = 0
    eligible_candidate_count: int = 0
    status: str = ""


def count_selected_coverage(
    selected_ids: Sequence[str],
    constraints: SelectionCoverageConstraints,
    clips_by_id: Mapping[str, ClipFeatures],
) -> tuple[dict[str, int], dict[str, set[str]]]:
    counts: dict[str, int] = {key: 0 for key in constraints.targets}
    speakers: dict[str, set[str]] = {key: set() for key in constraints.targets}
    for clip_id in selected_ids:
        keys = constraints.clip_coverage_keys.get(clip_id)
        if keys is None:
            clip = clips_by_id.get(clip_id)
            keys = set(clip.coverage_keys) if clip is not None else set()
        clip = clips_by_id.get(clip_id)
        for key in keys:
            if key not in counts:
                continue
            counts[key] += 1
            if clip is not None:
                speakers[key].add(clip.speaker_id)
    return counts, speakers


def classify_feature_status(
    *,
    configured_minimum: int,
    configured_desired: int,
    effective_minimum: int,
    effective_desired: int,
    selected_count: int,
    index_candidate_count: int | None,
    eligible_candidate_count: int,
    conflict: bool,
) -> tuple[AuditStatus, str]:
    if index_candidate_count is not None and index_candidate_count <= 0 and eligible_candidate_count <= 0:
        return "not_present_in_index", "feature absent from coverage index"
    if eligible_candidate_count <= 0:
        if index_candidate_count is not None and index_candidate_count > 0:
            return "not_present_in_eligible", "present in index but not in eligible catalog"
        return "candidate_missing", "no eligible candidates"

    if selected_count >= configured_desired and configured_desired > 0:
        return "configured_target_satisfied", ""
    if selected_count >= configured_minimum and configured_minimum > 0 and selected_count >= effective_desired:
        return "desired_satisfied", ""
    if selected_count >= effective_desired and effective_desired > 0:
        if effective_desired < configured_desired:
            return "corpus_limit_satisfied", "limited by eligible corpus size"
        return "satisfied", ""
    if selected_count >= effective_minimum and effective_minimum > 0:
        if selected_count < effective_desired:
            return "desired_unsatisfied", "minimum met but desired not reached"
        return "minimum_satisfied", ""
    if selected_count >= configured_minimum and configured_minimum > 0:
        return "minimum_satisfied", ""

    # Below effective minimum
    if conflict or (
        eligible_candidate_count > 0 and selected_count < effective_minimum
    ):
        if eligible_candidate_count > selected_count:
            return (
                "selection_constraint_conflict",
                "candidates exist but speaker/time/duplicate constraints blocked selection",
            )
        return "minimum_unsatisfied", "effective minimum not reached"
    return "minimum_unsatisfied", "effective minimum not reached"


def audit_coverage(
    constraints: SelectionCoverageConstraints,
    selected_ids: Sequence[str],
    clips_by_id: Mapping[str, ClipFeatures],
    *,
    conflict_features: set[str] | None = None,
) -> CoverageAuditReport:
    conflicts = conflict_features or set()
    counts, speakers = count_selected_coverage(selected_ids, constraints, clips_by_id)
    rows: list[FeatureAuditRow] = []
    warnings: list[str] = []
    hard_failures = 0
    algorithm_failures = 0

    for key, target in sorted(constraints.targets.items()):
        family, token = parse_feature_key(key)
        index_count = (
            int(target.index_candidate_count)
            if target.index_candidate_count is not None
            else 0
        )
        selected = int(counts.get(key, 0))
        status, reason = classify_feature_status(
            configured_minimum=target.configured_minimum,
            configured_desired=target.configured_desired,
            effective_minimum=target.effective_minimum,
            effective_desired=target.effective_desired,
            selected_count=selected,
            index_candidate_count=target.index_candidate_count,
            eligible_candidate_count=target.eligible_clip_count,
            conflict=key in conflicts,
        )
        rows.append(
            FeatureAuditRow(
                feature_family=family,
                feature=token,
                configured_minimum=target.configured_minimum,
                configured_desired=target.configured_desired,
                index_candidate_count=index_count,
                eligible_candidate_count=target.eligible_clip_count,
                eligible_speaker_count=target.eligible_unique_speaker_count,
                effective_minimum=target.effective_minimum,
                effective_desired=target.effective_desired,
                selected_count=selected,
                selected_speaker_count=len(speakers.get(key, set())),
                status=status,
                failure_reason=reason,
                required=target.required,
            )
        )

        if target.required and status in {
            "minimum_unsatisfied",
            "selection_constraint_conflict",
            "not_present_in_index",
            "not_present_in_eligible",
            "candidate_missing",
        }:
            # Candidates existed to meet effective min but select fell short → algorithm failure
            if (
                status == "minimum_unsatisfied"
                and target.eligible_clip_count >= target.effective_minimum
                and selected < target.effective_minimum
                and key not in conflicts
            ):
                algorithm_failures += 1
                warnings.append(
                    f"algorithm shortfall for {key}: selected={selected} "
                    f"effective_minimum={target.effective_minimum} "
                    f"(eligible={target.eligible_clip_count})"
                )
            hard_failures += 1

    policy = constraints.violation_policy
    should_fail = False
    if hard_failures and policy == "fail":
        # fail only when effective minimum was achievable but missed, or explicit fail policy
        should_fail = True
    if algorithm_failures and policy == "best_effort":
        warnings.append(
            "coverage best_effort: some effective minima were achievable but not selected"
        )

    summary = {
        "feature_count": len(rows),
        "required_unsatisfied": hard_failures,
        "algorithm_shortfalls": algorithm_failures,
        "violation_policy": policy,
        "selected_clip_count": len(selected_ids),
    }
    return CoverageAuditReport(
        rows=rows,
        warnings=warnings,
        should_fail=should_fail and policy == "fail",
        summary=summary,
    )


def build_missing_feature_requests(report: CoverageAuditReport) -> list[MissingFeatureRequest]:
    missing: list[MissingFeatureRequest] = []
    for row in report.rows:
        if row.status not in {
            "not_present_in_index",
            "not_present_in_eligible",
            "candidate_missing",
            "minimum_unsatisfied",
            "selection_constraint_conflict",
        }:
            continue
        if row.selected_count >= row.effective_minimum and row.effective_minimum > 0:
            continue
        if row.status in {"not_present_in_index", "not_present_in_eligible", "candidate_missing"} or (
            row.required and row.selected_count < row.effective_minimum
        ):
            missing.append(
                MissingFeatureRequest(
                    feature_family=row.feature_family,
                    feature=row.feature,
                    required_count=row.effective_minimum or row.configured_minimum,
                    selected_count=row.selected_count,
                    reason=row.failure_reason or row.status,
                    configured_target=row.configured_minimum,
                    index_candidate_count=row.index_candidate_count,
                    eligible_candidate_count=row.eligible_candidate_count,
                    status=row.status,
                )
            )
    return missing


def write_coverage_audit_reports(
    output_dir: Path,
    report: CoverageAuditReport,
    *,
    contributions: Sequence[dict[str, Any]] | Sequence[ReservationRecord] = (),
    acoustic_summary: Mapping[str, Any] | None = None,
    missing: Sequence[MissingFeatureRequest] = (),
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    audit_json = output_dir / "coverage-audit.json"
    write_json_atomic(
        audit_json,
        {
            "summary": report.summary,
            "warnings": report.warnings,
            "features": [asdict(row) for row in report.rows],
        },
    )
    paths["coverage_audit_json"] = audit_json

    audit_csv = output_dir / "coverage-audit.csv"
    fieldnames = [
        "feature_family",
        "feature",
        "configured_minimum",
        "configured_desired",
        "index_candidate_count",
        "eligible_candidate_count",
        "eligible_speaker_count",
        "effective_minimum",
        "effective_desired",
        "selected_count",
        "selected_speaker_count",
        "status",
        "failure_reason",
    ]
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})
    paths["coverage_audit_csv"] = audit_csv

    contrib_path = output_dir / "coverage-contributions.jsonl"
    with contrib_path.open("w", encoding="utf-8") as handle:
        for item in contributions:
            if isinstance(item, ReservationRecord):
                payload = {
                    "clip_id": item.clip_id,
                    "selection_phase": item.selection_phase,
                    "coverage_contributions": [asdict(c) for c in item.coverage_contributions],
                    "quality_score": item.quality_score,
                    "duration_sec": item.duration_sec,
                    "speaker_id": item.speaker_id,
                }
            else:
                payload = dict(item)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    paths["coverage_contributions"] = contrib_path

    if acoustic_summary is not None:
        acoustic_path = output_dir / "acoustic-diversity-summary.json"
        write_json_atomic(acoustic_path, dict(acoustic_summary))
        paths["acoustic_diversity_summary"] = acoustic_path

    missing_path = output_dir / "missing-features.json"
    write_json_atomic(
        missing_path,
        {"missing_features": [asdict(m) for m in missing]},
    )
    paths["missing_features"] = missing_path
    return paths
