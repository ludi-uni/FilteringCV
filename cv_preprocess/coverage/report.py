"""Coverage automation reports (JSON / CSV / Markdown / HTML)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cv_preprocess.config.coverage import CoverageAutomationConfig
from cv_preprocess.coverage.deficits import compute_deficits
from cv_preprocess.coverage.models import CoverageRunState, FeatureCoverageStatus, StopReason
from cv_preprocess.coverage.planner import CoveragePlan, plan_coverage
from cv_preprocess.reports.serializer import write_json_atomic


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_coverage_reports(
    run_dir: Path,
    *,
    state: CoverageRunState,
    config: CoverageAutomationConfig,
    plan: CoveragePlan | None = None,
    accepted_counts: dict[str, int] | None = None,
    index_records: list | None = None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    targets = config.iter_active_targets()
    counts = accepted_counts if accepted_counts is not None else dict(state.current_coverage)
    deficits = compute_deficits(targets, counts)

    if plan is None and index_records is not None:
        plan = plan_coverage(
            config=config,
            index_records=index_records,
            accepted_counts=counts,
            analyzed_clip_ids=set(state.analyzed_clip_ids),
            speaker_stats=state.speaker_pass_stats,
            global_attempts=state.global_pass_attempts,
            global_passes=state.global_pass_passes,
        )

    feature_rows: list[dict[str, Any]] = []
    unreachable_rows: list[dict[str, Any]] = []
    if plan is not None:
        for row in plan.feature_rows:
            payload = row.model_dump(mode="json")
            payload["accepted_after"] = int(counts.get(row.feature, row.accepted_after))
            payload["deficit"] = int(deficits.get(row.feature, 0))
            feature_rows.append(payload)
            if row.status in {
                FeatureCoverageStatus.UNREACHABLE,
                FeatureCoverageStatus.LIKELY_UNREACHABLE,
                FeatureCoverageStatus.CANDIDATE_EXHAUSTED,
            }:
                unreachable_rows.append(payload)
    else:
        for feature, target in targets.items():
            accepted = int(counts.get(feature, 0))
            deficit = int(deficits.get(feature, 0))
            status = (
                FeatureCoverageStatus.SATISFIED.value
                if deficit <= 0
                else FeatureCoverageStatus.DEFICIT.value
            )
            feature_rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "accepted_before": accepted,
                    "accepted_after": accepted,
                    "deficit": deficit,
                    "candidate_total": 0,
                    "candidate_remaining": 0,
                    "estimated_pass_rate": 0.0,
                    "expected_final_count": float(accepted),
                    "status": status,
                    "required": config.is_required(feature),
                }
            )

    analyzed = len(state.analyzed_clip_ids)
    accepted = len(state.accepted_clip_ids)
    rejected = len(state.rejected_clip_ids)
    pass_rate = (accepted / analyzed) if analyzed else 0.0

    summary = {
        "run_id": state.run_id,
        "status": state.status.value if isinstance(state.status, StopReason) else str(state.status),
        "started_at": state.started_at,
        "updated_at": state.updated_at,
        "iteration": state.iteration,
        "analyzed_clips": analyzed,
        "accepted_clips": accepted,
        "rejected_clips": rejected,
        "overall_pass_rate": pass_rate,
        "analyzed_audio_hours": state.analyzed_audio_sec / 3600.0,
        "remaining_deficit_total": int(sum(deficits.values())),
        "features": feature_rows,
        "speaker_pass_stats": {
            key: value.model_dump(mode="json") for key, value in state.speaker_pass_stats.items()
        },
        "rare_rescue_clip_ids": list(state.rare_rescue_clip_ids),
        "stop_detail": state.stop_detail,
    }

    paths: dict[str, Path] = {}
    summary_json = run_dir / "coverage-summary.json"
    write_json_atomic(summary_json, summary)
    paths["coverage-summary.json"] = summary_json

    summary_csv = run_dir / "coverage-summary.csv"
    _write_csv(summary_csv, feature_rows)
    paths["coverage-summary.csv"] = summary_csv

    unreachable_csv = run_dir / "unreachable-features.csv"
    _write_csv(unreachable_csv, unreachable_rows)
    paths["unreachable-features.csv"] = unreachable_csv

    rejected_csv = run_dir / "rejected-reasons.csv"
    rejected_rows: list[dict[str, Any]] = []
    rejected_jsonl = run_dir / "rejected-reasons.jsonl"
    if rejected_jsonl.is_file():
        for line in rejected_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            for item in payload.get("items", []):
                rejected_rows.append(
                    {
                        "iteration": payload.get("iteration"),
                        "clip_id": item.get("clip_id"),
                        "reason": item.get("reason"),
                    }
                )
    _write_csv(rejected_csv, rejected_rows)
    paths["rejected-reasons.csv"] = rejected_csv

    speaker_rows = []
    for speaker_id, stats in state.speaker_pass_stats.items():
        speaker_rows.append(
            {
                "speaker_id": speaker_id,
                "analyzed": stats.attempts,
                "accepted": stats.passes,
                "pass_rate": (stats.passes / stats.attempts) if stats.attempts else 0.0,
            }
        )
    _write_csv(run_dir / "speaker-stats.csv", speaker_rows)

    md_path = run_dir / "report.md"
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    paths["report.md"] = md_path

    html_path = run_dir / "report.html"
    html_path.write_text(_render_html(summary), encoding="utf-8")
    paths["report.html"] = html_path

    return paths


def generate_report_from_run_dir(run_dir: Path, config: CoverageAutomationConfig | None = None) -> dict[str, Path]:
    from cv_preprocess.coverage.state import load_run_state

    state = load_run_state(run_dir)
    cfg = config or CoverageAutomationConfig(enabled=True)
    return write_coverage_reports(run_dir, state=state, config=cfg)


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Coverage automation report",
        "",
        f"- status: `{summary['status']}`",
        f"- run_id: `{summary['run_id']}`",
        f"- iterations: {summary['iteration']}",
        f"- analyzed: {summary['analyzed_clips']}",
        f"- accepted: {summary['accepted_clips']}",
        f"- rejected: {summary['rejected_clips']}",
        f"- pass rate: {summary['overall_pass_rate']:.3f}",
        f"- audio hours: {summary['analyzed_audio_hours']:.3f}",
        f"- remaining deficit total: {summary['remaining_deficit_total']}",
        "",
        "## Features",
        "",
        "| feature | target | accepted | deficit | status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary.get("features", []):
        lines.append(
            f"| {row['feature']} | {row['target']} | {row.get('accepted_after', 0)} | "
            f"{row.get('deficit', 0)} | {row.get('status')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_html(summary: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{row['feature']}</td>"
        f"<td>{row['target']}</td>"
        f"<td>{row.get('accepted_after', 0)}</td>"
        f"<td>{row.get('deficit', 0)}</td>"
        f"<td>{row.get('status')}</td>"
        "</tr>"
        for row in summary.get("features", [])
    )
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8"/>
  <title>Coverage report {summary.get('run_id', '')}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>Coverage automation report</h1>
  <ul>
    <li>status: <code>{summary.get('status')}</code></li>
    <li>iterations: {summary.get('iteration')}</li>
    <li>analyzed: {summary.get('analyzed_clips')}</li>
    <li>accepted: {summary.get('accepted_clips')}</li>
    <li>rejected: {summary.get('rejected_clips')}</li>
    <li>pass rate: {summary.get('overall_pass_rate', 0):.3f}</li>
  </ul>
  <table>
    <thead><tr><th>feature</th><th>target</th><th>accepted</th><th>deficit</th><th>status</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
