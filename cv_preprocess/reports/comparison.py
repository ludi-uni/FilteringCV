from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.reports.coverage import compute_coverage_summary, js_distance


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _load_run_manifest(root: Path) -> dict[str, Any] | None:
    for candidate in (root / "run_manifest.json", root / "catalog" / "manifest.json"):
        payload = _read_json(candidate)
        if payload is not None:
            return payload
    return None


def _load_selected_clip_ids(root: Path) -> set[str]:
    selection_plan = root / "plans" / "selection_plan.parquet"
    if selection_plan.is_file():
        plan_df = pl.read_parquet(selection_plan)
        return set(
            plan_df.filter(pl.col("disposition") == ClipDisposition.SELECTED.value)["clip_id"].to_list()
        )

    metadata_path = root / "metadata.jsonl"
    if metadata_path.is_file():
        clip_ids: set[str] = set()
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            clip_id = row.get("clip_id") or row.get("utt_id")
            if clip_id:
                clip_ids.add(str(clip_id))
        return clip_ids
    return set()


def _coverage_js_by_feature(root: Path) -> dict[str, float]:
    catalog_dir = root / "catalog"
    clips_path = catalog_dir / "clips.parquet"
    feature_counts_path = catalog_dir / "feature_counts.parquet"
    if not clips_path.is_file() or not feature_counts_path.is_file():
        return {}
    clips = pl.read_parquet(clips_path)
    feature_counts = pl.read_parquet(feature_counts_path)
    report = compute_coverage_summary(clips, feature_counts)
    return dict(report.js_distance_to_uniform)


def compare_runs(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    """Compare two work directories or materialized output directories."""
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)

    left_manifest = _load_run_manifest(left_dir)
    right_manifest = _load_run_manifest(right_dir)

    left_clips = _load_selected_clip_ids(left_dir)
    right_clips = _load_selected_clip_ids(right_dir)

    left_coverage = _coverage_js_by_feature(left_dir)
    right_coverage = _coverage_js_by_feature(right_dir)
    coverage_js_delta: dict[str, float] = {}
    for feature_type in sorted(set(left_coverage) | set(right_coverage)):
        coverage_js_delta[feature_type] = js_distance(
            {feature_type: left_coverage.get(feature_type, 0.0)},
            {feature_type: right_coverage.get(feature_type, 0.0)},
        )

    config_diff: dict[str, Any] = {}
    if left_manifest and right_manifest:
        left_hash = left_manifest.get("config_hash") or left_manifest.get("pipeline_hash")
        right_hash = right_manifest.get("config_hash") or right_manifest.get("pipeline_hash")
        config_diff = {
            "left": left_hash,
            "right": right_hash,
            "same": left_hash == right_hash,
        }

    left_timings = (left_manifest or {}).get("stage_timings_sec", {})
    right_timings = (right_manifest or {}).get("stage_timings_sec", {})
    duration_delta = {
        stage: float(right_timings.get(stage, 0.0)) - float(left_timings.get(stage, 0.0))
        for stage in sorted(set(left_timings) | set(right_timings))
    }

    return {
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "config_diff": config_diff,
        "duration_delta_sec": duration_delta,
        "speaker_counts": {
            "left_selected_clips": len(left_clips),
            "right_selected_clips": len(right_clips),
            "only_left": sorted(left_clips - right_clips),
            "only_right": sorted(right_clips - left_clips),
            "intersection": sorted(left_clips & right_clips),
        },
        "coverage_js_delta": coverage_js_delta,
        "clip_set_diff": {
            "added": sorted(right_clips - left_clips),
            "removed": sorted(left_clips - right_clips),
            "unchanged_count": len(left_clips & right_clips),
        },
    }
