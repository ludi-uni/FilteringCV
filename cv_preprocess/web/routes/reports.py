from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import APIRouter, HTTPException, Request

from cv_preprocess.catalog.reader import read_clips
from cv_preprocess.coverage.paths import resolve_coverage_paths
from cv_preprocess.reports.coverage import compute_coverage_summary
from cv_preprocess.reports.rejection import compute_rejection_summary
from cv_preprocess.web.dependencies import get_app_state

router = APIRouter()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


@router.get("/coverage")
def coverage_report(request: Request) -> dict[str, Any]:
    state = get_app_state(request)
    clips_path = state.catalog_dir / "clips.parquet"
    feature_counts_path = state.catalog_dir / "feature_counts.parquet"
    if not clips_path.is_file():
        raise HTTPException(status_code=404, detail="catalog not found")
    clips = read_clips(clips_path)
    feature_counts = (
        pl.read_parquet(feature_counts_path)
        if feature_counts_path.is_file()
        else pl.DataFrame()
    )
    report = compute_coverage_summary(clips, feature_counts)
    return report.model_dump(mode="json")


@router.get("/coverage-automation")
def coverage_automation_report(request: Request) -> dict[str, Any]:
    """Summarize the GUI active coverage-automation run (if any)."""
    state = get_app_state(request)
    paths = resolve_coverage_paths(state.config, base_dir=state.project_root)
    summary = _read_json(paths.run_dir / "coverage-summary.json")
    index_ready = paths.index_path.is_file()
    run_ready = (paths.run_dir / "run-state.json").is_file()
    message = None
    if not state.config.coverage.enabled:
        message = "coverage.enabled is false"
    elif not run_ready:
        message = "No active coverage run yet — start coverage-build from Jobs"
    return {
        "available": True,
        "enabled": bool(state.config.coverage.enabled),
        "output_dir": str(paths.output_dir),
        "run_dir": str(paths.run_dir),
        "index_ready": index_ready,
        "run_ready": run_ready,
        "summary": summary,
        "message": message,
    }


@router.get("/rejection")
def rejection_report(request: Request) -> dict[str, Any]:
    state = get_app_state(request)
    clips_path = state.catalog_dir / "clips.parquet"
    if not clips_path.is_file():
        raise HTTPException(status_code=404, detail="catalog not found")
    clips = read_clips(clips_path)
    report = compute_rejection_summary(clips)
    return report.model_dump(mode="json")


@router.get("/run-manifest")
def run_manifest(request: Request) -> dict[str, Any]:
    state = get_app_state(request)
    manifest_path = state.work_dir / "run_manifest.json"
    payload = _read_json(manifest_path)
    if payload is None:
        raise HTTPException(status_code=404, detail="run manifest not found")
    return payload
