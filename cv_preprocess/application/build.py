from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from cv_preprocess.application.analyze import analyze_project
from cv_preprocess.application.audit import audit_dataset
from cv_preprocess.application.common import (
    AnalyzeResult,
    AuditReport,
    CancellationToken,
    MaterializeResult,
    ProgressEvent,
    ProgressSink,
    ScanResult,
    SelectionPlan,
    SplitPlan,
)
from cv_preprocess.application.materialize import materialize_dataset
from cv_preprocess.application.scan import scan_project
from cv_preprocess.application.select import load_selection_plan, select_dataset
from cv_preprocess.application.split import load_split_plan, plan_dataset_split
from cv_preprocess.catalog.reader import load_catalog
from cv_preprocess.config import PipelineConfig
from cv_preprocess.reports.serializer import write_json_atomic


def config_hash(config: PipelineConfig) -> str:
    payload = config.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_selection_plan(catalog, plan_path: Path) -> SelectionPlan:
    return load_selection_plan(catalog, plan_path)


def _stage_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "catalog": work_dir / "catalog" / "clips.parquet",
        "split_plan": work_dir / "plans" / "split_plan.json",
        "selection_plan": work_dir / "plans" / "selection_plan.parquet",
    }


def build_dataset(
    config: PipelineConfig,
    *,
    progress: ProgressSink | None = None,
    cancellation: CancellationToken | None = None,
    force: bool = False,
) -> tuple[ScanResult, AnalyzeResult, SplitPlan, SelectionPlan, MaterializeResult, AuditReport]:
    if not config.dataset_builder.enabled:
        raise ValueError("dataset_builder.enabled must be true to run build_dataset")

    work_dir = Path(config.dataset_builder.work_dir)
    paths = _stage_paths(work_dir)
    stage_timings: dict[str, float] = {}
    stage_skipped: dict[str, bool] = {}
    counts: dict[str, Any] = {}

    def _emit(stage: str, message: str, **metadata: Any) -> None:
        if progress is not None:
            progress(ProgressEvent(stage=stage, message=message, metadata=metadata))

    def _check_cancel() -> None:
        if cancellation is not None:
            cancellation.raise_if_cancelled()

    started = time.perf_counter()
    _check_cancel()
    _emit("build", "scan")
    scan_started = time.perf_counter()
    scan_result = scan_project(config, progress=progress)
    stage_timings["scan"] = time.perf_counter() - scan_started
    stage_skipped["scan"] = False

    _check_cancel()
    catalog = load_catalog(work_dir)
    if not force and paths["catalog"].is_file():
        _emit("build", "analyze skipped (catalog present)")
        analyze_result = AnalyzeResult(
            catalog=catalog,
            eligible_count=0,
            hard_rejected_count=0,
            warnings=["skipped: existing catalog"],
        )
        stage_timings["analyze"] = 0.0
        stage_skipped["analyze"] = True
    else:
        _emit("build", "analyze")
        analyze_started = time.perf_counter()
        analyze_result = analyze_project(config, progress=progress, cancellation=cancellation)
        stage_timings["analyze"] = time.perf_counter() - analyze_started
        stage_skipped["analyze"] = False
        catalog = analyze_result.catalog

    counts["eligible_count"] = analyze_result.eligible_count
    counts["hard_rejected_count"] = analyze_result.hard_rejected_count

    _check_cancel()
    if not force and paths["split_plan"].is_file():
        _emit("build", "plan-split skipped (split plan present)")
        split_plan = load_split_plan(catalog, paths["split_plan"])
        stage_timings["plan_split"] = 0.0
        stage_skipped["plan_split"] = True
    else:
        _emit("build", "plan-split")
        split_started = time.perf_counter()
        split_plan = plan_dataset_split(config, catalog, progress=progress)
        stage_timings["plan_split"] = time.perf_counter() - split_started
        stage_skipped["plan_split"] = False

    _check_cancel()
    if not force and paths["selection_plan"].is_file():
        _emit("build", "select skipped (selection plan present)")
        selection_plan = _load_selection_plan(catalog, paths["selection_plan"])
        stage_timings["select"] = 0.0
        stage_skipped["select"] = True
    else:
        _emit("build", "select")
        select_started = time.perf_counter()
        selection_plan = select_dataset(config, catalog, split_plan, progress=progress)
        stage_timings["select"] = time.perf_counter() - select_started
        stage_skipped["select"] = False

    counts["selected_count"] = len(selection_plan.selected_clip_ids)
    counts["reserve_count"] = len(selection_plan.reserve_clip_ids)

    _check_cancel()
    _emit("build", "materialize")
    materialize_started = time.perf_counter()
    materialize_result = materialize_dataset(
        config,
        catalog,
        selection_plan,
        progress=progress,
    )
    stage_timings["materialize"] = time.perf_counter() - materialize_started
    stage_skipped["materialize"] = False

    _check_cancel()
    _emit("build", "audit")
    audit_started = time.perf_counter()
    audit_report = audit_dataset(config, catalog, selection_plan)
    stage_timings["audit"] = time.perf_counter() - audit_started
    stage_skipped["audit"] = False

    stage_timings["total"] = time.perf_counter() - started
    manifest = {
        "schema_version": config.schema_version,
        "config_hash": config_hash(config),
        "backend": config.compute.backend,
        "stage_timings_sec": stage_timings,
        "stage_skipped": stage_skipped,
        "counts": counts,
        "cache": {
            "hits": None,
            "misses": None,
        },
        "materialize_output_root": materialize_result.output_root,
        "audit_passed": audit_report.passed,
    }
    manifest_path = work_dir / "run_manifest.json"
    write_json_atomic(manifest_path, manifest)

    return (
        scan_result,
        analyze_result,
        split_plan,
        selection_plan,
        materialize_result,
        audit_report,
    )
