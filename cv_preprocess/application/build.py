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
from cv_preprocess.compute.profiling import resource_snapshot
from cv_preprocess.coverage.indexer import build_clip_index
from cv_preprocess.coverage.paths import accepted_catalog_path, resolve_coverage_paths
from cv_preprocess.coverage.report import generate_report_from_run_dir
from cv_preprocess.coverage.runner import run_coverage
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


def _catalog_manifest(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "catalog" / "manifest.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _should_skip_full_analyze(
    *,
    force: bool,
    catalog_exists: bool,
    coverage_prepass: bool,
    work_dir: Path,
) -> bool:
    """Skip analyze only when a complete catalog already exists and coverage did not just pre-fill."""
    if force or not catalog_exists:
        return False
    if coverage_prepass:
        # Coverage may have written a partial catalog; always finish with reuse-aware analyze.
        return False
    manifest = _catalog_manifest(work_dir)
    if manifest.get("partial_analyze"):
        return False
    return True


def _run_coverage_prepass(
    config: PipelineConfig,
    *,
    force: bool,
    progress: ProgressSink | None,
    cancellation: CancellationToken | None,
) -> dict[str, Any]:
    """Lightweight index + selective analyze before full corpus analyze."""
    if cancellation is not None:
        cancellation.raise_if_cancelled()
    paths = resolve_coverage_paths(config)
    if progress is not None:
        progress(
            ProgressEvent(
                stage="build",
                message="coverage-index",
                metadata={"phase": "coverage", "output": str(paths.index_path)},
            )
        )
    index_result = build_clip_index(
        config,
        output=paths.index_path,
        force=force,
        incremental=not force,
        progress=progress,
        cancellation=cancellation,
    )
    if cancellation is not None:
        cancellation.raise_if_cancelled()
    catalog = accepted_catalog_path(config)
    resume = (
        paths.run_dir.is_dir()
        and (paths.run_dir / "run-state.json").is_file()
        and not force
    )
    if force and (paths.run_dir / "run-state.json").is_file():
        (paths.run_dir / "run-state.json").unlink()
    if progress is not None:
        progress(
            ProgressEvent(
                stage="build",
                message="coverage-run",
                metadata={"phase": "coverage", "resume": resume},
            )
        )
    state = run_coverage(
        config,
        index_path=paths.index_path,
        output_dir=paths.run_dir,
        accepted_metadata=catalog if catalog.is_file() else None,
        resume=resume,
        dry_run=False,
        progress=progress,
        cancellation=cancellation,
    )
    if cancellation is not None:
        cancellation.raise_if_cancelled()
    report_paths = generate_report_from_run_dir(paths.run_dir, config.coverage)
    return {
        "index_path": str(index_result.index_path),
        "clip_count": index_result.clip_count,
        "run_dir": str(paths.run_dir),
        "status": state.status.value,
        "analyzed": len(state.analyzed_clip_ids),
        "accepted": len(state.accepted_clip_ids),
        "report_files": {k: str(v) for k, v in report_paths.items()},
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
    coverage_summary: dict[str, Any] | None = None

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

    coverage_prepass = bool(config.coverage.enabled and config.coverage.insert_before_analyze)
    if coverage_prepass:
        _check_cancel()
        _emit("build", "coverage-prepass (before analyze)")
        coverage_started = time.perf_counter()
        coverage_summary = _run_coverage_prepass(
            config,
            force=force,
            progress=progress,
            cancellation=cancellation,
        )
        stage_timings["coverage"] = time.perf_counter() - coverage_started
        stage_skipped["coverage"] = False
        counts["coverage"] = {
            "status": coverage_summary.get("status"),
            "analyzed": coverage_summary.get("analyzed"),
            "accepted": coverage_summary.get("accepted"),
            "index_clips": coverage_summary.get("clip_count"),
        }
    else:
        stage_timings["coverage"] = 0.0
        stage_skipped["coverage"] = True

    _check_cancel()
    catalog = load_catalog(work_dir)
    catalog_exists = paths["catalog"].is_file()
    if _should_skip_full_analyze(
        force=force,
        catalog_exists=catalog_exists,
        coverage_prepass=coverage_prepass,
        work_dir=work_dir,
    ):
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
        _emit(
            "build",
            "analyze"
            + (" (reuse coverage results)" if coverage_prepass else ""),
        )
        analyze_started = time.perf_counter()
        analyze_result = analyze_project(
            config,
            progress=progress,
            cancellation=cancellation,
            reuse_existing=not force,
        )
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
    resources = resource_snapshot()
    manifest = {
        "schema_version": config.schema_version,
        "config_hash": config_hash(config),
        "backend": config.compute.backend,
        "stage_timings_sec": stage_timings,
        "stage_skipped": stage_skipped,
        "counts": counts,
        "resources": resources.as_dict(),
        "cache": {
            "hits": None,
            "misses": None,
        },
        "materialize_output_root": materialize_result.output_root,
        "audit_passed": audit_report.passed,
        "coverage_prepass": coverage_summary,
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
