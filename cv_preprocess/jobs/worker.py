from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any

from cv_preprocess.application.analyze import analyze_project
from cv_preprocess.application.audit import audit_dataset
from cv_preprocess.application.build import build_dataset
from cv_preprocess.application.common import ProgressEvent
from cv_preprocess.application.materialize import materialize_dataset
from cv_preprocess.application.scan import scan_project
from cv_preprocess.application.select import load_selection_plan, select_dataset
from cv_preprocess.application.split import load_split_plan, plan_dataset_split
from cv_preprocess.catalog.reader import load_catalog
from cv_preprocess.config import load_config
from cv_preprocess.coverage.counter import load_accepted_counts
from cv_preprocess.coverage.indexer import build_clip_index, load_index_jsonl
from cv_preprocess.coverage.paths import (
    accepted_catalog_path,
    project_base_from_config_path,
    resolve_coverage_paths,
)
from cv_preprocess.coverage.planner import plan_coverage
from cv_preprocess.coverage.report import generate_report_from_run_dir
from cv_preprocess.coverage.runner import run_coverage
from cv_preprocess.jobs.models import COVERAGE_JOB_TYPES, JobStatus, JobType
from cv_preprocess.jobs.progress import FileCancellationToken, JobProgressWriter
from cv_preprocess.jobs.store import JobStore
from cv_preprocess.reports.serializer import write_json_atomic


def _serialize_result(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if isinstance(payload, tuple):
        return {"items": [_serialize_result(item) for item in payload]}
    if isinstance(payload, dict):
        return payload
    return {"value": payload}


def _run_coverage_index(
    config: Any,
    *,
    force: bool,
    progress: JobProgressWriter,
    base_dir: Path,
    cancellation: FileCancellationToken | None = None,
) -> dict[str, Any]:
    paths = resolve_coverage_paths(config, base_dir=base_dir)
    progress(
        ProgressEvent(
            stage="coverage-index",
            message="building lightweight index",
            metadata={"phase": "prepare", "output": str(paths.index_path)},
        )
    )
    result = build_clip_index(
        config,
        output=paths.index_path,
        force=force,
        incremental=not force,
        progress=progress,
        cancellation=cancellation,
    )
    progress(
        ProgressEvent(
            stage="coverage-index",
            message="index complete",
            metadata={"phase": "done", "clip_count": result.clip_count},
        )
    )
    return {
        "index_path": str(result.index_path),
        "meta_path": str(result.meta_path),
        "clip_count": result.clip_count,
        "config_hash": result.meta.config_hash,
    }


def _run_coverage_plan(
    config: Any,
    *,
    progress: JobProgressWriter,
    base_dir: Path,
) -> dict[str, Any]:
    if not config.coverage.enabled:
        raise ValueError("coverage.enabled must be true for coverage-plan")
    paths = resolve_coverage_paths(config, base_dir=base_dir)
    if not paths.index_path.is_file():
        raise ValueError(f"coverage index not found: {paths.index_path} (run coverage-index first)")
    progress(
        ProgressEvent(
            stage="coverage-plan",
            message="planning next batch",
            metadata={"phase": "plan"},
        )
    )
    records = load_index_jsonl(paths.index_path)
    catalog = accepted_catalog_path(config, base_dir=base_dir)
    accepted = load_accepted_counts(
        catalog_clips=catalog if catalog.is_file() else None,
    )
    plan = plan_coverage(config=config.coverage, index_records=records, accepted_counts=accepted)
    write_json_atomic(paths.plan_path, plan.to_dict())
    progress(
        ProgressEvent(
            stage="coverage-plan",
            message="plan written",
            metadata={"phase": "done", "batch_size": plan.batch_size},
        )
    )
    return {"plan_path": str(paths.plan_path), "plan": plan.to_dict()}


def _run_coverage_run(
    config: Any,
    *,
    force: bool,
    progress: JobProgressWriter,
    base_dir: Path,
    dry_run: bool = False,
    cancellation: FileCancellationToken | None = None,
) -> dict[str, Any]:
    if not config.coverage.enabled:
        raise ValueError("coverage.enabled must be true for coverage-run")
    paths = resolve_coverage_paths(config, base_dir=base_dir)
    if not paths.index_path.is_file():
        raise ValueError(f"coverage index not found: {paths.index_path} (run coverage-index first)")
    catalog = accepted_catalog_path(config, base_dir=base_dir)
    resume = False
    if paths.run_dir.is_dir() and (paths.run_dir / "run-state.json").is_file() and not force:
        resume = True
    elif force and paths.run_dir.is_dir():
        state_file = paths.run_dir / "run-state.json"
        if state_file.is_file():
            state_file.unlink()

    progress(
        ProgressEvent(
            stage="coverage-run",
            message="resume" if resume else ("dry-run" if dry_run else "start"),
            metadata={"phase": "prepare", "run_dir": str(paths.run_dir)},
        )
    )
    state = run_coverage(
        config,
        index_path=paths.index_path,
        output_dir=paths.run_dir,
        accepted_metadata=catalog if catalog.is_file() else None,
        resume=resume,
        dry_run=dry_run,
        progress=progress,
        cancellation=cancellation,
    )
    progress(
        ProgressEvent(
            stage="coverage-run",
            message=f"finished status={state.status.value}",
            metadata={
                "phase": "done",
                "iteration": state.iteration,
                "analyzed": len(state.analyzed_clip_ids),
                "accepted": len(state.accepted_clip_ids),
            },
        )
    )
    return {
        "run_dir": str(paths.run_dir),
        "status": state.status.value,
        "iteration": state.iteration,
        "analyzed": len(state.analyzed_clip_ids),
        "accepted": len(state.accepted_clip_ids),
        "rejected": len(state.rejected_clip_ids),
        "remaining_deficits": state.remaining_deficits,
    }


def _run_coverage_report(
    config: Any,
    *,
    progress: JobProgressWriter,
    base_dir: Path,
) -> dict[str, Any]:
    paths = resolve_coverage_paths(config, base_dir=base_dir)
    if not (paths.run_dir / "run-state.json").is_file():
        raise ValueError(f"coverage run state not found: {paths.run_dir}")
    progress(
        ProgressEvent(
            stage="coverage-report",
            message="generating reports",
            metadata={"phase": "report"},
        )
    )
    report_paths = generate_report_from_run_dir(paths.run_dir, config.coverage)
    progress(
        ProgressEvent(
            stage="coverage-report",
            message="report complete",
            metadata={"phase": "done"},
        )
    )
    return {"run_dir": str(paths.run_dir), "files": {k: str(v) for k, v in report_paths.items()}}


def _run_job(
    job_type: JobType,
    config_path: Path,
    *,
    force: bool,
    progress: JobProgressWriter,
    cancellation: FileCancellationToken,
) -> dict[str, Any]:
    config = load_config(config_path)
    base_dir = project_base_from_config_path(config_path)
    is_coverage = job_type in COVERAGE_JOB_TYPES

    if is_coverage:
        if job_type != JobType.COVERAGE_INDEX and not config.coverage.enabled:
            raise ValueError("coverage.enabled must be true for this coverage job")
    elif not config.dataset_builder.enabled and job_type != JobType.SCAN:
        raise ValueError("dataset_builder.enabled must be true for dataset builder jobs")

    work_dir = Path(config.dataset_builder.work_dir)
    if not work_dir.is_absolute():
        work_dir = (base_dir / work_dir).resolve()

    if job_type == JobType.SCAN:
        result = scan_project(config, progress=progress)
        return {"scan": _serialize_result(result)}

    if job_type == JobType.ANALYZE:
        result = analyze_project(config, progress=progress, cancellation=cancellation)
        return {"analyze": _serialize_result(result)}

    if job_type == JobType.PLAN_SPLIT:
        catalog = load_catalog(work_dir)
        if catalog.clips_path is None:
            raise ValueError(f"catalog not found under {work_dir / 'catalog'}")
        result = plan_dataset_split(config, catalog, progress=progress)
        return {"split_plan": _serialize_result(result)}

    if job_type == JobType.SELECT:
        catalog = load_catalog(work_dir)
        if catalog.clips_path is None:
            raise ValueError(f"catalog not found under {work_dir / 'catalog'}")
        split_plan_path = work_dir / "plans" / "split_plan.json"
        if split_plan_path.is_file():
            split_plan = load_split_plan(catalog, split_plan_path)
        else:
            split_plan = plan_dataset_split(config, catalog, progress=progress)
        result = select_dataset(config, catalog, split_plan, progress=progress)
        return {"selection_plan": _serialize_result(result)}

    if job_type == JobType.MATERIALIZE:
        catalog = load_catalog(work_dir)
        if catalog.clips_path is None:
            raise ValueError(f"catalog not found under {work_dir / 'catalog'}")
        plan_path = work_dir / "plans" / "selection_plan.parquet"
        if not plan_path.is_file():
            raise ValueError(f"selection plan not found: {plan_path}")
        selection_plan = load_selection_plan(catalog, plan_path)
        result = materialize_dataset(config, catalog, selection_plan, progress=progress)
        return {"materialize": _serialize_result(result)}

    if job_type == JobType.AUDIT:
        catalog = load_catalog(work_dir)
        if catalog.clips_path is None:
            raise ValueError(f"catalog not found under {work_dir / 'catalog'}")
        plan_path = work_dir / "plans" / "selection_plan.parquet"
        if not plan_path.is_file():
            raise ValueError(f"selection plan not found: {plan_path}")
        selection_plan = load_selection_plan(catalog, plan_path)
        result = audit_dataset(config, catalog, selection_plan)
        return {"audit": _serialize_result(result)}

    if job_type == JobType.BUILD:
        (
            scan_result,
            analyze_result,
            split_plan,
            selection_plan,
            materialize_result,
            audit_report,
        ) = build_dataset(
            config,
            progress=progress,
            cancellation=cancellation,
            force=force,
        )
        return {
            "scan": _serialize_result(scan_result),
            "analyze": _serialize_result(analyze_result),
            "split_plan": _serialize_result(split_plan),
            "selection_plan": _serialize_result(selection_plan),
            "materialize": _serialize_result(materialize_result),
            "audit": _serialize_result(audit_report),
        }

    if job_type == JobType.COVERAGE_INDEX:
        return {
            "coverage_index": _run_coverage_index(
                config,
                force=force,
                progress=progress,
                base_dir=base_dir,
                cancellation=cancellation,
            )
        }

    if job_type == JobType.COVERAGE_PLAN:
        return {"coverage_plan": _run_coverage_plan(config, progress=progress, base_dir=base_dir)}

    if job_type == JobType.COVERAGE_RUN:
        cancellation.raise_if_cancelled()
        return {
            "coverage_run": _run_coverage_run(
                config,
                force=force,
                progress=progress,
                base_dir=base_dir,
                dry_run=False,
                cancellation=cancellation,
            )
        }

    if job_type == JobType.COVERAGE_REPORT:
        return {
            "coverage_report": _run_coverage_report(config, progress=progress, base_dir=base_dir)
        }

    if job_type == JobType.COVERAGE_BUILD:
        cancellation.raise_if_cancelled()
        index_result = _run_coverage_index(
            config,
            force=force,
            progress=progress,
            base_dir=base_dir,
            cancellation=cancellation,
        )
        cancellation.raise_if_cancelled()
        run_result = _run_coverage_run(
            config,
            force=force,
            progress=progress,
            base_dir=base_dir,
            dry_run=False,
            cancellation=cancellation,
        )
        cancellation.raise_if_cancelled()
        report_result = _run_coverage_report(config, progress=progress, base_dir=base_dir)
        return {
            "coverage_index": index_result,
            "coverage_run": run_result,
            "coverage_report": report_result,
        }

    raise ValueError(f"unsupported job type: {job_type.value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FilteringCV dataset builder job worker")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--db-path", required=True, type=Path)
    args = parser.parse_args(argv)

    store = JobStore(args.db_path)
    job = store.get_job(args.job_id)
    cancel_token = FileCancellationToken(store.cancel_token_path(job.id))
    progress = JobProgressWriter(store, job.id)

    store.update_status(job.id, JobStatus.RUNNING, pid=os.getpid())
    progress(ProgressEvent(stage="worker", message="started", metadata={"pid": os.getpid()}))

    try:
        result = _run_job(
            job.job_type,
            Path(args.config),
            force=job.force,
            progress=progress,
            cancellation=cancel_token,
        )
        if cancel_token.cancelled:
            store.update_status(job.id, JobStatus.CANCELLED, clear_pid=True)
            return 0
        store.update_status(job.id, JobStatus.SUCCEEDED, result=result, clear_pid=True)
        progress(ProgressEvent(stage="worker", message="finished"))
        return 0
    except RuntimeError as exc:
        if "cancelled" in str(exc).lower() or cancel_token.cancelled:
            store.update_status(job.id, JobStatus.CANCELLED, clear_pid=True)
            return 0
        store.update_status(job.id, JobStatus.FAILED, error_message=str(exc), clear_pid=True)
        return 1
    except Exception as exc:
        if cancel_token.cancelled:
            store.update_status(job.id, JobStatus.CANCELLED, clear_pid=True)
            return 0
        tb = traceback.format_exc()
        store.update_status(job.id, JobStatus.FAILED, error_message=f"{exc}\n{tb}", clear_pid=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
