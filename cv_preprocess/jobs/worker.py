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
from cv_preprocess.jobs.models import JobStatus, JobType
from cv_preprocess.jobs.progress import FileCancellationToken, JobProgressWriter
from cv_preprocess.jobs.store import JobStore


def _serialize_result(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if isinstance(payload, tuple):
        return {"items": [_serialize_result(item) for item in payload]}
    if isinstance(payload, dict):
        return payload
    return {"value": payload}


def _run_job(
    job_type: JobType,
    config_path: Path,
    *,
    force: bool,
    progress: JobProgressWriter,
    cancellation: FileCancellationToken,
) -> dict[str, Any]:
    config = load_config(config_path)
    if not config.dataset_builder.enabled and job_type != JobType.SCAN:
        raise ValueError("dataset_builder.enabled must be true for dataset builder jobs")

    work_dir = Path(config.dataset_builder.work_dir)

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
