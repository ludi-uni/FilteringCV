from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request

from cv_preprocess.jobs.models import JobStatus
from cv_preprocess.web.dependencies import get_app_state

router = APIRouter()


@router.get("")
def dashboard_summary(request: Request) -> dict[str, Any]:
    state = get_app_state(request)
    jobs = state.job_store.list_jobs(limit=200)
    status_counts: dict[str, int] = {status.value: 0 for status in JobStatus}
    for job in jobs:
        status_counts[job.status.value] = status_counts.get(job.status.value, 0) + 1

    manifest_path = state.work_dir / "run_manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            manifest = payload

    clips_path = state.catalog_dir / "clips.parquet"
    catalog_ready = clips_path.is_file()
    return {
        "config_path": str(state.config_path),
        "work_dir": str(state.work_dir),
        "output_dir": str(state.output_dir),
        "catalog_ready": catalog_ready,
        "job_status_counts": status_counts,
        "recent_jobs": [
            {
                "id": job.id,
                "job_type": job.job_type.value,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
            }
            for job in jobs[:10]
        ],
        "run_manifest": manifest,
    }
