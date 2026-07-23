from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from cv_preprocess.jobs.models import CreateJobRequest, JobRecord, JobSummary, TERMINAL_JOB_STATUSES
from cv_preprocess.jobs.models import ProgressRecord
from cv_preprocess.web.dependencies import get_app_state

router = APIRouter()


def _to_summary(job: JobRecord) -> JobSummary:
    return JobSummary(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error_message=job.error_message,
    )


@router.post("", response_model=JobRecord)
def create_job(request: Request, body: CreateJobRequest) -> JobRecord:
    state = get_app_state(request)
    job = state.job_store.create_job(
        job_type=body.job_type,
        config_path=state.config_path,
        force=body.force,
    )
    state.job_runner.start_job(job.id)
    return job


@router.get("", response_model=list[JobSummary])
def list_jobs(request: Request, limit: int = 50, offset: int = 0) -> list[JobSummary]:
    state = get_app_state(request)
    jobs = state.job_store.list_jobs(limit=limit, offset=offset)
    return [_to_summary(job) for job in jobs]


@router.get("/{job_id}", response_model=JobRecord)
def get_job(request: Request, job_id: str) -> JobRecord:
    state = get_app_state(request)
    try:
        return state.job_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@router.post("/{job_id}/cancel", response_model=JobRecord)
def cancel_job(request: Request, job_id: str) -> JobRecord:
    state = get_app_state(request)
    try:
        job = state.job_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    if job.status in TERMINAL_JOB_STATUSES:
        return job
    state.job_runner.cancel_job(job_id)
    return state.job_store.get_job(job_id)


@router.get("/{job_id}/progress", response_model=list[ProgressRecord])
def list_progress(
    request: Request,
    job_id: str,
    after_id: int = 0,
    limit: int = 500,
) -> list[ProgressRecord]:
    state = get_app_state(request)
    try:
        state.job_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return state.job_store.list_progress(job_id, after_id=after_id, limit=limit)
