from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class JobType(str, Enum):
    SCAN = "scan"
    ANALYZE = "analyze"
    PLAN_SPLIT = "plan-split"
    SELECT = "select"
    MATERIALIZE = "materialize"
    AUDIT = "audit"
    BUILD = "build"


TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.CANCELLED,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.INTERRUPTED,
    }
)


class JobRecord(BaseModel):
    id: str
    job_type: JobType
    status: JobStatus = JobStatus.QUEUED
    config_path: str
    force: bool = False
    pid: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    result: dict[str, Any] | None = None


class ProgressRecord(BaseModel):
    id: int | None = None
    job_id: str
    stage: str
    message: str = ""
    current: int | None = None
    total: int | None = None
    fraction: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CreateJobRequest(BaseModel):
    job_type: JobType
    force: bool = False


class JobSummary(BaseModel):
    id: str
    job_type: JobType
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
