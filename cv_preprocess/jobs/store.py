from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from cv_preprocess.jobs.models import (
    TERMINAL_JOB_STATUSES,
    JobRecord,
    JobStatus,
    JobType,
    ProgressRecord,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_path TEXT NOT NULL,
                    force INTEGER NOT NULL DEFAULT 0,
                    pid INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    error_message TEXT,
                    result_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);

                CREATE TABLE IF NOT EXISTS progress_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    current INTEGER,
                    total INTEGER,
                    fraction REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_progress_job_id ON progress_events(job_id);
                """
            )

    def create_job(self, *, job_type: JobType, config_path: Path, force: bool = False) -> JobRecord:
        job_id = uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, job_type, status, config_path, force, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_type.value,
                    JobStatus.QUEUED.value,
                    str(config_path),
                    int(force),
                    now,
                    now,
                ),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job not found: {job_id}")
        return self._row_to_job(row)

    def list_jobs(self, *, limit: int = 100, offset: int = 0) -> list[JobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        pid: int | None = None,
        error_message: str | None = None,
        result: dict[str, Any] | None = None,
        clear_pid: bool = False,
    ) -> JobRecord:
        now = _utc_now()
        fields: list[str] = ["status = ?", "updated_at = ?"]
        values: list[Any] = [status.value, now]

        if status == JobStatus.RUNNING:
            fields.append("started_at = COALESCE(started_at, ?)")
            values.append(now)
        if status in TERMINAL_JOB_STATUSES:
            fields.append("finished_at = ?")
            values.append(now)
        if pid is not None:
            fields.append("pid = ?")
            values.append(pid)
        if clear_pid:
            fields.append("pid = NULL")
        if error_message is not None:
            fields.append("error_message = ?")
            values.append(error_message)
        if result is not None:
            fields.append("result_json = ?")
            values.append(json.dumps(result, ensure_ascii=False))

        values.append(job_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
                tuple(values),
            )
        return self.get_job(job_id)

    def mark_stale_running_as_interrupted(self) -> int:
        now = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?, finished_at = ?, pid = NULL,
                    error_message = COALESCE(error_message, 'interrupted on server restart')
                WHERE status IN (?, ?, ?)
                """,
                (
                    JobStatus.INTERRUPTED.value,
                    now,
                    now,
                    JobStatus.RUNNING.value,
                    JobStatus.CANCELLING.value,
                    JobStatus.QUEUED.value,
                ),
            )
            return int(cursor.rowcount)

    def has_active_jobs(self) -> bool:
        active = (
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
            JobStatus.CANCELLING.value,
        )
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT 1 FROM jobs WHERE status IN ({','.join('?' * len(active))}) LIMIT 1",
                active,
            ).fetchone()
        return row is not None

    def append_progress(self, record: ProgressRecord) -> ProgressRecord:
        now = _utc_now()
        metadata_json = json.dumps(record.metadata, ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO progress_events (
                    job_id, stage, message, current, total, fraction, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.job_id,
                    record.stage,
                    record.message,
                    record.current,
                    record.total,
                    record.fraction,
                    metadata_json,
                    now,
                ),
            )
            progress_id = int(cursor.lastrowid)
        return ProgressRecord(
            id=progress_id,
            job_id=record.job_id,
            stage=record.stage,
            message=record.message,
            current=record.current,
            total=record.total,
            fraction=record.fraction,
            metadata=record.metadata,
            created_at=datetime.fromisoformat(now),
        )

    def list_progress(self, job_id: str, *, after_id: int = 0, limit: int = 500) -> list[ProgressRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM progress_events
                WHERE job_id = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (job_id, after_id, limit),
            ).fetchall()
        return [self._row_to_progress(row) for row in rows]

    def cancel_token_path(self, job_id: str) -> Path:
        return self.db_path.parent / "jobs" / job_id / "cancel.token"

    def progress_jsonl_path(self, job_id: str) -> Path:
        return self.db_path.parent / "jobs" / job_id / "progress.jsonl"

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        result_json = row["result_json"]
        result = json.loads(result_json) if result_json else None
        return JobRecord(
            id=row["id"],
            job_type=JobType(row["job_type"]),
            status=JobStatus(row["status"]),
            config_path=row["config_path"],
            force=bool(row["force"]),
            pid=row["pid"],
            created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
            updated_at=_parse_dt(row["updated_at"]) or datetime.now(UTC),
            started_at=_parse_dt(row["started_at"]),
            finished_at=_parse_dt(row["finished_at"]),
            error_message=row["error_message"],
            result=result,
        )

    def _row_to_progress(self, row: sqlite3.Row) -> ProgressRecord:
        metadata = json.loads(row["metadata_json"] or "{}")
        return ProgressRecord(
            id=int(row["id"]),
            job_id=row["job_id"],
            stage=row["stage"],
            message=row["message"],
            current=row["current"],
            total=row["total"],
            fraction=row["fraction"],
            metadata=metadata if isinstance(metadata, dict) else {},
            created_at=_parse_dt(row["created_at"]) or datetime.now(UTC),
        )
