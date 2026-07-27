from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
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

logger = logging.getLogger(__name__)

# SQLite WAL needs reliable shared-memory locking. Windows-bind mounts under
# WSL/Docker (9p / drvfs / fuseblk) frequently raise OperationalError:
# "locking protocol" / "disk I/O error" under concurrent API + worker access.
_UNSAFE_FS_TYPES = frozenset(
    {
        "9p",
        "drvfs",
        "fuse",
        "fuseblk",
        "fusectl",
        "cifs",
        "smb",
        "smbfs",
        "nfs",
        "nfs4",
        "afs",
    }
)

_RETRYABLE_LOCK_MARKERS = (
    "locking protocol",
    "database is locked",
    "database is busy",
    "disk i/o error",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def filesystem_type(path: Path) -> str | None:
    """Best-effort mount type for ``path`` (Linux ``statvfs`` / ``/proc/mounts``)."""
    try:
        import psutil  # optional
    except Exception:
        psutil = None  # type: ignore[assignment]

    resolved = Path(path).resolve()
    if psutil is not None:
        try:
            parts = psutil.disk_partitions(all=True)
            best = None
            best_len = -1
            for part in parts:
                mount = part.mountpoint
                if resolved == Path(mount) or str(resolved).startswith(str(Path(mount)) + os.sep):
                    if len(mount) > best_len:
                        best = part.fstype
                        best_len = len(mount)
            if best:
                return best.lower()
        except Exception:
            pass

    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    best_fstype = None
    best_len = -1
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mountpoint, fstype = parts[1], parts[2]
        # /proc/mounts escapes spaces as \040
        mountpoint = mountpoint.replace("\\040", " ")
        if resolved == Path(mountpoint) or str(resolved).startswith(mountpoint.rstrip("/") + "/"):
            if len(mountpoint) > best_len:
                best_fstype = fstype.lower()
                best_len = len(mountpoint)
    return best_fstype


def prefers_rollback_journal(path: Path) -> bool:
    """True when WAL is unsafe on the filesystem backing ``path``."""
    fstype = filesystem_type(path)
    if fstype is None:
        # WSL/Docker Desktop often mounts Windows drives at /mnt/* or bind-mounts
        # the project at /workspace from a Windows host (seen as 9p).
        resolved = str(Path(path).resolve())
        if resolved.startswith("/mnt/"):
            return True
        return False
    return fstype in _UNSAFE_FS_TYPES or fstype.startswith("fuse")


def is_retryable_sqlite_error(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_LOCK_MARKERS)


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._use_wal = not prefers_rollback_journal(self.db_path)
        self._init_db()

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        # timeout= on connect is soft; busy_timeout is the authoritative wait.
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        # Avoid mmap on flaky network/9p mounts.
        if not self._use_wal:
            try:
                conn.execute("PRAGMA mmap_size=0")
            except sqlite3.Error:
                pass
            conn.execute("PRAGMA synchronous=FULL")
        else:
            conn.execute("PRAGMA synchronous=NORMAL")

    def _apply_journal_mode(self, conn: sqlite3.Connection) -> str:
        desired = "WAL" if self._use_wal else "DELETE"
        try:
            mode = str(conn.execute(f"PRAGMA journal_mode={desired}").fetchone()[0]).lower()
        except sqlite3.OperationalError as exc:
            logger.warning(
                "sqlite journal_mode=%s failed on %s (%s); falling back to DELETE",
                desired,
                self.db_path,
                exc,
            )
            self._use_wal = False
            mode = str(conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower()
        if desired == "WAL" and mode != "wal":
            # Filesystem accepted the pragma call but did not enable WAL.
            self._use_wal = False
            mode = str(conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower()
        return mode

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level="DEFERRED")
        self._configure_connection(conn)
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def _run_write(self, operation: Any, *, attempts: int = 8) -> Any:
        """Retry transient lock / 9p protocol errors on write transactions."""
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                with self._connect() as conn:
                    return operation(conn)
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if not is_retryable_sqlite_error(exc) or attempt >= attempts - 1:
                    raise
                # If WAL is the culprit, switch mid-process and retry.
                if "locking protocol" in str(exc).lower() and self._use_wal:
                    logger.warning(
                        "sqlite locking protocol on %s; switching journal_mode to DELETE",
                        self.db_path,
                    )
                    self._use_wal = False
                    try:
                        with self._connect() as conn:
                            self._apply_journal_mode(conn)
                    except sqlite3.Error:
                        pass
                time.sleep(0.05 * (2**attempt))
        assert last_exc is not None
        raise last_exc

    def _init_db(self) -> None:
        def _init(conn: sqlite3.Connection) -> None:
            self._apply_journal_mode(conn)
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

        self._run_write(_init)

    def create_job(self, *, job_type: JobType, config_path: Path, force: bool = False) -> JobRecord:
        job_id = uuid.uuid4().hex
        now = _utc_now()

        def _insert(conn: sqlite3.Connection) -> None:
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

        self._run_write(_insert)
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

        def _update(conn: sqlite3.Connection) -> None:
            conn.execute(
                f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?",
                tuple(values),
            )

        self._run_write(_update)
        return self.get_job(job_id)

    def mark_stale_running_as_interrupted(self) -> int:
        now = _utc_now()

        def _mark(conn: sqlite3.Connection) -> int:
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

        return int(self._run_write(_mark))

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

        def _insert(conn: sqlite3.Connection) -> int:
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
            return int(cursor.lastrowid)

        progress_id = int(self._run_write(_insert))
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
