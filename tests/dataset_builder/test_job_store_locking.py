"""JobStore SQLite hardening for 9p / WSL mounts."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from cv_preprocess.application.common import ProgressEvent
from cv_preprocess.jobs.models import JobType, ProgressRecord
from cv_preprocess.jobs.progress import JobProgressWriter
from cv_preprocess.jobs.store import (
    JobStore,
    filesystem_type,
    is_retryable_sqlite_error,
    prefers_rollback_journal,
)


def test_prefers_rollback_on_9p() -> None:
    with patch("cv_preprocess.jobs.store.filesystem_type", return_value="9p"):
        assert prefers_rollback_journal(Path("/workspace/work/jobs.sqlite3")) is True


def test_prefers_wal_on_ext4() -> None:
    with patch("cv_preprocess.jobs.store.filesystem_type", return_value="ext4"):
        assert prefers_rollback_journal(Path("/var/lib/filteringcv/jobs.sqlite3")) is False


def test_job_store_uses_delete_journal_on_unsafe_fs(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    with patch("cv_preprocess.jobs.store.prefers_rollback_journal", return_value=True):
        store = JobStore(db_path)
    assert store._use_wal is False
    conn = sqlite3.connect(db_path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert str(mode).lower() == "delete"


def test_append_progress_works(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.create_job(job_type=JobType.ANALYZE, config_path=tmp_path / "c.yaml")
    assert is_retryable_sqlite_error(sqlite3.OperationalError("locking protocol"))
    record = store.append_progress(
        ProgressRecord(job_id=job.id, stage="analyze", message="ok", current=1, total=10)
    )
    assert record.id is not None
    assert store.list_progress(job.id)


def test_progress_writer_survives_sqlite_lock(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.create_job(job_type=JobType.ANALYZE, config_path=tmp_path / "c.yaml")
    writer = JobProgressWriter(store, job.id, min_interval_sec=0.0, min_step=1)

    def boom(_record: ProgressRecord) -> ProgressRecord:
        raise sqlite3.OperationalError("locking protocol")

    with patch.object(store, "append_progress", side_effect=boom):
        writer(
            ProgressEvent(
                stage="analyze",
                message="clip",
                current=1,
                total=100,
                fraction=0.01,
                metadata={},
            )
        )
    # JSONL fallback must exist even when sqlite fails
    assert store.progress_jsonl_path(job.id).is_file()
    lines = store.progress_jsonl_path(job.id).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_filesystem_type_returns_string_or_none() -> None:
    # Smoke: should not raise on current environment
    fstype = filesystem_type(Path("/workspace"))
    assert fstype is None or isinstance(fstype, str)
