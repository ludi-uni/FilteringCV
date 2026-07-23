from __future__ import annotations

import time
from pathlib import Path

from cv_preprocess.config import load_config
from cv_preprocess.jobs.models import JobStatus, JobType
from cv_preprocess.jobs.runner import JobRunner
from cv_preprocess.jobs.store import JobStore


def test_worker_stderr_flood_does_not_deadlock(tmp_path: Path) -> None:
    """Regression: stderr=PIPE without a reader freezes workers once the pipe fills."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
schema_version: 2
input:
  corpus_root: .
  clip_tsv: validated.tsv
dataset_builder:
  enabled: true
  work_dir: work
""".strip()
        + "\n",
        encoding="utf-8",
    )
    load_config(config_path)
    db_path = tmp_path / "work" / "jobs.sqlite3"
    store = JobStore(db_path)
    job = store.create_job(job_type=JobType.ANALYZE, config_path=config_path, force=False)

    runner = JobRunner(store, config_path=config_path)
    # Replace the worker module invocation with a stderr-flooding stub.
    original_start = runner.start_job

    def start_with_flood(job_id: str) -> None:
        import os
        import subprocess
        import sys
        import threading

        from cv_preprocess.jobs.progress import FileCancellationToken

        job_rec = store.get_job(job_id)
        assert job_rec.status == JobStatus.QUEUED
        FileCancellationToken(store.cancel_token_path(job_id)).clear()
        stderr_path = store.progress_jsonl_path(job_id).with_name("worker.stderr.log")
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_file = stderr_path.open("wb")
        # Flood ~256KiB to stderr then exit 0 — would hang with unread PIPE.
        code = (
            "import sys\n"
            "sys.stderr.write('x' * (256 * 1024))\n"
            "sys.stderr.flush()\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            shell=False,
            start_new_session=True if os.name == "posix" else False,
        )
        with runner._lock:
            runner._processes[job_id] = process
            runner._stderr_handles[job_id] = stderr_file
        threading.Thread(
            target=runner._watch_process,
            args=(job_id, process, stderr_path),
            daemon=True,
        ).start()
        store.update_status(job_id, JobStatus.RUNNING, pid=process.pid)

    runner.start_job = start_with_flood  # type: ignore[method-assign]
    start_with_flood(job.id)

    deadline = time.time() + 10.0
    while time.time() < deadline:
        current = store.get_job(job.id)
        if current.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("worker stalled — stderr pipe deadlock regression")

    assert store.get_job(job.id).status == JobStatus.SUCCEEDED
    stderr_log = store.progress_jsonl_path(job.id).with_name("worker.stderr.log")
    assert stderr_log.is_file()
    assert stderr_log.stat().st_size >= 256 * 1024
    _ = original_start
