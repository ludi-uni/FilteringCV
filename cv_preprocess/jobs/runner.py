from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from cv_preprocess.jobs.models import JobStatus, TERMINAL_JOB_STATUSES
from cv_preprocess.jobs.progress import FileCancellationToken
from cv_preprocess.jobs.store import JobStore


class JobRunner:
    def __init__(self, store: JobStore, *, config_path: Path) -> None:
        self.store = store
        self.config_path = Path(config_path)
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.Lock()

    def start_job(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job.status != JobStatus.QUEUED:
            raise ValueError(f"job {job_id} is not queued (status={job.status.value})")

        cancel_token = FileCancellationToken(self.store.cancel_token_path(job_id))
        cancel_token.clear()

        cmd = [
            sys.executable,
            "-m",
            "cv_preprocess.jobs.worker",
            "--job-id",
            job_id,
            "--config",
            str(self.config_path),
            "--db-path",
            str(self.store.db_path),
        ]
        popen_kwargs: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.PIPE,
            "shell": False,
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(cmd, **popen_kwargs)
        with self._lock:
            self._processes[job_id] = process

        watcher = threading.Thread(
            target=self._watch_process,
            args=(job_id, process),
            name=f"job-watcher-{job_id[:8]}",
            daemon=True,
        )
        watcher.start()

    def cancel_job(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job.status in TERMINAL_JOB_STATUSES:
            return
        if job.status == JobStatus.QUEUED:
            self.store.update_status(job_id, JobStatus.CANCELLED, clear_pid=True)
            return

        self.store.update_status(job_id, JobStatus.CANCELLING)
        FileCancellationToken(self.store.cancel_token_path(job_id)).cancel()
        with self._lock:
            process = self._processes.get(job_id)
        if process is not None and process.poll() is None:
            self._terminate_process(process)

    def shutdown(self) -> None:
        with self._lock:
            processes = list(self._processes.items())
        for job_id, process in processes:
            if process.poll() is None:
                self.store.update_status(job_id, JobStatus.INTERRUPTED)
                self._terminate_process(process)

    def _watch_process(self, job_id: str, process: subprocess.Popen[bytes]) -> None:
        return_code = process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        with self._lock:
            self._processes.pop(job_id, None)

        try:
            job = self.store.get_job(job_id)
        except KeyError:
            return

        if job.status in TERMINAL_JOB_STATUSES:
            return

        if job.status == JobStatus.CANCELLING:
            self.store.update_status(job_id, JobStatus.CANCELLED, clear_pid=True)
            return

        if return_code == 0:
            if job.status != JobStatus.SUCCEEDED:
                self.store.update_status(job_id, JobStatus.SUCCEEDED, clear_pid=True)
            return

        message = stderr.strip() or f"worker exited with code {return_code}"
        self.store.update_status(
            job_id,
            JobStatus.FAILED,
            error_message=message,
            clear_pid=True,
        )

    def _terminate_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    return
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
