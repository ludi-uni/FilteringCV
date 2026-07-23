from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from cv_preprocess.application.common import ProgressEvent
from cv_preprocess.jobs.models import ProgressRecord
from cv_preprocess.jobs.store import JobStore

ProgressListener = Callable[[ProgressRecord], None]


class FileCancellationToken:
    def __init__(self, token_path: Path) -> None:
        self.token_path = Path(token_path)

    @property
    def cancelled(self) -> bool:
        return self.token_path.is_file()

    def cancel(self) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text("1", encoding="utf-8")

    def clear(self) -> None:
        if self.token_path.is_file():
            self.token_path.unlink()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError("operation cancelled")


class JobProgressWriter:
    """Persist progress to SQLite + JSONL.

    High-frequency stage updates (e.g. per-clip analyze) are throttled so we do
    not open a SQLite transaction for every clip. Milestone events
    (``current is None`` or first/last) always flush.
    """

    def __init__(
        self,
        store: JobStore,
        job_id: str,
        *,
        listeners: list[ProgressListener] | None = None,
        min_interval_sec: float = 0.5,
        min_step: int = 25,
    ) -> None:
        self.store = store
        self.job_id = job_id
        self._jsonl_path = store.progress_jsonl_path(job_id)
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._listeners = list(listeners or [])
        self._min_interval_sec = float(min_interval_sec)
        self._min_step = max(1, int(min_step))
        self._last_flush_monotonic = 0.0
        self._last_flush_current: int | None = None
        self._last_flush_message: str | None = None

    def add_listener(self, listener: ProgressListener) -> None:
        self._listeners.append(listener)

    def _should_flush(self, event: ProgressEvent) -> bool:
        if event.current is None or event.total is None:
            return True
        if event.current <= 1 or event.current >= event.total:
            return True
        if event.message != self._last_flush_message:
            return True
        phase = event.metadata.get("phase") if event.metadata else None
        if phase in {"prepare", "reserve", "done", "load", "features", "start", "complete", "split"}:
            return True
        now = time.monotonic()
        if (now - self._last_flush_monotonic) >= self._min_interval_sec:
            return True
        if self._last_flush_current is None:
            return True
        if (event.current - self._last_flush_current) >= self._min_step:
            return True
        return False

    def __call__(self, event: ProgressEvent) -> None:
        if not self._should_flush(event):
            return
        record = ProgressRecord(
            job_id=self.job_id,
            stage=event.stage,
            message=event.message,
            current=event.current,
            total=event.total,
            fraction=event.fraction,
            metadata=dict(event.metadata),
        )
        saved = self.store.append_progress(record)
        payload = saved.model_dump(mode="json")
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._last_flush_monotonic = time.monotonic()
        self._last_flush_current = event.current
        self._last_flush_message = event.message
        for listener in self._listeners:
            listener(saved)


class ProgressHub:
    """In-process pub/sub for WebSocket broadcasting."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[ProgressRecord]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def subscribe(self, job_id: str) -> asyncio.Queue[ProgressRecord]:
        queue: asyncio.Queue[ProgressRecord] = asyncio.Queue()
        async with self._lock:
            self._subscribers[job_id].add(queue)
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue[ProgressRecord]) -> None:
        async with self._lock:
            subscribers = self._subscribers.get(job_id)
            if subscribers is None:
                return
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(job_id, None)

    def publish(self, record: ProgressRecord) -> None:
        subscribers = list(self._subscribers.get(record.job_id, ()))
        for queue in subscribers:
            try:
                queue.put_nowait(record)
            except asyncio.QueueFull:
                continue

    def make_listener(self) -> ProgressListener:
        return self.publish
