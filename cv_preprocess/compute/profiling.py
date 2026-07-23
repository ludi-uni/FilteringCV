from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ResourceSnapshot:
    wall_sec: float
    cpu_sec: float
    rss_bytes: int | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "wall_sec": self.wall_sec,
            "cpu_sec": self.cpu_sec,
            "rss_bytes": self.rss_bytes,
        }


def _rss_bytes() -> int | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = int(usage.ru_maxrss)
        if rss > 0:
            return rss
    except (ImportError, OSError, ValueError):
        pass
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def resource_snapshot() -> ResourceSnapshot:
    cpu_sec = time.process_time()
    return ResourceSnapshot(
        wall_sec=time.perf_counter(),
        cpu_sec=cpu_sec,
        rss_bytes=_rss_bytes(),
    )


def measure_callable(fn: Callable[[], T]) -> tuple[T, dict[str, Any]]:
    """Run ``fn`` and return its result plus wall/cpu/rss deltas."""
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    start_rss = _rss_bytes()
    result = fn()
    end_wall = time.perf_counter()
    end_cpu = time.process_time()
    end_rss = _rss_bytes()
    timing = {
        "wall_sec": end_wall - start_wall,
        "cpu_sec": end_cpu - start_cpu,
        "rss_bytes_start": start_rss,
        "rss_bytes_end": end_rss,
        "rss_bytes_delta": (
            (end_rss - start_rss) if start_rss is not None and end_rss is not None else None
        ),
    }
    return result, timing
