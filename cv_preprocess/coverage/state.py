"""Persistent coverage-run state with atomic writes."""

from __future__ import annotations

import json
from pathlib import Path

from cv_preprocess.coverage.models import CoverageRunState, SpeakerPassStats, StopReason
from cv_preprocess.reports.serializer import write_json_atomic

STATE_FILENAME = "run-state.json"


def state_path(run_dir: Path) -> Path:
    return Path(run_dir) / STATE_FILENAME


def save_run_state(run_dir: Path, state: CoverageRunState) -> Path:
    path = state_path(run_dir)
    write_json_atomic(path, state)
    return path


def load_run_state(run_dir: Path) -> CoverageRunState:
    path = state_path(run_dir)
    if not path.is_file():
        raise FileNotFoundError(f"coverage run state not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Compat: speaker_pass_stats may be plain dicts
    stats_raw = raw.get("speaker_pass_stats") or {}
    parsed_stats = {
        key: SpeakerPassStats.model_validate(value) if isinstance(value, dict) else value
        for key, value in stats_raw.items()
    }
    raw["speaker_pass_stats"] = parsed_stats
    if isinstance(raw.get("status"), str):
        raw["status"] = StopReason(raw["status"])
    return CoverageRunState.model_validate(raw)


def append_jsonl(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
