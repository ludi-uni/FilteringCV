from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

OverrideAction = Literal[
    "force_include",
    "force_exclude",
    "hard_reject",
    "return_to_reserve",
]


class ClipOverride(BaseModel):
    clip_id: str
    action: OverrideAction
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_overrides(path: Path) -> dict[str, ClipOverride]:
    path = Path(path)
    if not path.is_file():
        return {}
    overrides: dict[str, ClipOverride] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
            override = ClipOverride.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"invalid override at {path}:{line_no}: {exc}") from exc
        overrides[override.clip_id] = override
    return overrides


def resolved_overrides_path(work_dir: Path) -> Path:
    return Path(work_dir) / "overrides.jsonl"


def apply_override_flags(override: ClipOverride | None) -> str | None:
    if override is None:
        return None
    return override.action
