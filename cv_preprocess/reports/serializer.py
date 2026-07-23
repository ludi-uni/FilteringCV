from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def write_json_atomic(path: Path, payload: BaseModel | dict[str, Any]) -> None:
    """Write JSON via a .partial file and atomic rename."""
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        text = payload.model_dump_json(indent=2)
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)
