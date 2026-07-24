from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

LAST_CONFIG_DIR_NAME = ".filteringcv"
LAST_CONFIG_FILENAME = "last_config.json"


def last_config_path(project_root: Path) -> Path:
    return project_root.resolve() / LAST_CONFIG_DIR_NAME / LAST_CONFIG_FILENAME


def to_project_relative(project_root: Path, config_path: Path) -> str:
    root = project_root.resolve()
    resolved = config_path.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("config path outside project root") from exc
    return rel.as_posix()


def write_last_config(project_root: Path, config_relative: str) -> Path:
    root = project_root.resolve()
    # Normalize via Path to reject absolute / traversal when resolving
    candidate = (root / config_relative).resolve()
    rel = to_project_relative(root, candidate)
    path = last_config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_path": rel,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_last_config(project_root: Path) -> str | None:
    path = last_config_path(project_root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("config_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    if Path(raw).is_absolute() or ".." in Path(raw).parts:
        return None
    return Path(raw).as_posix()


def resolve_last_config_file(project_root: Path) -> Path | None:
    rel = read_last_config(project_root)
    if rel is None:
        return None
    candidate = (project_root.resolve() / rel).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate
