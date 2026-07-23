from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request

from cv_preprocess.application.materialize import resolve_materialize_output_root
from cv_preprocess.config import PipelineConfig, load_config
from cv_preprocess.jobs.runner import JobRunner
from cv_preprocess.jobs.store import JobStore


@dataclass(frozen=True)
class AppState:
    config_path: Path
    project_root: Path
    config: PipelineConfig
    job_store: JobStore
    job_runner: JobRunner
    work_dir: Path
    output_dir: Path
    catalog_dir: Path
    audio_cache_dir: Path


def get_app_state(request: Request) -> AppState:
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        raise HTTPException(status_code=500, detail="application not initialized")
    return state


def reject_path_traversal(value: str) -> None:
    if not value or value.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="invalid path")
    parts = Path(value).parts
    if ".." in parts:
        raise HTTPException(status_code=400, detail="path traversal rejected")


def resolve_within_root(root: Path, relative: str) -> Path:
    reject_path_traversal(relative)
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path outside allowed root") from exc
    return candidate


def allowed_audio_roots(state: AppState) -> list[Path]:
    roots = [
        state.catalog_dir,
        state.audio_cache_dir,
        state.output_dir,
        state.work_dir / "catalog",
        state.work_dir / "audio_cache",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def resolve_audio_file(state: AppState, relative_path: str) -> Path:
    reject_path_traversal(relative_path)
    normalized = Path(relative_path)
    if normalized.is_absolute():
        raise HTTPException(status_code=400, detail="invalid path")
    for root in allowed_audio_roots(state):
        candidate = (root / normalized).resolve()
        if not candidate.is_file():
            continue
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        return candidate
    raise HTTPException(status_code=404, detail="audio file not found")


def build_app_state(config_path: Path, project_root: Path) -> AppState:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    config = load_config(config_path)
    work_dir = (project_root / config.dataset_builder.work_dir).resolve()
    output_dir = resolve_materialize_output_root(config)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    catalog_dir = work_dir / "catalog"
    audio_cache_dir = work_dir / "audio_cache"
    db_path = work_dir / "jobs.sqlite3"
    store = JobStore(db_path)
    runner = JobRunner(store, config_path=config_path)
    return AppState(
        config_path=config_path,
        project_root=project_root,
        config=config,
        job_store=store,
        job_runner=runner,
        work_dir=work_dir,
        output_dir=output_dir,
        catalog_dir=catalog_dir,
        audio_cache_dir=audio_cache_dir,
    )
