from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

from cv_preprocess.web.dependencies import (
    AppSession,
    AppState,
    build_app_state,
    get_app_session,
    resolve_within_root,
)
from cv_preprocess.web.last_config import to_project_relative, write_last_config

router = APIRouter()


class SessionStatusResponse(BaseModel):
    bound: bool
    config_path: str | None
    project_root: str


class ConfigListItem(BaseModel):
    path: str
    exists: bool


class ConfigListResponse(BaseModel):
    configs: list[ConfigListItem]


class BindRequest(BaseModel):
    path: str


class CreateRequest(BaseModel):
    path: str = "config/default.yaml"
    overwrite: bool = False


def _session_status(session: AppSession) -> SessionStatusResponse:
    if session.app_state is None:
        return SessionStatusResponse(
            bound=False,
            config_path=None,
            project_root=str(session.project_root),
        )
    return SessionStatusResponse(
        bound=True,
        config_path=to_project_relative(session.project_root, session.app_state.config_path),
        project_root=str(session.project_root),
    )


def _sync_app_state_alias(request: Request, state: AppState | None) -> None:
    request.app.state.app_state = state


def bind_session(session: AppSession, config_path: Path) -> AppState:
    if session.app_state is not None:
        session.app_state.job_runner.shutdown()
    state = build_app_state(config_path, session.project_root)
    session.app_state = state
    write_last_config(
        session.project_root,
        to_project_relative(session.project_root, state.config_path),
    )
    return state


def _require_yaml_file(path: Path) -> None:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise HTTPException(status_code=400, detail="config path must be a .yaml or .yml file")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="config file not found")


def _bind_or_raise(session: AppSession, config_path: Path) -> AppState:
    try:
        return bind_session(session, config_path)
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", ()))
            msg = err.get("msg", "validation error")
            errors.append(f"{loc}: {msg}" if loc else str(msg))
        raise HTTPException(status_code=400, detail=errors or ["invalid config"]) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=SessionStatusResponse)
def get_session(request: Request) -> SessionStatusResponse:
    return _session_status(get_app_session(request))


@router.get("/configs", response_model=ConfigListResponse)
def list_configs(request: Request) -> ConfigListResponse:
    session = get_app_session(request)
    config_dir = session.project_root / "config"
    items: list[ConfigListItem] = []
    if config_dir.is_dir():
        paths = sorted(
            {
                p
                for pattern in ("*.yaml", "*.yml")
                for p in config_dir.glob(pattern)
                if p.is_file()
            },
            key=lambda p: p.as_posix(),
        )
        for path in paths:
            rel = to_project_relative(session.project_root, path)
            items.append(ConfigListItem(path=rel, exists=True))
    return ConfigListResponse(configs=items)


@router.post("/bind", response_model=SessionStatusResponse)
def bind_config(request: Request, body: BindRequest) -> SessionStatusResponse:
    session = get_app_session(request)
    target = resolve_within_root(session.project_root, body.path)
    _require_yaml_file(target)
    _bind_or_raise(session, target)
    _sync_app_state_alias(request, session.app_state)
    return _session_status(session)


@router.post("/create", response_model=SessionStatusResponse)
def create_config(request: Request, body: CreateRequest) -> SessionStatusResponse:
    session = get_app_session(request)
    source = resolve_within_root(session.project_root, "config/example.yaml")
    if not source.is_file():
        raise HTTPException(status_code=404, detail="config/example.yaml not found")
    target = resolve_within_root(session.project_root, body.path)
    if target.suffix.lower() not in {".yaml", ".yml"}:
        raise HTTPException(status_code=400, detail="config path must be a .yaml or .yml file")
    if target.exists() and not body.overwrite:
        raise HTTPException(status_code=409, detail="config already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    _bind_or_raise(session, target)
    _sync_app_state_alias(request, session.app_state)
    return _session_status(session)


@router.post("/unbind", response_model=SessionStatusResponse)
def unbind_config(request: Request) -> SessionStatusResponse:
    session = get_app_session(request)
    if session.app_state is not None:
        if session.app_state.job_store.has_active_jobs():
            raise HTTPException(status_code=409, detail="active jobs prevent unbind")
        session.app_state.job_runner.shutdown()
        session.app_state = None
        _sync_app_state_alias(request, None)
    return _session_status(session)
