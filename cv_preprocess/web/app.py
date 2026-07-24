from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cv_preprocess.jobs.progress import ProgressHub
from cv_preprocess.web.dependencies import AppSession, build_app_state
from cv_preprocess.web.routes import (
    audio,
    catalog,
    compare,
    config_api,
    dashboard,
    jobs,
    overrides,
    reports,
    session,
)
from cv_preprocess.web.websocket import create_websocket_router

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def create_app(config_path: Path | None, project_root: Path) -> FastAPI:
    project_root = project_root.resolve()
    app_state = build_app_state(config_path, project_root) if config_path is not None else None
    app_session = AppSession(project_root=project_root, app_state=app_state)
    progress_hub = ProgressHub()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app_session.app_state is not None:
            app_session.app_state.job_store.mark_stale_running_as_interrupted()
        app.state.app_session = app_session
        app.state.progress_hub = progress_hub
        # backward-compat alias; prefer app_session
        app.state.app_state = app_session.app_state
        yield
        if app_session.app_state is not None:
            app_session.app_state.job_runner.shutdown()

    app = FastAPI(title="FilteringCV Dataset Builder", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
    app.include_router(catalog.router, prefix="/api/catalog", tags=["catalog"])
    app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
    app.include_router(audio.router, prefix="/api/audio", tags=["audio"])
    app.include_router(overrides.router, prefix="/api/overrides", tags=["overrides"])
    app.include_router(compare.router, prefix="/api/compare", tags=["compare"])
    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
    app.include_router(config_api.router, prefix="/api/config", tags=["config"])
    app.include_router(session.router, prefix="/api/session", tags=["session"])

    ws_router = create_websocket_router(progress_hub)
    app.include_router(ws_router)

    frontend_dist = (project_root / "frontend" / "dist").resolve()
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app
