# GUI Recommended Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make interactive GUI the recommended operator path via optional `-c`, project-local last-config memory, a setup bind/create screen, a start helper, and GUI-first docs.

**Architecture:** Keep one FastAPI process. Introduce an `AppSession` that holds `project_root` plus optional bound `AppState`. Unbound mode serves only `/api/session*`; bind/create rebuilds `AppState` in-process. Frontend gates on `GET /api/session` and shows a Setup page until bound. CLI resolves `-c` → last_config → unbound.

**Tech Stack:** Python 3.10+, FastAPI, Typer, Pydantic v2, React + TypeScript + Vite, pytest, bash start script.

## Global Constraints

- Last-config path: `.filteringcv/last_config.json` under `project_root` (gitignored); store **relative** `config_path` only.
- Default create source: `config/example.yaml`; default target: `config/default.yaml`.
- Active jobs (`queued` / `running` / `cancelling`) block unbind/rebind with HTTP 409; no force-switch flag.
- Unbound main APIs return HTTP 503 with detail `config not bound`.
- Paths for list/bind/create must stay under `project_root`; reject traversal.
- Do not GUI-port legacy preprocess/phoneme/partition tools in this plan.
- Existing `create_app(config_path, project_root)` callers that pass a real config must keep working (bound at startup).

## File structure

| Path | Responsibility |
|------|----------------|
| `cv_preprocess/web/last_config.py` | Read/write/normalize `.filteringcv/last_config.json` |
| `cv_preprocess/web/session_resolve.py` | Resolve startup config: CLI `-c` / last_config / None |
| `cv_preprocess/web/dependencies.py` | `AppSession`, optional `AppState`, `require_app_state` (503) |
| `cv_preprocess/web/app.py` | `create_app(config_path \| None, …)`, mount session router |
| `cv_preprocess/web/routes/session.py` | Session REST API |
| `cv_preprocess/jobs/store.py` | `has_active_jobs()` helper |
| `cv_preprocess/cli.py` | Optional `-c`, resolve + create_app |
| `scripts/start-gui.sh` | Build frontend if needed; start gui |
| `frontend/src/api/session.ts` (+ types) | Session client |
| `frontend/src/pages/Setup.tsx` | Bind / create UI |
| `frontend/src/session/SessionGate.tsx` | Route gate + switch config |
| `frontend/src/App.tsx`, `Layout.tsx` | Wire Setup / Switch |
| `.gitignore`, `README.md`, `docs/gui.md`, `docs/開発環境.md`, `docs/architecture.md` | Positioning |
| `tests/dataset_builder/test_last_config.py` | Unit tests for last_config |
| `tests/dataset_builder/test_session_api.py` | Session API + unbound 503 |
| `tests/dataset_builder/test_session_resolve.py` | Startup resolution |

---

### Task 1: last_config helper

**Files:**
- Create: `cv_preprocess/web/last_config.py`
- Test: `tests/dataset_builder/test_last_config.py`

**Interfaces:**
- Consumes: `pathlib.Path`, `datetime` UTC, `json`
- Produces:
  - `LAST_CONFIG_DIR_NAME = ".filteringcv"`
  - `LAST_CONFIG_FILENAME = "last_config.json"`
  - `last_config_path(project_root: Path) -> Path`
  - `read_last_config(project_root: Path) -> str | None`  # relative path or None
  - `write_last_config(project_root: Path, config_relative: str) -> Path`
  - `to_project_relative(project_root: Path, config_path: Path) -> str`
  - `resolve_last_config_file(project_root: Path) -> Path | None`  # absolute file if exists and is file

- [ ] **Step 1: Write the failing tests**

```python
# tests/dataset_builder/test_last_config.py
from __future__ import annotations

from pathlib import Path

import pytest

from cv_preprocess.web.last_config import (
    read_last_config,
    resolve_last_config_file,
    to_project_relative,
    write_last_config,
)


def test_write_and_read_relative(tmp_path: Path) -> None:
    rel = write_last_config(tmp_path, "config/default.yaml")
    assert rel.name == "last_config.json"
    assert read_last_config(tmp_path) == "config/default.yaml"


def test_to_project_relative_rejects_outside(tmp_path: Path) -> None:
    outside = tmp_path.parent / "other.yaml"
    outside.write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        to_project_relative(tmp_path, outside)


def test_resolve_missing_returns_none(tmp_path: Path) -> None:
    assert resolve_last_config_file(tmp_path) is None


def test_resolve_stale_path_returns_none(tmp_path: Path) -> None:
    write_last_config(tmp_path, "config/missing.yaml")
    assert resolve_last_config_file(tmp_path) is None


def test_resolve_valid_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config"
    cfg.mkdir()
    target = cfg / "default.yaml"
    target.write_text("schema_version: 2\n", encoding="utf-8")
    write_last_config(tmp_path, "config/default.yaml")
    assert resolve_last_config_file(tmp_path) == target.resolve()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace && .venv/bin/pytest tests/dataset_builder/test_last_config.py -v`  
Expected: FAIL with `ModuleNotFoundError` or import error for `cv_preprocess.web.last_config`

- [ ] **Step 3: Implement `cv_preprocess/web/last_config.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace && .venv/bin/pytest tests/dataset_builder/test_last_config.py -v`  
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add cv_preprocess/web/last_config.py tests/dataset_builder/test_last_config.py
git commit -m "feat: add project-local last_config helper for GUI"
```

---

### Task 2: Session resolve + optional AppSession / require_bound

**Files:**
- Create: `cv_preprocess/web/session_resolve.py`
- Modify: `cv_preprocess/web/dependencies.py`
- Modify: `cv_preprocess/web/app.py`
- Modify: `cv_preprocess/jobs/store.py` (add `has_active_jobs`)
- Test: `tests/dataset_builder/test_session_resolve.py`
- Ensure existing: `tests/dataset_builder/test_api_jobs.py` still pass

**Interfaces:**
- Consumes: `last_config.*`, `build_app_state`, `JobStore`
- Produces:
  - `resolve_gui_config_path(*, project_root: Path, cli_config: Path | None) -> Path | None`
  - `@dataclass class AppSession: project_root: Path; app_state: AppState | None`
  - `get_app_session(request) -> AppSession`
  - `get_app_state(request) -> AppState` raises HTTP 503 `config not bound` when unbound
  - `JobStore.has_active_jobs() -> bool`
  - `create_app(config_path: Path | None, project_root: Path) -> FastAPI`

- [ ] **Step 1: Write failing resolve tests**

```python
# tests/dataset_builder/test_session_resolve.py
from __future__ import annotations

from pathlib import Path

from cv_preprocess.web.last_config import write_last_config
from cv_preprocess.web.session_resolve import resolve_gui_config_path


def test_cli_config_wins(tmp_path: Path) -> None:
    cfg = tmp_path / "a.yaml"
    cfg.write_text("x\n", encoding="utf-8")
    write_last_config(tmp_path, "b.yaml")
    assert resolve_gui_config_path(project_root=tmp_path, cli_config=cfg) == cfg.resolve()


def test_falls_back_to_last(tmp_path: Path) -> None:
    cfg = tmp_path / "config" / "default.yaml"
    cfg.parent.mkdir()
    cfg.write_text("x\n", encoding="utf-8")
    write_last_config(tmp_path, "config/default.yaml")
    assert resolve_gui_config_path(project_root=tmp_path, cli_config=None) == cfg.resolve()


def test_none_when_no_cli_no_last(tmp_path: Path) -> None:
    assert resolve_gui_config_path(project_root=tmp_path, cli_config=None) is None
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /workspace && .venv/bin/pytest tests/dataset_builder/test_session_resolve.py -v`  
Expected: FAIL import `session_resolve`

- [ ] **Step 3: Implement resolve + AppSession + has_active_jobs + create_app signature**

`cv_preprocess/web/session_resolve.py`:

```python
from __future__ import annotations

from pathlib import Path

from cv_preprocess.web.last_config import resolve_last_config_file


def resolve_gui_config_path(*, project_root: Path, cli_config: Path | None) -> Path | None:
    if cli_config is not None:
        return cli_config.resolve()
    return resolve_last_config_file(project_root)
```

In `JobStore`, add:

```python
def has_active_jobs(self) -> bool:
    active = ("queued", "running", "cancelling")
    with self._connect() as conn:
        row = conn.execute(
            f"SELECT 1 FROM jobs WHERE status IN ({','.join('?' * len(active))}) LIMIT 1",
            active,
        ).fetchone()
    return row is not None
```

Update `dependencies.py`:

```python
@dataclass
class AppSession:
    project_root: Path
    app_state: AppState | None = None


def get_app_session(request: Request) -> AppSession:
    session = getattr(request.app.state, "app_session", None)
    if session is None:
        raise HTTPException(status_code=500, detail="application not initialized")
    return session


def get_app_state(request: Request) -> AppState:
    session = get_app_session(request)
    if session.app_state is None:
        raise HTTPException(status_code=503, detail="config not bound")
    return session.app_state
```

Keep `build_app_state` as today. Update `create_app`:

```python
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
        # backward-compat alias used nowhere critical; prefer session
        app.state.app_state = app_session.app_state
        yield
        if app_session.app_state is not None:
            app_session.app_state.job_runner.shutdown()

    # ... middleware and routers as today ...
    # websocket: only when bound — create router with a store proxy OR
    # mount websocket only if bound; for unbound use a dummy store after bind swap.
```

**WebSocket note:** `create_websocket_router(hub, store)` currently needs a store at app creation. Change websocket factory to resolve store from `app.state.app_session.app_state` per connection; if unbound, close with code 1013 or send error then close.

Minimal approach for this task: pass `job_store` only when bound; when unbound, register WS route that rejects:

In `websocket.py`, look up session on connect; if `app_state is None`, `await websocket.close(code=1013)` and return.

Update websocket router construction to not require store at import time — take hub only, load store from session inside the endpoint.

- [ ] **Step 4: Run resolve tests + existing API smoke**

Run:

```bash
cd /workspace && .venv/bin/pytest \
  tests/dataset_builder/test_session_resolve.py \
  tests/dataset_builder/test_api_jobs.py \
  tests/dataset_builder/test_api_config.py \
  tests/dataset_builder/test_api_security.py -v
```

Expected: all PASS (bound `create_app(path, root)` still works)

- [ ] **Step 5: Commit**

```bash
git add cv_preprocess/web/session_resolve.py cv_preprocess/web/dependencies.py \
  cv_preprocess/web/app.py cv_preprocess/web/websocket.py cv_preprocess/jobs/store.py \
  tests/dataset_builder/test_session_resolve.py
git commit -m "feat: support unbound GUI AppSession and config resolve"
```

---

### Task 3: Session API (list / bind / create / unbind)

**Files:**
- Create: `cv_preprocess/web/routes/session.py`
- Modify: `cv_preprocess/web/routes/__init__.py` (export if needed)
- Modify: `cv_preprocess/web/app.py` (include router)
- Test: `tests/dataset_builder/test_session_api.py`

**Interfaces:**
- Consumes: `AppSession`, `build_app_state`, `write_last_config`, `to_project_relative`, `load_config` / `PipelineConfig`, `resolve_within_root`
- Produces HTTP:
  - `GET /api/session` → `{ bound: bool, config_path: str | null, project_root: str }`
  - `GET /api/session/configs` → `{ configs: [{ path: str, exists: bool }] }`
  - `POST /api/session/bind` body `{ path: str }` 
  - `POST /api/session/create` body `{ path?: str, overwrite?: bool }` (default path `config/default.yaml`)
  - `POST /api/session/unbind`

Bind/create helpers on session (put in `session.py` or `dependencies.py`):

```python
def bind_session(session: AppSession, config_path: Path) -> AppState:
    # shutdown previous runner if any
    # build_app_state, assign session.app_state, write_last_config
```

- [ ] **Step 1: Write failing API tests**

```python
# tests/dataset_builder/test_session_api.py
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cv_preprocess.web.app import create_app

MINIMAL = """
schema_version: 2
input:
  corpus_root: .
  clip_tsv: validated.tsv
dataset_builder:
  enabled: true
  work_dir: work
""".strip() + "\n"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "example.yaml").write_text(MINIMAL, encoding="utf-8")
    return tmp_path


def test_unbound_session_and_jobs_503(project: Path) -> None:
    with TestClient(create_app(None, project)) as client:
        s = client.get("/api/session")
        assert s.status_code == 200
        assert s.json()["bound"] is False
        assert client.get("/api/jobs").status_code == 503
        assert client.get("/api/dashboard").status_code == 503


def test_list_configs_includes_example(project: Path) -> None:
    with TestClient(create_app(None, project)) as client:
        resp = client.get("/api/session/configs")
        assert resp.status_code == 200
        paths = {c["path"] for c in resp.json()["configs"]}
        assert "config/example.yaml" in paths


def test_create_bind_unbind(project: Path) -> None:
    with TestClient(create_app(None, project)) as client:
        created = client.post(
            "/api/session/create",
            json={"path": "config/default.yaml", "overwrite": False},
        )
        assert created.status_code == 200
        assert created.json()["bound"] is True
        assert (project / "config" / "default.yaml").is_file()
        assert client.get("/api/dashboard").status_code == 200

        again = client.post(
            "/api/session/create",
            json={"path": "config/default.yaml", "overwrite": False},
        )
        assert again.status_code == 409

        unbound = client.post("/api/session/unbind")
        assert unbound.status_code == 200
        assert unbound.json()["bound"] is False
        assert client.get("/api/jobs").status_code == 503


def test_bind_invalid_yaml_400(project: Path) -> None:
    bad = project / "config" / "bad.yaml"
    bad.write_text("dataset_builder: []\n", encoding="utf-8")
    with TestClient(create_app(None, project)) as client:
        resp = client.post("/api/session/bind", json={"path": "config/bad.yaml"})
        assert resp.status_code == 400


def test_path_traversal_rejected(project: Path) -> None:
    with TestClient(create_app(None, project)) as client:
        resp = client.post("/api/session/bind", json={"path": "../outside.yaml"})
        assert resp.status_code in (400, 403)


def test_unbind_blocked_when_job_active(project: Path) -> None:
    cfg = project / "config" / "default.yaml"
    cfg.write_text(MINIMAL, encoding="utf-8")
    with TestClient(create_app(cfg, project)) as client:
        from unittest.mock import patch
        with patch("cv_preprocess.jobs.runner.JobRunner.start_job"):
            job = client.post("/api/jobs", json={"job_type": "scan", "force": False})
            assert job.status_code == 200
        # leave job queued
        resp = client.post("/api/session/unbind")
        assert resp.status_code == 409
```

- [ ] **Step 2: Run tests — expect fail (404 on /api/session)**

Run: `cd /workspace && .venv/bin/pytest tests/dataset_builder/test_session_api.py -v`

- [ ] **Step 3: Implement `session.py` and wire router**

Key implementation rules:

- List configs: glob `config/*.yaml` and `config/*.yml` under `project_root` (files only), return posix relative paths sorted.
- `bind`: `resolve_within_root(project_root, path)` → must be `.yaml`/`.yml` file → `load_config` / `build_app_state`; on `ValidationError` → 400 with error strings; shutdown previous `job_runner` if bound; set `session.app_state`; `write_last_config`.
- `create`: source `config/example.yaml` via `resolve_within_root`; if missing → 404; target default `config/default.yaml`; if exists and not overwrite → 409; `shutil.copyfile`; then bind.
- `unbind`: if bound and `has_active_jobs()` → 409; else shutdown runner, `session.app_state = None` (do not delete last_config file).

Mount: `app.include_router(session.router, prefix="/api/session", tags=["session"])`.

- [ ] **Step 4: Run session + prior API tests**

```bash
cd /workspace && .venv/bin/pytest \
  tests/dataset_builder/test_session_api.py \
  tests/dataset_builder/test_last_config.py \
  tests/dataset_builder/test_session_resolve.py \
  tests/dataset_builder/test_api_jobs.py \
  tests/dataset_builder/test_api_config.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cv_preprocess/web/routes/session.py cv_preprocess/web/app.py \
  cv_preprocess/web/routes/__init__.py tests/dataset_builder/test_session_api.py
git commit -m "feat: add GUI session bind/create/unbind API"
```

---

### Task 4: CLI optional `-c` + write last_config on explicit `-c`

**Files:**
- Modify: `cv_preprocess/cli.py` (`cmd_gui`, `_default_gui_project_root`)
- Optional small test via Typer CliRunner if already used; otherwise rely on `resolve_gui_config_path` tests + manual check note

**Interfaces:**
- Consumes: `resolve_gui_config_path`, `write_last_config`, `to_project_relative`, `create_app`
- Produces: `cv-preprocess gui` with `-c` optional

- [ ] **Step 1: Change `cmd_gui` signature and body**

```python
def _default_gui_project_root(config: Path | None) -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "frontend").is_dir() or (cwd / "pyproject.toml").is_file():
        return cwd
    if config is not None:
        for candidate in (config.resolve().parent, *config.resolve().parents):
            if (candidate / "frontend").is_dir() or (candidate / "pyproject.toml").is_file():
                return candidate
    return cwd


@app.command("gui")
def cmd_gui(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        path_type=Path,
        help="Pipeline YAML (optional; uses last config or setup screen)",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (default localhost only)"),
    port: int = typer.Option(8765, "--port", help="Bind port"),
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        path_type=Path,
        help="Project root for work/output/frontend (default: cwd or repo with frontend/)",
    ),
) -> None:
    """Start the dataset builder FastAPI GUI (serves frontend/dist when built)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "GUI dependencies missing; install with: uv sync --extra gui"
        ) from exc

    from cv_preprocess.web.app import create_app
    from cv_preprocess.web.last_config import to_project_relative, write_last_config
    from cv_preprocess.web.session_resolve import resolve_gui_config_path

    root = project_root.resolve() if project_root is not None else _default_gui_project_root(config)
    resolved = resolve_gui_config_path(project_root=root, cli_config=config)
    if config is not None and resolved is not None:
        write_last_config(root, to_project_relative(root, resolved))
    dist = root / "frontend" / "dist"
    if not dist.is_dir():
        typer.echo(
            f"Warning: frontend build not found at {dist}. "
            "Run: ./scripts/start-gui.sh   # or: cd frontend && pnpm install && pnpm build",
            err=True,
        )
    app = create_app(resolved, root)
    uvicorn.run(app, host=host, port=port, log_level="info")
```

- [ ] **Step 2: Sanity check help**

Run: `cd /workspace && .venv/bin/cv-preprocess gui --help`  
Expected: `--config` not marked required

- [ ] **Step 3: Commit**

```bash
git add cv_preprocess/cli.py
git commit -m "feat: make cv-preprocess gui -c optional with last_config"
```

---

### Task 5: Frontend Setup + session gate

**Files:**
- Modify: `frontend/src/api/types.ts`, `frontend/src/api/client.ts`
- Create: `frontend/src/pages/Setup.tsx`
- Create: `frontend/src/session/SessionGate.tsx` (or `hooks/useSession.ts` + gate in App)
- Modify: `frontend/src/App.tsx`, `frontend/src/components/Layout.tsx`

**Interfaces:**
- Consumes: `/api/session*` 
- Produces: Setup UI; main routes only when `bound`; Switch config calls unbind then shows Setup

- [ ] **Step 1: Add types + client methods**

```typescript
// types
export interface SessionState {
  bound: boolean;
  config_path: string | null;
  project_root: string;
}
export interface SessionConfigItem {
  path: string;
}
export interface SessionConfigsResponse {
  configs: SessionConfigItem[];
}

// client
session: () => request<SessionState>("/api/session"),
listSessionConfigs: () => request<SessionConfigsResponse>("/api/session/configs"),
bindSession: (path: string) =>
  request<SessionState>("/api/session/bind", { method: "POST", body: JSON.stringify({ path }) }),
createSession: (body: { path?: string; overwrite?: boolean }) =>
  request<SessionState>("/api/session/create", { method: "POST", body: JSON.stringify(body) }),
unbindSession: () =>
  request<SessionState>("/api/session/unbind", { method: "POST", body: "{}" }),
```

- [ ] **Step 2: Implement `Setup.tsx`**

UI requirements:

- Fetch configs on mount
- Select + “Use this config” → `bindSession`
- Form: target path default `config/default.yaml`, checkbox overwrite, button “Create from example”
- Show API errors (400/409/403)
- On success, call `onBound()` callback from parent

- [ ] **Step 3: Implement SessionGate in App**

```tsx
// Pseudocode structure
function App() {
  const [session, setSession] = useState<SessionState | null>(null);
  const [loading, setLoading] = useState(true);
  const refresh = async () => { setSession(await api.session()); };

  useEffect(() => { refresh().finally(() => setLoading(false)); }, []);

  if (loading) return <p>Loading…</p>;
  if (!session?.bound) {
    return <Setup onBound={refresh} />;
  }
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout configPath={session.config_path} onSwitchConfig={async () => {
          await api.unbindSession();
          await refresh();
        }} />}>
          ...existing routes...
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
```

Handle unbind 409: show error toast/banner “Finish or cancel jobs before switching”.

Add Layout footer/nav button “Switch config”.

- [ ] **Step 4: Typecheck / build**

```bash
cd /workspace/frontend && pnpm install && pnpm exec tsc -b --pretty false && pnpm build
```

Expected: exit 0; `frontend/dist` updated

- [ ] **Step 5: Commit**

```bash
git add frontend/src
git commit -m "feat: add GUI setup screen and session binding gate"
```

---

### Task 6: start-gui.sh + gitignore

**Files:**
- Create: `scripts/start-gui.sh` (executable)
- Modify: `.gitignore` — add `.filteringcv/`

- [ ] **Step 1: Write script**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d frontend/dist && "$SKIP_BUILD" -eq 0 ]]; then
  if ! command -v pnpm >/dev/null 2>&1; then
    echo "pnpm not found. Install pnpm, or run: cd frontend && npm install -g pnpm" >&2
    exit 1
  fi
  (cd frontend && pnpm install && pnpm build)
fi

if [[ -x .venv/bin/cv-preprocess ]]; then
  exec .venv/bin/cv-preprocess gui "$@"
fi
exec uv run cv-preprocess gui "$@"
```

Note: after parsing `--skip-build`, remaining args should forward to `gui` — adjust parsing so unknown options after `--` go to gui, or document that host/port are passed after `--`:

Prefer:

```bash
# usage: ./scripts/start-gui.sh [--skip-build] [-- --host 127.0.0.1 --port 8765]
```

Implement with a simple split on `--`.

- [ ] **Step 2: chmod +x and dry-run help**

```bash
chmod +x scripts/start-gui.sh
# Ensure .gitignore contains:
# .filteringcv/
```

- [ ] **Step 3: Commit**

```bash
git add scripts/start-gui.sh .gitignore
git commit -m "chore: add start-gui helper and ignore .filteringcv"
```

---

### Task 7: Docs — GUI recommended

**Files:**
- Modify: `README.md` (使い方 / GUI section)
- Modify: `docs/gui.md` (rewrite positioning)
- Modify: `docs/開発環境.md` (post-create flow → start-gui)
- Modify: `docs/architecture.md` (one line: GUI recommended for interactive ops)
- Modify: `docs/dataset-builder.md` (optional one-liner pointing to GUI)

- [ ] **Step 1: Rewrite `docs/gui.md`**

Lead with: GUI is the **recommended** interactive way to run the dataset builder. CLI remains for CI/headless/automation.

Document:

1. `uv sync --extra gui --extra sidon`
2. `./scripts/start-gui.sh`
3. Open `http://127.0.0.1:8765`
4. Setup screen: pick YAML or create from `example.yaml`
5. Last config in `.filteringcv/last_config.json`
6. Optional `cv-preprocess gui -c config/default.yaml`
7. Existing screens list + security notes

- [ ] **Step 2: Update README**

- Near top of 使い方: recommended interactive path → GUI link + `./scripts/start-gui.sh`
- Change “#### GUI（任意）” to “#### GUI（推奨・対話操作用）”
- Keep CLI tables; label builder CLI as automation / advanced

- [ ] **Step 3: Update 開発環境.md**

After Dev Container sync, recommend `./scripts/start-gui.sh` instead of only `cp` + CLI build. Keep `cp config/example.yaml config/default.yaml` as optional (also doable in Setup UI).

- [ ] **Step 4: Commit**

```bash
git add README.md docs/gui.md docs/開発環境.md docs/architecture.md docs/dataset-builder.md
git commit -m "docs: position GUI as recommended interactive entry"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run focused pytest suite**

```bash
cd /workspace && .venv/bin/pytest \
  tests/dataset_builder/test_last_config.py \
  tests/dataset_builder/test_session_resolve.py \
  tests/dataset_builder/test_session_api.py \
  tests/dataset_builder/test_api_jobs.py \
  tests/dataset_builder/test_api_config.py \
  tests/dataset_builder/test_api_security.py \
  tests/dataset_builder/test_overrides_api.py \
  tests/dataset_builder/test_audio_range.py -v
```

Expected: all PASS

- [ ] **Step 2: Frontend build**

```bash
cd /workspace/frontend && pnpm build
```

Expected: exit 0

- [ ] **Step 3: Manual smoke (if environment allows)**

```bash
cd /workspace && ./scripts/start-gui.sh --skip-build
# GET http://127.0.0.1:8765/api/session → bound true/false consistent with last_config
```

- [ ] **Step 4: Commit any fixups** (only if needed)

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| Docs GUI-first | 7 |
| start-gui.sh + auto pnpm build + `--skip-build` | 6 |
| `-c` optional | 4 |
| last_config project-local relative | 1, 4 |
| Setup list / create / bind | 3, 5 |
| Unbound 503 on main APIs | 2, 3 |
| Switch config + 409 when jobs active | 3, 5 |
| No force-switch / no legacy GUI port | honored (non-goals) |
| Security path rules | 3 tests |

## Placeholder / consistency notes

- `create_app(None, project_root)` is the unbound entry; bound tests keep `create_app(path, root)`.
- `get_app_state` always 503 when unbound (detail exact string `config not bound`).
- Session JSON field `config_path` is project-relative when bound, else `null`.
