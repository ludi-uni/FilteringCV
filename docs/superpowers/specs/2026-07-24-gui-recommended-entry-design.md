# GUI Recommended Entry — Design Spec

**Date:** 2026-07-24  
**Status:** Approved (conversation)  
**Scope:** Positioning GUI as the recommended way to operate FilteringCV, plus launch/setup and in-GUI config select/create (minimal project switching).

## Goal

Make the FastAPI + React GUI the **recommended** operator path for day-to-day dataset builder work, by:

1. Rewriting docs so the default story starts with GUI (CLI remains for CI / headless / advanced use).
2. Adding a thin launch helper that reduces frontend-build + start friction.
3. Allowing `cv-preprocess gui` without `-c`, with last-used config memory and a setup screen to pick or create a YAML.

## Decisions locked

| Topic | Choice |
|-------|--------|
| Overall gap priority | Start with launch + docs positioning (not full P0 feature parity) |
| Depth | Docs + start helper + GUI config select/create |
| Config binding UX | `-c` optional; remember last config; else setup screen |
| Last-config storage | Project-local `.filteringcv/last_config.json` (gitignored) |
| Architecture | Staged startup: setup mode → bind → existing app screens (Approach 1) |

## Non-goals

- GUI for legacy `preprocess`, `secondary`, phoneme-manifest / MFA-NFA map suggest, `dataset-partition`, `benchmark-selection`, text debug commands
- Rejection report screen and other P0 builder ops gaps (follow-up)
- Home-directory or machine-global last-config store
- Multi-user auth
- Force-switch config while jobs are running

## Startup flow

```text
cv-preprocess gui [--config/-c optional] [--host] [--port] [--project-root]
        │
        ├─ -c given → load YAML (fail → exit)
        │              → write last_config → bound (main UI)
        │
        └─ -c omitted
              ├─ .filteringcv/last_config.json valid → load → bound
              └─ missing / invalid / unloadable → unbound (setup UI)
```

### Setup (unbound) mode

- List candidate YAML files under the project (primarily `config/*.yaml`).
- Create a new config by copying `config/example.yaml` (default destination `config/default.yaml`).
- On success: bind config, update `last_config.json`, enter main UI.

### Main (bound) mode

- Existing screens: Dashboard, Jobs, Config, Coverage, Clips, Compare.
- Nav action “Switch config” returns to setup flow (subject to job-running rules below).

## Local state file

- Path: `.filteringcv/last_config.json` at project root.
- Example:

```json
{
  "config_path": "config/default.yaml",
  "updated_at": "2026-07-24T02:00:00+00:00"
}
```

- `config_path` is **relative to `project_root`**. Absolute paths are not stored.
- Add `.filteringcv/` to `.gitignore`.

## Launch helper

- Add a thin script (e.g. `scripts/start-gui.sh`) that:
  1. If `frontend/dist` is missing, run `pnpm install && pnpm build` in `frontend/` (if `pnpm` is missing, exit with a clear install message). Optional `--skip-build` skips this step.
  2. Starts `cv-preprocess gui` from the repo root (no `-c` required).
- Document this as the recommended start path in README, `docs/gui.md`, and `docs/開発環境.md`.
- Position CLI builder commands as advanced / CI / headless.

## Session API

| Method | Purpose |
|--------|---------|
| `GET /api/session` | Bound flag, relative config path (if any), project root summary |
| `GET /api/session/configs` | Candidate YAML list (paths under `project_root` only) |
| `POST /api/session/bind` | Bind existing YAML; update `last_config` |
| `POST /api/session/create` | Copy from `example.yaml` to target path; bind |
| `POST /api/session/unbind` | Clear binding for switch flow |

### Unbound behavior

- Session endpoints above remain available.
- Main-app routes (jobs, catalog, reports, audio, overrides, compare, dashboard, config edit/save) return **503** with `config not bound`.

### Create semantics

- Default source: `config/example.yaml`.
- Default target: `config/default.yaml`.
- If target exists → **409** unless an explicit overwrite flag is set.

## Config switch while jobs run

- Before `unbind` or re-`bind` to another file: if any job is `queued` / `running` / `cancelling` → **409** (point user to Jobs).
- No force-switch flag in this delivery.
- On successful switch: shut down previous `JobRunner`, rebuild `AppState` for the new config (job DB under the new config’s `work_dir`).

## Error handling

| Situation | Behavior |
|-----------|----------|
| Stale / broken `last_config` | Log warning; open setup UI; do not crash |
| Bind YAML fails Pydantic validation | **400** + error list; do not bind |
| Create target exists | **409** (unless overwrite) |
| Unbound + main API | **503** `config not bound` |
| Path outside `project_root` / traversal | **403** / **400** (existing security rules) |

## Frontend

- Unbound: only Setup route (list / select / create).
- Bound: existing routes; add “Switch config” in shell nav.
- After bind, refresh client state from `GET /api/session` (and existing dashboard/config calls).

## Security (unchanged principles)

- No arbitrary filesystem reads outside configured / project roots.
- Reject path traversal.
- Parameterized SQL for job store.
- No `shell=True` in workers.
- Config enumerate / create / bind limited to YAML under `project_root`.

## Testing (minimum)

- Read/write `last_config` and relative-path normalization.
- Startup resolution: `-c` present; no `-c` + valid last; no `-c` + invalid last → setup.
- `bind` / `create` / `unbind` success; validation failure; 409 when jobs active; 409 on create-exists.
- Main APIs return 503 when unbound.
- Path traversal rejected on session endpoints.

## Docs changes

- README: recommended flow = install extras → start GUI helper → setup screen / last config.
- `docs/gui.md`: no longer “optional headless-first”; GUI is recommended for interactive use; CLI noted for automation.
- `docs/開発環境.md`: align Dev Container / local steps with GUI-first entry.
- Keep architecture note that Core API is shared; headless CLI remains fully supported.

## Implementation sketch (non-binding)

Likely touch points:

- `cv_preprocess/cli.py` — make `-c` optional for `gui`
- `cv_preprocess/web/app.py` / `dependencies.py` — support unbound → bound `AppState` swap
- New `cv_preprocess/web/routes/session.py` (+ last_config helper module)
- `frontend/` — Setup page, session client, nav switch, route gate
- `scripts/start-gui.sh`, `.gitignore`, docs listed above

## Success criteria

- A new contributor can follow README, run the start helper, pick or create a config in the browser, and reach the existing dashboard without manually inventing `-c` every time.
- Reopening the GUI in the same repo restores the last config when the file still exists.
- Switching config is blocked while jobs are active; otherwise works without restarting the process from the shell.
- Docs consistently describe GUI as the recommended interactive path.
