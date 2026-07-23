# GUI (optional extra)

The FilteringCV GUI is a FastAPI backend with a Vite + React + TypeScript frontend. It is **optional** — the dataset builder Core API works fully headless via CLI.

## Install

```bash
uv sync --extra gui
```

This installs FastAPI, uvicorn, websockets, and python-multipart.

## Frontend build

When the `frontend/` package is present in the repository:

```bash
cd frontend
pnpm install
pnpm build
```

The built static assets are served by the FastAPI app. Rebuild after frontend changes.

## Run

```bash
cv-preprocess gui -c config/my_builder.yaml
```

Default bind: `127.0.0.1:8765` (localhost only).

## Screens

- **Dashboard** — recent runs and stage status
- **Jobs** — queued/running/completed jobs with cancel and live progress
- **Config** — edit the loaded YAML in Form or YAML mode; **Save & overwrite YAML** writes back to the original config path after Pydantic validation. Search and chips (Filters / Builder / Audio / Gates / Changed) help focus speaker filters and related settings
- **Coverage** — feature distribution vs pool target
- **Clips** — paginated catalog browser with audio preview
- **Run Comparison** — diff two `work/` or output directories

## Security defaults

- No arbitrary filesystem reads outside configured roots
- Path traversal rejected
- Parameterized SQL for job store
- No `shell=True` in subprocess workers

The GUI calls the same `cv_preprocess.application` Core API as the CLI; it does not shell out to `cv-preprocess` subcommands.
