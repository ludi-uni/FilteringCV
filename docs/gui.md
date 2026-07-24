# GUI (recommended for interactive use)

The FilteringCV GUI is the **recommended** way to run the dataset builder interactively. It is a FastAPI backend with a Vite + React + TypeScript frontend that calls the same `cv_preprocess.application` Core API as the CLI.

Use the **CLI** for CI, headless servers, scripting, and automation. Use the **GUI** when you want to configure, inspect, and run builder stages with live progress.

## Install

```bash
uv sync --extra gui --extra sidon
```

This installs FastAPI, uvicorn, websockets, and python-multipart. Add other extras (`dasheng`, `sgmse`, `hifigan`, `wpe_dfn`, etc.) if your YAML requires them — same as for CLI.

## Start (recommended)

```bash
./scripts/start-gui.sh
```

The helper builds `frontend/` with `pnpm` when `frontend/dist` is missing (use `--skip-build` to skip), then runs `cv-preprocess gui`. Open **http://127.0.0.1:8765** in your browser (default bind: localhost only).

Pass extra uvicorn/CLI flags after `--`, for example:

```bash
./scripts/start-gui.sh -- --host 127.0.0.1 --port 8765
```

## Setup screen

On first launch (or after **Switch config**), the GUI shows a **Setup** screen until a pipeline YAML is bound:

1. **Existing configs** — pick a `.yaml` / `.yml` under `config/` and click **Use this config**.
2. **Create from example** — copy [`config/example.yaml`](../config/example.yaml) to a new path (default `config/default.yaml`). Enable **overwrite** if the file already exists.

You do not need to run `cp config/example.yaml config/default.yaml` manually unless you prefer editing YAML on disk before opening the GUI.

## Last config memory

The last bound config path is stored in **`.filteringcv/last_config.json`** (project-local, gitignored). On the next start, `cv-preprocess gui` without `-c` reuses that file when it still exists.

To pin a config from the CLI (also updates last config):

```bash
cv-preprocess gui -c config/default.yaml
```

## Manual frontend build

When not using `start-gui.sh`, or after frontend changes:

```bash
cd frontend
pnpm install
pnpm build
cv-preprocess gui
```

## Screens

- **Setup** — bind an existing YAML or create from `example.yaml` (shown until a config is bound)
- **Dashboard** — recent runs and stage status
- **Jobs** — queued/running/completed jobs with cancel and live progress
- **Config** — edit the loaded YAML in Form or YAML mode; **Save & overwrite YAML** writes back to the original config path after Pydantic validation. Search and chips (Filters / Builder / Audio / Gates / Changed) help focus speaker filters and related settings
- **Coverage** — feature distribution vs pool target
- **Clips** — paginated catalog browser with audio preview
- **Run Comparison** — diff two `work/` or output directories

Use **Switch config** in the layout to unbind and return to Setup (blocked while jobs are active).

## Security defaults

- No arbitrary filesystem reads outside configured roots
- Path traversal rejected
- Parameterized SQL for job store
- No `shell=True` in subprocess workers

The GUI does not shell out to `cv-preprocess` subcommands.

## Related docs

- [dataset-builder.md](dataset-builder.md) — builder stages and CLI reference
- [開発環境.md](開発環境.md) — Dev Container and optional extras
