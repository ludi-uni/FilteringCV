# GUI (recommended for interactive use)

The FilteringCV GUI is the **recommended** way to run the dataset builder interactively. It is a FastAPI backend with a Vite + React + TypeScript frontend that calls the same `cv_preprocess.application` Core API as the CLI.

Use the **CLI** for CI, headless servers, scripting, and automation. Use the **GUI** when you want to configure, inspect, and run builder stages with live progress.

## Install

```bash
uv sync --extra sidon --extra gui --extra dev
```

This installs FastAPI, uvicorn, websockets, and python-multipart. Add other extras (`dasheng`, `sgmse`, `hifigan`, `wpe_dfn`, etc.) if your YAML requires them — same as for CLI.

## Start (recommended)

```bash
./scripts/start-gui.sh
```

The helper builds `frontend/` with `pnpm` when `frontend/dist` is missing (use `--skip-build` to skip), then runs `cv-preprocess gui`.

### Open in the browser (devcontainer / Cursor Remote)

The GUI listens inside the container. Windows/macOS browsers do **not** reach it until port **8765** is forwarded:

1. Start: `./scripts/start-gui.sh`
2. Cursor / VS Code → **Ports** panel (パネル「ポート」)
3. **Forward a Port** → `8765`（既に出ていればその行の Local Address を開く）
4. Open the forwarded URL (usually **http://127.0.0.1:8765/**)

`.devcontainer/devcontainer.json` sets `forwardPorts: [8765]` so rebuild/reopen should auto-forward.

In Docker, `./scripts/start-gui.sh` binds **`0.0.0.0`** by default. Override:

```bash
./scripts/start-gui.sh -- --host 0.0.0.0 --port 8765
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
- **Jobs** — pipeline stages in order (`scan` → `analyze` → `plan-split` → `select` → `materialize` → `audit`), plus **`build`** (recommended one-shot). When `coverage.enabled` is true, a second section appears for rare-phoneme automation (`coverage-index` → … → **`coverage-build`**). Each row shows what the stage does and what it produces; start with **Build** for a first corpus run
- **Config** — edit the loaded YAML in Form or YAML mode; **Save & overwrite YAML** writes back to the original config path after Pydantic validation. Search and chips (Filters / Builder / Audio / Gates / Changed) help focus speaker filters and related settings. Builder chip includes **Coverage automation**
- **Coverage** — catalog feature-pool stats, plus active coverage-automation run summary when configured
- **Clips** — paginated catalog browser with audio preview
- **Run Comparison** — diff two `work/` or output directories

Use **Switch config** in the layout to unbind and return to Setup (blocked while jobs are active).

### Jobs order (quick reference)

1. `scan` — corpus / TSV sanity check
2. *(optional, when `coverage.enabled`)* `coverage-index` → `coverage-run` — selective quality analyze **before** full analyze
3. `analyze` — catalog + audio cache (reuses clips already analyzed by coverage)
4. `plan-split` — train/val/test plan (semantics depend on protocol; see below)
5. `select` — **coverage-aware（既定オン）**: `coverage.features` の必須目標を予約してから残りを貪欲選択。監査は `work/reports/selection/`
6. `materialize` — write final WAV/metadata; by default also `exports/piper_plus` and `exports/style_bert_vits2`
7. `audit` — integrity checks
★ `build` — runs the above with resume (inserts coverage before analyze when enabled)

### Coverage automation (optional Force Build)

Enable in Config (`coverage.enabled: true`, `insert_before_analyze: true` by default):

- **Build** automatically runs coverage **before** analyze so you do not wait on a full heavy pass first.
- Or run `coverage-build` alone, then `analyze` / continue Build.

**Final-set guarantee** is separate and **always on** by default: `selection.coverage_constraints.enabled: true` makes `select` reserve clips for those same targets (even if you re-run select alone). See [coverage-automation.md](coverage-automation.md#coverage-aware-select).

### Coverage-aware select (what to do)

1. Set targets under `coverage.features` (e.g. `phoneme.targets.v: 5`).
2. Run **Build** (or coverage-run → analyze → select).
3. Open `work/reports/selection/coverage-audit.csv` — statuses like `corpus_limit_satisfied` mean the corpus had fewer eligible clips than the configured target.
4. To disable: set `selection.coverage_constraints.enabled: false` (and optionally `acoustic_diversity.enabled: false`).

See [coverage-automation.md](coverage-automation.md).

### Why `plan-split` before `select`?

Stage order is always `plan-split` → `select`, but **what that means depends on `dataset_builder.split.protocol`:**

| Protocol | Effective flow | Why |
|----------|----------------|-----|
| **`unseen_speaker`** (common default) | Assign **speakers** to train/val/test first, then **select within each bucket** | Keeps the same speaker out of train and val/test. Selecting the whole pool first, then splitting speakers, often wrecks per-split coverage. |
| **`seen_speaker` / `single_speaker`** | **Select globally**, then attach split labels to selected clips | Speaker leakage across splits is allowed (or single-speaker). Clip assignment after select matches the “select then split” intuition. |

So “shouldn’t we select then split?” is right for the latter protocols; for **`unseen_speaker` the current order is intentional**. See [dataset-builder.md](dataset-builder.md).

## Security defaults

- No arbitrary filesystem reads outside configured roots
- Path traversal rejected
- Parameterized SQL for job store
- No `shell=True` in subprocess workers

The GUI does not shell out to `cv-preprocess` subcommands.

## Job store on WSL / Docker Desktop

`work/jobs.sqlite3` lives under `dataset_builder.work_dir`. When that path is on a **Windows bind mount** (9p / DrvFs — common for `/workspace` in Dev Containers), SQLite **WAL** mode is unsafe and can raise `OperationalError: locking protocol` during Build progress updates.

FilteringCV auto-detects these filesystems and uses rollback journal (`DELETE`) instead, with retries. Progress also falls back to JSONL if a write still fails, so analyze/build should not abort solely on a progress DB glitch.

If an old `jobs.sqlite3-wal` / `-shm` pair remains after a crash, delete those sidecars (or re-open the GUI once so the store migrates) before retrying.

## Related docs

- [dataset-builder.md](dataset-builder.md) — builder stages and CLI reference
- [開発環境.md](開発環境.md) — Dev Container and optional extras
