# Dataset Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship FilteringCV dataset builder (Core API + Parquet catalog + selection/split + materialize + FastAPI/React GUI) without breaking legacy preprocess.

**Architecture:** Extend `PipelineConfig` (`schema_version` + `dataset_builder`); analyze via PreprocessSession adapter that emits ELIGIBLE/HARD_REJECTED + audio_cache; select/split/materialize/audit as pure Core API; GUI/CLI share Core API; ComputeBackend protocol with Polars default and Python fallback.

**Tech Stack:** Python 3.10+, Pydantic v2, Polars, Typer, FastAPI/uvicorn (gui extra), React+Vite+TS+pnpm, SQLite WAL, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-07-23-dataset-builder-design.md`

## Global Constraints

- Reuse existing decode/resample/denoise/NFA/MFA/ASR/OpenJTalk; do not rewrite them
- `dataset_builder.enabled: false` preserves legacy preprocess outputs
- No Rust/Tauri code or empty crates
- Default GUI bind `127.0.0.1:8765`; no shell=True; parameterized SQL
- Speaker caps are selection constraints, never quality rejects in builder path
- Same inputs+config+seed → same results; no input-order adoption
- Update `uv.lock` after dependency changes
- Each phase ends with green `pytest` (targeted then broader) and `ruff check`

## File map (new)

```text
cv_preprocess/
  application/{__init__,common,scan,analyze,split,select,materialize,audit,build}.py
  catalog/{__init__,models,schema,reader,writer,feature_index,ids,cache}.py
  selection/{__init__,protocol,python_backend,scoring,constraints,local_search,overrides}.py
  linguistic/{__init__,features,ngrams,mora,fullcontext}.py
  split/{__init__,protocol,unseen_speaker,seen_speaker,single_speaker,leakage}.py
  reports/{__init__,models,coverage,quality,rejection,comparison,serializer}.py
  compute/{__init__,protocol,python_backend,polars_backend,loader,profiling}.py
  jobs/{__init__,models,store,runner,worker,progress}.py
  web/{__init__,app,dependencies,websocket,routes/*.py}
  config/dataset_builder.py   # new pydantic blocks
frontend/                     # Vite React app
docs/{architecture,dataset-builder,gui,catalog-schema,selection-algorithm,migration-v1-v2,rust-boundary}.md
tests/dataset_builder/        # new test package
```

## Phase 1 — Foundation

### Task 1: Disposition + stable clip_id + schema stubs

**Files:**
- Create: `cv_preprocess/catalog/models.py`, `cv_preprocess/catalog/ids.py`, `cv_preprocess/catalog/schema.py`, `cv_preprocess/catalog/__init__.py`
- Create: `tests/dataset_builder/test_stable_clip_id.py`
- Modify: `cv_preprocess/config/pipeline.py` (add `schema_version`, stub `dataset_builder`)
- Create: `cv_preprocess/config/dataset_builder.py`

**Interfaces:**
- Produces: `ClipDisposition`, `stable_clip_id(...)`, `CLIPS_SCHEMA` column list, `DatasetBuilderConfig(enabled: bool = False, work_dir: Path = Path("work"), ...)`

- [ ] **Step 1:** Add failing tests for `stable_clip_id` invariance and disposition enum values
- [ ] **Step 2:** Implement models/ids/schema + minimal `DatasetBuilderConfig` on `PipelineConfig`
- [ ] **Step 3:** `uv run pytest tests/dataset_builder/test_stable_clip_id.py -q` PASS
- [ ] **Step 4:** Commit

### Task 2: Progress/cancel + Core API stubs + polars dep

**Files:**
- Create: `cv_preprocess/application/common.py` (`ProgressSink`, `CancellationToken`, result types)
- Create: `cv_preprocess/application/{scan,analyze,split,select,materialize,audit,build}.py`
- Modify: `pyproject.toml` (polars core; gui/optimizer extras placeholders), run `uv lock`
- Create: `tests/dataset_builder/test_config_validation.py`

- [ ] Implement stubs that raise `NotImplementedError` only where unfinished; `scan_project` can wrap existing `pipeline.scan`
- [ ] Config validation: split ratios sum≈1, non-negative weights/durations
- [ ] Tests + commit

### Task 3: Catalog writer/reader + analyze disposition fork

**Files:**
- Create: `cv_preprocess/catalog/{writer,reader,cache}.py`
- Modify: `cv_preprocess/pipeline/preprocess/clip_accept.py` and session path for builder mode
- Create: `cv_preprocess/application/analyze.py` (real)
- Create: `tests/dataset_builder/test_cache_key.py`, `tests/dataset_builder/test_disposition_separation.py`

**Behavior:**
- When `dataset_builder.enabled`: quality pass → cache WAV + ELIGIBLE row; fail → HARD_REJECTED; skip speaker cap
- Legacy path unchanged when disabled

- [ ] Tests with synthetic waveforms/mocks (no NFA required)
- [ ] Commit

### Task 4: CLI wiring Phase 1

**Files:**
- Modify: `cv_preprocess/cli.py` — add `analyze`, `build` (build may call stages), keep `preprocess`
- Tests: CLI help smoke; preprocess still works with enabled=false

## Phase 2 — Linguistic + catalog aggregates

### Task 5: ngrams, mora, fullcontext, features

**Files:** `cv_preprocess/linguistic/*`, tests for ngrams/mora/JS later shared

### Task 6: feature_counts, speaker_stats, duplicate_groups, coverage/rejection report models

**Files:** catalog writer extensions, `cv_preprocess/reports/*`, tests

## Phase 3 — Selection

### Task 7: scoring (target dist, diminishing returns), constraints, greedy select

### Task 8: local_search, overrides.jsonl, selection explanations + selection_report

## Phase 4 — Split

### Task 9: unseen/seen/single protocols + leakage + train preservation + split_report

## Phase 5 — Materialize + audit + resume

### Task 10: materialize modes, atomic rename, run_manifest profiling fields, audit, stage resume in build

## Phase 6 — Jobs + FastAPI + React GUI

### Task 11: jobs SQLite/store/runner/worker/progress

### Task 12: FastAPI routes + websocket + security tests

### Task 13: frontend Dashboard/Jobs/Coverage/Clips/Comparison; `pnpm build`; static mount; `cv-preprocess gui`

## Phase 7 — ComputeBackend + docs + polish

### Task 14: ComputeBackend protocol, Polars + Python backends, loader auto, benchmark-selection CLI

### Task 15: docs listed in spec; README update; migration notes; full `ruff` + `pytest`; final report

## Execution note

Prefer implementing Phase 1–5 Core API + tests first (headless complete), then GUI. Use synthetic fixtures throughout. After each task: run focused tests; after each phase: broader suite + ruff.
