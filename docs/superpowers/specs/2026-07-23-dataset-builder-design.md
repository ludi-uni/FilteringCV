# FilteringCV Next-Gen Dataset Builder — Design Spec

**Date:** 2026-07-23  
**Status:** Approved  
**Config approach:** A — extend `PipelineConfig`  
**Analyze approach:** 1 — Adapter + disposition fork

## Goal

Re-architect FilteringCV from sequential quality-gate-and-accept preprocessing into a local dataset builder that constructs a corpus under quality, linguistic coverage, speaker, duplicate, split-leakage, and reproducibility constraints — while reusing existing audio/NFA/MFA/ASR/OpenJTalk code and keeping a Python backend (no Rust/Tauri in this delivery).

## Non-goals (this delivery)

- Rust / PyO3 / Tauri implementation (boundary + benchmarks only)
- OR-Tools as a hard dependency (optional extra only)
- Deleting legacy `preprocess` or `secondary`
- Silently changing legacy output formats when `dataset_builder.enabled` is false

## Architecture

```text
React GUI  ──HTTP/WS──►  FastAPI (optional extra)
                              │
CLI (typer)  ─────────────────┼──►  application/* Core API
                              │
                              ▼
                    catalog / selection / split / linguistic /
                    compute / reports / jobs
                              │
                              ▼
              existing pipeline/* + audio/* + text/* (adapters)
```

CLI and GUI both call Core API only. GUI must not parse CLI stdout; CLI must not call GUI routes.

## Config (Approach A)

Extend existing `PipelineConfig`:

- `schema_version: int` — missing → treat as 1; migrate toward 2 with warnings
- `dataset_builder: DatasetBuilderConfig` — `enabled` default `false`
- Nested blocks: `selection`, `speaker_constraints`, `duplicates`, `distribution_temperature`, `feature_support`, `materialize`, `compute`
- Extend `split` for protocols `unseen_speaker` | `seen_speaker` | `single_speaker`, ratios, preserve_train, leakage_policy, optimizer

Compatibility:

- `dataset_builder.enabled: false` → current `preprocess` behavior unchanged
- `enabled: true` → `build` Core API; `preprocess` may warn and delegate
- YAML ↔ Pydantic round-trip; no GUI-only hidden settings

`AppConfig` in API signatures means the loaded `PipelineConfig` (same object).

## Disposition model

```python
class ClipDisposition(str, Enum):
    HARD_REJECTED = "hard_rejected"
    ELIGIBLE = "eligible"
    SELECTED = "selected"
    RESERVE = "reserve"
```

- **hard_rejected:** decode failure, fatal quality, align/ASR hard fail — not trainable
- **eligible:** usable candidate after analyze (speaker caps do NOT apply here)
- **selected / reserve:** assigned only by select (not by input order)

`max_clips_per_speaker` / duration caps are selection constraints, never quality reject reasons in the builder path.

## Core API

```python
scan_project(config, *, progress=None) -> ScanResult
analyze_project(config, *, progress=None, cancellation=None) -> AnalyzeResult
plan_dataset_split(config, catalog, *, progress=None) -> SplitPlan
select_dataset(config, catalog, split_plan, *, backend=None, progress=None) -> SelectionPlan
materialize_dataset(config, catalog, selection_plan, *, progress=None) -> MaterializeResult
audit_dataset(config, catalog, selection_plan) -> AuditReport
build_dataset(...)  # orchestrates stages with stage-level resume
```

Progress via `ProgressSink`; cancel via `CancellationToken`. No CLI-specific code inside Core API.

## Analyze adapter (Approach 1)

Reuse `PreprocessSession` decode → gates → enhance path.

Fork at final acceptance:

1. Run existing quality / MFA / NFA / ASR gates unchanged
2. On hard failure → catalog row `HARD_REJECTED` + reason (no final WAV)
3. On pass → write processed WAV to content-addressed `work/audio_cache/<pipeline_hash>/...` and mark `ELIGIBLE`
4. Do **not** apply speaker max / first-come adoption
5. Do **not** write final `output/wavs` during analyze

Legacy `preprocess` keeps current accept path when builder disabled.

## Stable IDs & catalog

```text
clip_id = sha256(
  source_release + normalized_relative_source_path
  + str(source_row_index) + audio_sha256 + text_raw
).hexdigest()
```

Work tree:

```text
work/
  catalog/{clips,feature_counts,speaker_stats,duplicate_groups}.parquet
  catalog/manifest.json
  audio_cache/<pipeline_hash>/<prefix>/<audio-sha256>.wav
  plans/{split_plan.json,selection_plan.parquet}
  overrides.jsonl
  reports/
  jobs.sqlite3
```

Polars is the primary read/write/aggregate tool. List or JSON-string columns are schema-fixed.

## Linguistic features

Extract: phone, biphone, triphone, mora, mora-bigram, full-context labels, accent-nucleus-related, accent-phrase length, pause/phrase boundary, sentence-length band, speaking-rate band, interrogative/declarative.

Feature source priority: `aligned` > `asr_inferred` > `text_g2p`.  
Full-context unavailable → warn and continue with phone/mora.  
`sil`/`pau`/word-boundary tokens down-weighted or excluded via config.

## Duplicates

Group (do not silently drop): `exact_audio`, `same_source_path`, `same_sentence_id`, `same_normalized_text`, `same_speaker_same_text`, optional `near_duplicate_text`.

Selection applies configurable max_selected / penalties.

## Selection

Deterministic greedy + optional local search (`selected↔reserve` swaps 1↔1, 1↔2, 2↔1) with iteration and wall-time limits.

Target distribution: `P_target(f) ∝ P_pool(f)^alpha` (temperature per feature family).  
No forced equal counts. Feature support thresholds exclude ultra-rare G2P noise from strong rescue weights.  
Diminishing returns: `utility_f(count) = weight_f * log(1 + count / tau_f)`.  
Speaker constraints prefer **duration** over clip count.

Overrides in `work/overrides.jsonl` only; re-select without re-analyze.

## Split protocols

| Protocol | Order |
|----------|-------|
| `unseen_speaker` | analyze → speaker split plan → select within splits |
| `seen_speaker` / `single_speaker` | analyze → select → leakage-aware split |

Train preservation for critical low-support features. Leakage policy for speaker / audio_hash / sentence_id / normalized_text.

Optimizer: `auto` with greedy_local_search fallback; OR-Tools optional.

## Materialize & atomicity

Modes: `copy` | `hardlink` | `symlink` (fallback to copy).  
Only `SELECTED` clips. Partial writes use `.partial` then atomic rename.  
Outputs: wavs, metadata/train/validation/test jsonl, reports, plans, run_manifest.

## Compute backend boundary

```python
class ComputeBackend(Protocol):
    def count_features(...) -> FeatureCounts: ...
    def build_duplicate_groups(...) -> DuplicateGroups: ...
    def score_candidates(...) -> CandidateScores: ...
    def update_selection_state(...) -> SelectionState: ...
```

Implementations: `PolarsComputeBackend` (default for `auto`), `PythonComputeBackend` (fallback).  
Reserve name `NativeComputeBackend` in docs only — no stub crate.

Run manifest records wall/CPU/RSS, counts, backend, cache hits/misses.  
CLI: `cv-preprocess benchmark-selection`.

## Jobs & GUI

- SQLite WAL job store + subprocess worker (not in-process heavy work)
- Status: queued/running/cancelling/cancelled/succeeded/failed/interrupted
- Progress → SQLite + JSONL + WebSocket (`ProgressEvent`)
- Cancel: token + process-group kill; partial artifacts
- FastAPI serves built static frontend; default bind `127.0.0.1:8765`
- Screens: Dashboard, Jobs, Coverage, Clips (paginated), Run Comparison
- Security: no arbitrary path read, path traversal reject, parameterized SQL, no `shell=True`

## CLI additions

`scan`, `analyze`, `plan-split`, `select`, `materialize`, `audit`, `build`, `gui`, `compare-runs`, `benchmark-selection`  
Keep existing `preprocess`, `secondary`, etc.

## Package layout (new)

```text
cv_preprocess/
  application/   catalog/   selection/   linguistic/
  split/         reports/   compute/     jobs/   web/
frontend/        # Vite React TS, pnpm
```

Existing `pipeline/` remains; builder calls it via adapters.

## Dependencies

- Core: add `polars` (and pyarrow if needed for parquet)
- Extra `gui`: fastapi, uvicorn, websockets, …
- Extra `optimizer`: ortools (optional)
- Do not break CUDA/NFA/SGMSE/protobuf pins; update `uv.lock`

## Testing

Unit: n-grams, mora, stable id, config, target dist, diminishing returns, duplicates, leakage, JS distance, reports, overrides, cache key.  
Order/property: shuffle TSV/candidates → same result; seed stability; hard_reject never selected; speaker/duration caps; unseen speaker isolation; train preservation; reserve≠reject.  
Comparison: greedy coverage ≥ random / > head selection on synthetic catalog.  
API: routes, jobs, cancel, audio range, path traversal, localhost default, overrides, compare.  
Fixtures/mocks only — no requirement for full Common Voice or external models.

## Completion criteria

Match the 20 items in the implementation brief (disposition split, order independence, parquet catalog, re-select without re-analyze, coverage-aware select, temperature targets, reject/reserve reports, three split protocols, leakage checks, shared Core API, GUI jobs/audio/overrides/compare, Polars+Python compute backends, Rust boundary without Rust code, legacy pipelines intact, ruff+pytest green, docs match reality).

## Decisions locked

1. Config approach **A**
2. Analyze approach **1** (adapter + disposition fork)
3. No Rust implementation this delivery
4. GUI is optional extra; Core API works headless
