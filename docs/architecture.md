# FilteringCV Architecture

FilteringCV (`cv-preprocess`) is a Common Voice preprocessing and dataset-building toolkit. The codebase separates legacy sequential preprocessing from the newer dataset builder while sharing audio, text, and gate infrastructure.

## Layers

```text
CLI (Typer) / GUI (FastAPI + React — recommended for interactive ops)
        │
        ▼
application/*  — Core API (scan, analyze, select, materialize, audit, build)
        │
        ├── catalog/     Parquet catalog read/write, stable clip IDs
        ├── selection/   Greedy coverage selection + local search
        ├── split/       Train/val/test protocols and leakage checks
        ├── linguistic/  N-grams, mora, full-context feature extraction
        ├── compute/     ComputeBackend boundary (Polars default, Python fallback)
        ├── reports/     Coverage, rejection, run comparison
        └── jobs/        SQLite job store (GUI extra)
        │
        ▼
pipeline/* + audio/* + text/*  — Existing decode, gates, MFA/NFA, G2P
```

CLI and GUI both call **Core API** functions in `cv_preprocess.application`. Neither parses the other's output format. For day-to-day interactive builder work, prefer the GUI ([gui.md](gui.md)); use the CLI for automation and CI.

## Disposition model

| Disposition | Set by | Meaning |
|-------------|--------|---------|
| `hard_rejected` | analyze | Fatal quality/decode/text failure |
| `eligible` | analyze | Usable candidate in catalog |
| `selected` | select | Chosen for export |
| `reserve` | select | Ranked backup pool |

Speaker caps and duplicate limits apply at **selection**, not as quality rejects during analyze.

## Work tree

```text
work/
  catalog/clips.parquet
  catalog/feature_counts.parquet
  catalog/speaker_stats.parquet
  catalog/duplicate_groups.parquet
  catalog/manifest.json
  audio_cache/<pipeline_hash>/...
  plans/split_plan.json
  plans/selection_plan.parquet
  overrides.jsonl
  run_manifest.json
```

## Compute boundary

Catalog aggregates (`count_features`, `build_duplicate_groups`) and selection scoring helpers are accessed through `ComputeBackend` (`cv_preprocess.compute`). See [rust-boundary.md](rust-boundary.md) for future native acceleration.

## Config

`PipelineConfig` (`schema_version`, `dataset_builder`, `compute`) is loaded from YAML. When `dataset_builder.enabled: false`, legacy `preprocess` behavior is unchanged.

## Related docs

- [dataset-builder.md](dataset-builder.md) — builder workflow and CLI
- [catalog-schema.md](catalog-schema.md) — Parquet columns
- [selection-algorithm.md](selection-algorithm.md) — scoring math
- [migration-v1-v2.md](migration-v1-v2.md) — upgrading from legacy preprocess
