# Migration: Legacy Preprocess → Dataset Builder v2

This guide covers moving from sequential `preprocess` output to the dataset builder (`schema_version: 2` workflow).

## What changes

| Legacy (`preprocess`) | Builder (`build`) |
|-----------------------|-------------------|
| Immediate accept/reject per clip | Analyze → catalog with dispositions |
| Speaker caps as reject reasons | Speaker caps at selection only |
| `output/wavs` during preprocess | Audio cache at analyze; export at materialize |
| `metadata.jsonl` primary artifact | `work/catalog/clips.parquet` + plans |

## Enable builder

```yaml
schema_version: 2
dataset_builder:
  enabled: true
  work_dir: work
  target_duration_hours: 10
```

Keep `dataset_builder.enabled: false` to preserve legacy behavior.

## Command mapping

| Legacy | Builder equivalent |
|--------|------------------|
| `cv-preprocess preprocess` | `cv-preprocess build` (or stage-by-stage) |
| N/A | `cv-preprocess analyze` |
| N/A | `cv-preprocess select` |
| N/A | `cv-preprocess materialize` |

`preprocess` with `enabled: true` warns and delegates to `build`.

## Config additions

New top-level/nested blocks (defaults are safe):

- `dataset_builder.selection` — feature weights, local search, reserve ratio
- `dataset_builder.split` — unseen/seen/single speaker protocols
- `dataset_builder.duplicates` — per-kind max_selected
- `dataset_builder.distribution_temperature` — per-family α
- `compute.backend` — `auto` (Polars) | `python`

## Artifacts

Legacy `metadata.jsonl` is still produced by `preprocess` when builder is disabled. With builder enabled, use `materialize` output at the dataset root (`metadata.jsonl`, `validated.tsv`, `metadata.csv`, `train.jsonl` / `train.tsv`, etc.).

## Overrides

Builder-specific: `work/overrides.jsonl` for per-clip force include/exclude. Not read by legacy preprocess.

## Rollback

Set `dataset_builder.enabled: false` and run `preprocess` as before. Builder `work/` artifacts are ignored.

## Rust / native acceleration

No Rust code ships in v2. See [rust-boundary.md](rust-boundary.md) for the planned `NativeComputeBackend` boundary.
