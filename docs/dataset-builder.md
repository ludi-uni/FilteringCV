# Dataset Builder

The dataset builder constructs a curated TTS corpus under quality, coverage, speaker, duplicate, and split constraints. Enable it with `dataset_builder.enabled: true` in your pipeline YAML.

## Quick start

```bash
# 1. Configure input.corpus_root and dataset_builder.work_dir in config YAML
cv-preprocess scan -c config/my_builder.yaml
cv-preprocess build -c config/my_builder.yaml
```

`build` orchestrates: scan → analyze → plan-split → select → materialize → audit, with stage resume (use `--force` to redo stages).

## Stages

| Stage | Command | Output |
|-------|---------|--------|
| Scan | `scan` | Corpus summary JSON |
| Analyze | `analyze` | `work/catalog/*.parquet`, audio cache |
| Plan split | `plan-split` | `work/plans/split_plan.json` |
| Select | `select` | `work/plans/selection_plan.parquet` |
| Materialize | `materialize` | WAVs + metadata under output dir |
| Audit | `audit` | Integrity checks |
| Full pipeline | `build` | All of the above + `run_manifest.json` |

## Key config blocks

```yaml
dataset_builder:
  enabled: true
  work_dir: work
  target_duration_hours: 10
  random_seed: 42
  selection:
    reserve_ratio: 0.1
    feature_weights: { phone: 1.0, mora: 1.0, quality: 0.2 }
    local_search:
      enabled: true
  speaker_constraints:
    max_duration_sec_per_speaker: 3600
  duplicates:
    exact_audio: { enabled: true, max_selected: 1 }
  split:
    protocol: unseen_speaker
    train: 0.9
    val: 0.05
    test: 0.05

compute:
  backend: auto   # auto | polars | python
```

## Overrides

Edit `work/overrides.jsonl` to force-include, force-exclude, or hard-reject clips, then re-run `select` without re-analyzing.

## Benchmarking

Compare selection performance on an existing catalog:

```bash
cv-preprocess benchmark-selection --catalog work/catalog/clips.parquet --repeat 3
```

Optional: `--backend python|polars|auto`, `--config` for custom selection weights.

## Legacy compatibility

When `dataset_builder.enabled: false`, `cv-preprocess preprocess` runs the original pipeline unchanged. When enabled, `preprocess` warns and delegates to `build`.

See also: [selection-algorithm.md](selection-algorithm.md), [catalog-schema.md](catalog-schema.md).
