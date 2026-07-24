# Dataset Builder

The dataset builder constructs a curated TTS corpus under quality, coverage, speaker, duplicate, and split constraints. Enable it with `dataset_builder.enabled: true` in your pipeline YAML.

## Quick start

**Interactive (recommended):** after `uv sync --extra sidon --extra gui --extra dev`, run `./scripts/start-gui.sh` and use the Setup screen to pick or create a config. See [gui.md](gui.md).

**CLI / automation:**

```bash
# 1. Copy the template and edit corpus paths / speakers (optional if using GUI Setup)
cp config/example.yaml config/default.yaml

# 2. Install Sidon (and GUI if needed)
uv sync --extra sidon --extra gui --extra dev

# 3. Scan then build
cv-preprocess scan -c config/default.yaml
cv-preprocess build -c config/default.yaml
```

`build` orchestrates: scan → analyze → plan-split → select → materialize → audit, with stage resume (use `--force` to redo stages).

## Stages

| Stage | Command | Output |
|-------|---------|--------|
| Scan | `scan` | Corpus summary JSON |
| Analyze | `analyze` | `work/catalog/*.parquet`, audio cache (Sidon enhance) |
| Plan split | `plan-split` | `work/plans/split_plan.json` |
| Select | `select` | `work/plans/selection_plan.parquet` |
| Materialize | `materialize` | `wavs/` + LJSpeech `validated.tsv` / `metadata.csv` + `metadata.jsonl` + split manifests |
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
    max_duration_minutes: 120
  duplicates:
    exact_audio: { enabled: true, max_selected: 1 }
  split:
    protocol: unseen_speaker
    train: 0.85
    val: 0.10
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

## TTS audio quality (Sidon)

Materialize only exports `work/audio_cache` WAVs. Cleaning happens in **analyze**:

1. Put Sidon in `audio_pipeline_enhance` (`type: sidon_restore`) and set `two_pass_denoise.enabled: true` (see `config/example.yaml`).
2. Install deps: `uv sync --extra sidon`
3. Re-run analyze after changing the audio pipeline (`pipeline_hash` changes → new cache):
   `cv-preprocess build -c config/default.yaml --force`
4. Then select → materialize run as part of `build`

Without the enhance chain, exported `wavs/` are essentially decoded/resampled only.

See also: [selection-algorithm.md](selection-algorithm.md), [catalog-schema.md](catalog-schema.md).
