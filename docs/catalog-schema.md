# Catalog Schema

Parquet artifacts under `{work_dir}/catalog/` share a fixed schema version recorded in `manifest.json`.

## clips.parquet

Primary catalog table. One row per analyzed clip.

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | string | Stable SHA256-based ID |
| `source_release` | string | Corpus release label |
| `normalized_relative_source_path` | string | POSIX-relative audio path |
| `source_row_index` | int | Index in source TSV |
| `audio_sha256` | string | Raw audio file hash |
| `text_raw` | string | Original sentence |
| `text_norm` | string | Normalized text for TTS |
| `speaker_id` | string | `client_id` from TSV |
| `sentence_id` | string | Optional sentence ID |
| `pipeline_hash` | string | Config fingerprint for cache |
| `split` | string | train/val/test (after split plan) |
| `disposition` | string | `eligible`, `hard_rejected`, `selected`, `reserve` |
| `reject_reason` | string | Set when hard_rejected |
| `duration_sec` | float | Processed duration |
| `quality_score` | float | 0–100 gate score |
| `estimated_snr_db` | float | SNR estimate |
| `silence_ratio` | float | Silence fraction |
| `phonemes` | string | Space-separated phoneme tokens |
| `feature_source` | string | `text_g2p`, `aligned`, etc. |
| `biphones` | list[string] | Bigram tokens |
| `triphones` | list[string] | Trigram tokens |
| `moras` | list[string] | Mora tokens |
| `fullcontext_labels` | list[string] | Full-context labels |
| `audio_cache_rel_path` | string | Relative path under `work_dir` |
| `analyzed_at` | string | ISO timestamp |

Additional nullable columns (`duplicate_group_ids`, `selection_rank`, `override_flags`, etc.) are reserved for downstream stages.

## feature_counts.parquet

Aggregated linguistic token frequencies (built by `ComputeBackend.count_features`).

| Column | Type |
|--------|------|
| `feature_type` | string (`phone`, `biphone`, `triphone`, `mora`, `fullcontext`) |
| `feature` | string |
| `count` | int |
| `speaker_count` | int |
| `utterance_count` | int |

## speaker_stats.parquet

Per-speaker rollups: `speaker_id`, `n_clips`, `n_eligible`, `duration_sec_sum`, `mean_quality`.

## duplicate_groups.parquet

Groups of two or more clips sharing an identity key.

| Column | Type |
|--------|------|
| `group_id` | string (16-char stable hash) |
| `kind` | string (see below) |
| `clip_ids` | list[string] |
| `size` | int |

**Kinds:** `exact_audio`, `same_source_path`, `same_sentence_id`, `same_normalized_text`, `same_speaker_same_text`.

## manifest.json

Analyze metadata: `schema_version`, `pipeline_hash`, `eligible_count`, `hard_rejected_count`, warnings.

## Plans

- `plans/split_plan.json` — speaker/clip split assignments
- `plans/selection_plan.parquet` — `clip_id`, `disposition`, `selection_score`, `selection_rank`, contribution JSON
