# Trainer Export (piper_plus / Style-Bert-VITS2) Design

**Date:** 2026-07-26
**Status:** Approved / implemented
**Approach:** Dedicated `cv_preprocess/export/` module, hooked from materialize + standalone CLI

## Goal

Convert FilteringCV materialize output into training layouts expected by:

1. **piper_plus** (LJSpeech-compatible preprocess input)
2. **Style-Bert-VITS2** (`Data/{model}/raw` + `esd.list`)

Existing materialize artifacts (`wavs/`, `metadata.jsonl`, `validated.tsv`, `metadata.csv`) remain unchanged.

## Non-goals

- Running piper_plus or Style-Bert-VITS2 training
- Phoneme alignment / cleaned 7-column SBV2 lists
- Cadence sidecars / ONNX export
- Changing speaker selection or coverage-aware select

## Decisions (from brainstorming)

| Topic | Choice |
|-------|--------|
| When to export | **Both**: during materialize (configurable) and via standalone CLI re-export |
| Resampling | **Configurable**; default **off** (layout-only). Optional: Piper 22050 Hz, SBV2 44100 Hz mono |
| SBV2 packaging | **Configurable**; default **single_model** multi-speaker folder; optional `per_speaker` |
| Text field | **Configurable**; default `text_norm` (`text_raw` allowed) |
| Piper speaker columns | **Auto**: 1 unique speaker → `id\|text`; 2+ → `id\|speaker\|text` |

## Architecture

```text
materialize (existing LJSpeech bundle)
        │
        ▼
cv_preprocess/export/
  protocol.py           ExportFormat, ExportRequest, ExportResult
  common.py             load metadata.jsonl, place/copy/link WAV, optional resample
  piper_plus.py         wav/ + metadata.csv (+ optional README)
  style_bert_vits2.py   Data/{model}/raw + esd.list
  runner.py             dispatch formats, shared validation

Hooks:
  application/materialize.py  → after staging manifests, if trainer_exports.enabled
  cli.py                      → export-trainer command
  jobs (optional)             → reuse CLI/application API; materialize job covers default path
```

Input for re-export: a completed materialize root containing `metadata.jsonl` and referenced WAV files.

## Config

```yaml
dataset_builder:
  materialize:
    trainer_exports:
      enabled: true
      formats:
        - piper_plus
        - style_bert_vits2
      text_field: text_norm   # text_norm | text_raw
      resample: false
      mode: null              # null → inherit materialize.mode (copy|hardlink|symlink)
      piper_plus:
        sample_rate: 22050
        wav_dirname: wav
      style_bert_vits2:
        sample_rate: 44100
        model_name: filteringcv
        language: JP
        packaging: single_model   # single_model | per_speaker
```

Defaults: both formats enabled when `trainer_exports.enabled` is true; resample off.

## Output layouts

Under `{materialize_output_root}/exports/`:

### piper_plus/

```text
exports/piper_plus/
  wav/{id}.wav
  metadata.csv
  README.txt          # short preprocess/train hint (optional but recommended)
```

`metadata.csv` (no header, `|` delimiter):

- Single speaker: `{id}|{text}`
- Multi speaker: `{id}|{speaker}|{text}`

`{id}` is the WAV stem (no extension). Paths are relative to the piper export root as expected by `piper_train.preprocess` (`wav/{id}.wav`).

### style_bert_vits2/

**single_model (default):**

```text
exports/style_bert_vits2/
  Data/{model_name}/
    esd.list
    raw/{id}.wav
```

`esd.list` UTF-8 no BOM:

```text
{id}.wav|{speaker}|{language}|{text}
```

**per_speaker:**

```text
exports/style_bert_vits2/
  Data/{speaker}/
    esd.list
    raw/{id}.wav
```

## CLI

```bash
cv-preprocess export-trainer \
  -c config/default.yaml \
  --format piper_plus \
  --input <materialize_output_root> \
  --output <optional override root>

cv-preprocess export-trainer \
  -c config/default.yaml \
  --format all \
  --input <materialize_output_root> \
  --resample
```

CLI flags override config for `format`, `resample`, and output location. `--format all` runs every format listed in config (or both built-ins if config empty).

## Materialize integration

After writing core manifests into staging (and before atomic publish):

1. If `trainer_exports.enabled`, run export writers targeting `staging/exports/...`
2. Include export paths in returned `manifest_paths` / job result metadata
3. Failure of export should fail materialize (same as missing WAV) so Build does not silently skip trainer packs

Re-export CLI may write into `{input}/exports/` or `--output`.

## Error handling

| Case | Behavior |
|------|----------|
| Missing WAV for a metadata row | Fail with clip id |
| Empty text after field selection | Skip clip + warning |
| Resample requested but decode fails | Fail that clip |
| Unknown format name | Fail fast |
| Zero clips after skips | Fail (empty export) |

## Testing

- Piper single-speaker CSV (2 columns)
- Piper multi-speaker CSV (3 columns)
- SBV2 `esd.list` line shape + `raw/` placement
- SBV2 `per_speaker` directory layout
- Re-export from existing materialize fixture without re-analyze
- Default `resample: false` does not change sample rate
- Optional resample changes rate when enabled (mocked or tiny wav)
- materialize with `trainer_exports.enabled: false` leaves no `exports/` dir

## Docs / UX

- README + `docs/dataset-builder.md`: how to enable and where files land
- `docs/gui.md` / Jobs materialize summary: mentions `exports/piper_plus` and `exports/style_bert_vits2`
- Update `config/default.yaml` / `example.yaml` with commented or enabled `trainer_exports` block

## Implementation sketch (files)

| Action | Path |
|--------|------|
| Create | `cv_preprocess/export/__init__.py` |
| Create | `cv_preprocess/export/protocol.py` |
| Create | `cv_preprocess/export/common.py` |
| Create | `cv_preprocess/export/piper_plus.py` |
| Create | `cv_preprocess/export/style_bert_vits2.py` |
| Create | `cv_preprocess/export/runner.py` |
| Modify | `cv_preprocess/config/dataset_builder.py` (`TrainerExportsConfig`, nested configs) |
| Modify | `cv_preprocess/application/materialize.py` |
| Modify | `cv_preprocess/cli.py` |
| Modify | `config/default.yaml`, `config/example.yaml` |
| Create | `tests/dataset_builder/test_trainer_exports.py` |
| Modify | docs as above |

## Open points (resolved unless user objects)

- Language tag for SBV2 defaults to `JP` (Japanese CV pipeline).
- `model_name` defaults to `filteringcv`; override in YAML.
- No GUI-only export job required for MVP; materialize + CLI suffice.
