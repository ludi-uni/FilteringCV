# Trainer Exports Implementation Plan

> **For agentic workers:** Use inline execution or subagent-driven development. Steps use checkbox syntax.

**Goal:** Export FilteringCV materialize output to piper_plus and Style-Bert-VITS2 training layouts (materialize hook + `export-trainer` CLI).

**Architecture:** New `cv_preprocess/export/` package with format writers; `materialize` calls runner when `trainer_exports.enabled`; CLI reuses the same runner from an existing materialize root.

**Tech Stack:** Python 3.12, Pydantic v2, soundfile/librosa (optional resample), Typer CLI, pytest.

## Global Constraints

- Do not change existing LJSpeech materialize files (`wavs/`, `metadata.jsonl`, `validated.tsv`).
- Default: layout-only (no resample); Piper auto single/multi CSV; SBV2 `single_model` + `JP` + `text_norm`.
- Empty text → skip + warning; missing WAV → fail; empty export → fail.
- No piper_plus / SBV2 training execution.

---

### Task 1: Config models

**Files:**
- Modify: `cv_preprocess/config/dataset_builder.py`
- Modify: `config/default.yaml`, `config/example.yaml`

- [ ] Add `PiperPlusExportConfig`, `StyleBertVits2ExportConfig`, `TrainerExportsConfig`
- [ ] Nest under `MaterializeConfig.trainer_exports` (default `enabled=True`, both formats)
- [ ] YAML blocks in default/example

### Task 2: Export package core

**Files:**
- Create: `cv_preprocess/export/{__init__,protocol,common,piper_plus,style_bert_vits2,runner}.py`

- [ ] Protocol types + load metadata.jsonl + place WAV (+ optional resample via soundfile/librosa)
- [ ] piper_plus writer + SBV2 writer
- [ ] `run_trainer_exports(...)` dispatcher

### Task 3: Materialize + CLI

**Files:**
- Modify: `cv_preprocess/application/materialize.py`
- Modify: `cv_preprocess/cli.py`
- Modify: `frontend/src/jobs/pipeline.ts` (materialize produces text)
- Docs: README, dataset-builder.md, gui.md

- [ ] Hook exports before publish
- [ ] `export-trainer` command
- [ ] UX/docs notes

### Task 4: Tests

**Files:**
- Create: `tests/dataset_builder/test_trainer_exports.py`

- [ ] Piper 1-spk / multi-spk, SBV2 single_model / per_speaker, re-export, disabled exports, optional resample

---
