# GUI Pipeline Progress Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill silent progress gaps across the dataset-builder pipeline (especially select `features` / `coverage_reservation`) and make Jobs UI show stage, phase, counters, and a stale-processing hint.

**Architecture:** Keep the existing `ProgressEvent` → `JobProgressWriter` → SQLite/JSONL → WebSocket poll → `Jobs.tsx` path. Add analyze-style incremental emits in silent loops, expand always-flush phases, and clarify the live progress header in the GUI. No new transport or schema.

**Tech Stack:** Python 3.12, pytest, existing FastAPI job worker, React/TypeScript Vite frontend (`Jobs.tsx`).

**Spec:** `docs/superpowers/specs/2026-07-30-gui-pipeline-progress-design.md`

## Global Constraints

- Reuse `ProgressEvent` / `ProgressSink`; do not invent a new progress protocol.
- Do not redesign WebSocket or wire `ProgressHub` into the worker subprocess.
- Progress is best-effort: never fail a job solely because a progress write failed (keep JSONL fallback).
- Do not change selection / audit / materialize algorithms beyond progress hooks.
- No new dependencies.
- Delivery latency may remain ~1s (existing WS SQLite poll).

## File map

| File | Responsibility |
|------|----------------|
| `cv_preprocess/jobs/progress.py` | Always-flush phases for select loops |
| `cv_preprocess/application/select.py` | Feature-build loop progress |
| `cv_preprocess/selection/coverage_reservation.py` | Reservation loop progress callback |
| `cv_preprocess/selection/python_backend.py` | Pass progress into reservation |
| `cv_preprocess/application/scan.py` | start/done progress |
| `cv_preprocess/application/split.py` | start/done progress |
| `cv_preprocess/application/audit.py` | Phased progress + wire callers |
| `cv_preprocess/jobs/worker.py` | Pass progress into audit; coverage-plan/report fractions |
| `cv_preprocess/application/build.py` | Pass progress into audit |
| `cv_preprocess/application/analyze.py` | `catalog_write` phase |
| `cv_preprocess/application/materialize.py` | `trainer_export` phase |
| `frontend/src/pages/Jobs.tsx` | Clearer live progress + 5s stale hint |
| `tests/dataset_builder/test_progress_visibility.py` | New unit tests for emits + flush |
| `tests/dataset_builder/test_coverage_aware_select.py` | Extend reservation progress if useful |

---

### Task 1: Always-flush select phases in JobProgressWriter

**Files:**
- Modify: `cv_preprocess/jobs/progress.py`
- Create: `tests/dataset_builder/test_progress_visibility.py`

**Interfaces:**
- Consumes: `JobProgressWriter._should_flush`, `ProgressEvent.metadata["phase"]`
- Produces: always-flush for phases `greedy`, `local_search`, `coverage_reservation` (in addition to existing set)

- [ ] **Step 1: Write the failing test**

Create `tests/dataset_builder/test_progress_visibility.py`:

```python
from __future__ import annotations

from cv_preprocess.application.common import ProgressEvent
from cv_preprocess.jobs.progress import JobProgressWriter
from cv_preprocess.jobs.store import JobStore
from cv_preprocess.jobs.models import JobType


def test_should_flush_select_loop_phases(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.create_job(JobType.SELECT, config_path=tmp_path / "c.yaml")
    writer = JobProgressWriter(store, job.id, min_interval_sec=60.0, min_step=10_000)
    # Seed last flush so interval/step would otherwise block.
    writer(
        ProgressEvent(
            stage="select",
            message="seed",
            current=1,
            total=100,
            fraction=0.01,
            metadata={"phase": "prepare"},
        )
    )
    # Keep message and current nearly unchanged so only always-flush phases
    # (not message-change / step / interval) can cause these writes.
    for phase in ("greedy", "local_search", "coverage_reservation"):
        writer(
            ProgressEvent(
                stage="select",
                message="seed",
                current=2,
                total=100,
                fraction=0.02,
                metadata={"phase": phase},
            )
        )
    rows = store.list_progress(job.id)
    phases = [r.metadata.get("phase") for r in rows]
    assert "greedy" in phases
    assert "local_search" in phases
    assert "coverage_reservation" in phases
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dataset_builder/test_progress_visibility.py::test_should_flush_select_loop_phases -v`

Expected: FAIL (phases dropped by throttle — missing from `list_progress`)

- [ ] **Step 3: Write minimal implementation**

In `cv_preprocess/jobs/progress.py`, extend the always-flush `phase` set inside `_should_flush`:

```python
        if phase in {
            "prepare",
            "reserve",
            "done",
            "load",
            "features",
            "start",
            "complete",
            "split",
            "index",
            "run",
            "coverage",
            "plan",
            "report",
            "greedy",
            "local_search",
            "coverage_reservation",
        }:
            return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/dataset_builder/test_progress_visibility.py::test_should_flush_select_loop_phases -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cv_preprocess/jobs/progress.py tests/dataset_builder/test_progress_visibility.py
git commit -m "fix: always flush select greedy/local_search/coverage_reservation progress"
```

---

### Task 2: Select feature-build progress

**Files:**
- Modify: `cv_preprocess/application/select.py`
- Modify: `tests/dataset_builder/test_progress_visibility.py`

**Interfaces:**
- Consumes: `ProgressSink`, `ProgressEvent`
- Produces: `_clip_features_from_catalog(..., progress: ProgressSink | None = None)` emitting `phase=features` with updating `current`/`total`/`fraction` (≈0.02→0.05); `select_dataset` passes `progress` through

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset_builder/test_progress_visibility.py`. Prefer copying the minimal column set from `tests/dataset_builder/test_selection_order_independence.py` helpers rather than inventing incomplete rows.

```python
import polars as pl

from cv_preprocess.application.select import _clip_features_from_catalog
from cv_preprocess.config import PipelineConfig


def test_clip_features_emits_incremental_progress():
    config = PipelineConfig()
    # Build 1200 eligible rows with the same columns the existing select helpers use.
    rows = [
        {
            "clip_id": f"c{i}",
            "speaker_id": "s1",
            "duration_sec": 1.0,
            "disposition": "ELIGIBLE",
            "phonemes": "a",
            "text_norm": "あ",
            "quality_score": 80.0,
            "audio_sha256": "",
            "sentence_id": f"sent{i}",
            "estimated_snr_db": None,
            "silence_ratio": None,
            "rms": None,
            "peak": None,
            "f0_median": None,
            "f0_range": None,
            "speech_rate": None,
            "alignment_confidence": None,
            "split": None,
        }
        for i in range(1200)
    ]
    df = pl.DataFrame(rows)
    events: list[ProgressEvent] = []
    _clip_features_from_catalog(df, config, {}, None, progress=events.append)
    feature_events = [e for e in events if e.metadata.get("phase") == "features"]
    assert len(feature_events) >= 3
    assert feature_events[-1].current == feature_events[-1].total
    assert feature_events[-1].total == 1200
```

If `_features_from_row` requires extra columns, extend the row dict from the existing helper instead of guessing.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dataset_builder/test_progress_visibility.py::test_clip_features_emits_incremental_progress -v`

Expected: FAIL (`progress` unexpected keyword or only 0–1 events)

- [ ] **Step 3: Write minimal implementation**

In `cv_preprocess/application/select.py`:

1. Change signature:

```python
def _clip_features_from_catalog(
    df: pl.DataFrame,
    config: PipelineConfig,
    overrides: dict[str, Any],
    split_plan: SplitPlan | None,
    progress: ProgressSink | None = None,
) -> list[ClipFeatures]:
```

2. Inside the loop, track index over `df.height` (or enumerated processed count). Throttle with `time.monotonic()` (≥0.5s) **or** every 500 rows; always emit at start (`current=0`) and end. Emit:

```python
ProgressEvent(
    stage="select",
    message=f"building candidate features {current}/{total}",
    current=current,
    total=total,
    fraction=0.02 + 0.03 * (current / total if total else 1.0),
    metadata={"phase": "features"},
)
```

3. In `select_dataset`, pass `progress=progress` into `_clip_features_from_catalog(...)`. Keep or replace the existing pre-loop `phase=features` milestone so the first incremental emit remains coherent (avoid duplicate identical noise if redundant).

Call sites that omit `progress` (`benchmark_selection.py`, tests) remain valid via default `None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/dataset_builder/test_progress_visibility.py::test_clip_features_emits_incremental_progress -v`

Expected: PASS

Also run: `uv run pytest tests/dataset_builder/test_selection_order_independence.py tests/dataset_builder/test_selection_vs_head.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cv_preprocess/application/select.py tests/dataset_builder/test_progress_visibility.py
git commit -m "feat: emit incremental progress while building select features"
```

---

### Task 3: Coverage reservation loop progress

**Files:**
- Modify: `cv_preprocess/selection/coverage_reservation.py`
- Modify: `cv_preprocess/selection/python_backend.py`
- Modify: `tests/dataset_builder/test_progress_visibility.py`

**Interfaces:**
- Consumes: existing `reserve_coverage_clips(...)` args; `_emit_progress` in `python_backend.py`
- Produces: `reserve_coverage_clips(..., progress: ProgressSink | None = None, progress_label: str | None = None)` emitting throttled `phase=coverage_reservation` with `metadata.remaining_deficit_total`

- [ ] **Step 1: Write the failing test**

Reuse small synthetic clips from `tests/dataset_builder/test_coverage_aware_select.py` (`_clip`, constraint builders). Add:

```python
def test_reserve_coverage_clips_emits_progress():
    # Copy a working setup from test_coverage_aware_select that calls reserve_coverage_clips.
    events: list[ProgressEvent] = []
    result = reserve_coverage_clips(
        candidates,
        constraints,
        constraint_config=constraint_config,
        progress=events.append,
    )
    assert result.reserved_ids
    phases = [e.metadata.get("phase") for e in events]
    assert "coverage_reservation" in phases
    assert any(e.current is not None and e.current > 0 for e in events)
    assert any("remaining_deficit_total" in e.metadata for e in events)
```

Fill `candidates` / `constraints` / `constraint_config` by copying an existing successful reservation test body.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dataset_builder/test_progress_visibility.py::test_reserve_coverage_clips_emits_progress -v`

Expected: FAIL (`progress` unexpected keyword)

- [ ] **Step 3: Write minimal implementation**

In `coverage_reservation.py`:

1. Add optional `progress: ProgressSink | None = None` and `progress_label: str | None = None`.
2. Import `ProgressEvent`, `ProgressSink` from `cv_preprocess.application.common`.
3. Before the `while True` loop, compute `initial_deficit_total = sum(_current_deficits(...).values())` (use `max(1, ...)` for totals).
4. After each successful reserve (throttled ≥0.5s; always first/last), emit:

```python
deficits = _current_deficits(constraints, selected_counts, required_only=True)
remaining = sum(deficits.values())
msg = f"reserved={len(reserved)} deficit_left={remaining}"
if progress_label:
    msg = f"[{progress_label}] {msg}"
progress(
    ProgressEvent(
        stage="select",
        message=msg,
        current=len(reserved),
        total=max(initial_deficit_total, 1),
        fraction=0.05 + 0.10 * (1.0 - remaining / max(initial_deficit_total, 1)),
        metadata={
            "phase": "coverage_reservation",
            "remaining_deficit_total": remaining,
            **({"label": progress_label} if progress_label else {}),
        },
    )
)
```

In `python_backend.py`, pass `progress=progress` and `progress_label=progress_label` into `reserve_coverage_clips(...)`. Keep the existing start/complete emissions around the call.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/dataset_builder/test_progress_visibility.py::test_reserve_coverage_clips_emits_progress tests/dataset_builder/test_coverage_aware_select.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cv_preprocess/selection/coverage_reservation.py cv_preprocess/selection/python_backend.py tests/dataset_builder/test_progress_visibility.py
git commit -m "feat: emit progress during coverage reservation"
```

---

### Task 4: Thin stages — scan, plan-split, coverage-plan/report fractions

**Files:**
- Modify: `cv_preprocess/application/scan.py`
- Modify: `cv_preprocess/application/split.py`
- Modify: `cv_preprocess/jobs/worker.py` (`_run_coverage_plan`, `_run_coverage_report`)
- Modify: `tests/dataset_builder/test_progress_visibility.py`

**Interfaces:**
- Produces: start (`fraction=0`, `phase=start`) + done (`fraction=1`, `phase=complete`/`done`) for each thin stage

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import patch

from cv_preprocess.application.scan import scan_project
from cv_preprocess.config import PipelineConfig


def test_scan_project_emits_start_and_done():
    events: list[ProgressEvent] = []
    fake = {"clip_count": 0, "speaker_count": 0, "corpus_root": ".", "tsv_path": None}
    with patch("cv_preprocess.application.scan.scan_corpus", return_value=fake):
        scan_project(PipelineConfig(), progress=events.append)
    assert len(events) >= 2
    assert events[0].fraction == 0.0
    assert events[-1].fraction == 1.0
```

Also add `fraction=0.0` / `fraction=1.0` on coverage-plan and coverage-report emits in `worker.py` (keep existing phases). For plan-split, emit start before work and complete after `write_split_plan`. Prefer a mocked lightweight test if fixtures exist; otherwise code change + scan test is the required automated coverage for this task.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dataset_builder/test_progress_visibility.py::test_scan_project_emits_start_and_done -v`

Expected: FAIL (only one event / missing fractions)

- [ ] **Step 3: Write minimal implementation**

`scan.py`:

```python
    if progress is not None:
        progress(
            ProgressEvent(
                stage="scan",
                message="scanning corpus",
                fraction=0.0,
                metadata={"phase": "start"},
            )
        )
    raw = scan_corpus(config)
    if progress is not None:
        progress(
            ProgressEvent(
                stage="scan",
                message="scan complete",
                fraction=1.0,
                metadata={"phase": "complete"},
            )
        )
    return ScanResult.model_validate(raw)
```

`split.py` (`plan_dataset_split`): emit start before work and complete after `write_split_plan` with `fraction` 0→1 and `phase` start/complete.

`worker.py` `_run_coverage_plan` / `_run_coverage_report`: add `fraction=0.0` to the first event and `fraction=1.0` to the done event.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/dataset_builder/test_progress_visibility.py -q`

Expected: PASS for new tests

- [ ] **Step 5: Commit**

```bash
git add cv_preprocess/application/scan.py cv_preprocess/application/split.py cv_preprocess/jobs/worker.py tests/dataset_builder/test_progress_visibility.py
git commit -m "feat: add start/done progress fractions for thin pipeline stages"
```

---

### Task 5: Audit phased progress

**Files:**
- Modify: `cv_preprocess/application/audit.py`
- Modify: `cv_preprocess/jobs/worker.py`
- Modify: `cv_preprocess/application/build.py`
- Modify: `tests/dataset_builder/test_progress_visibility.py`

**Interfaces:**
- Produces: `audit_dataset(..., progress: ProgressSink | None = None)` with phases `load` → `constraints` → `leakage` → `write` (fractions e.g. 0.1 / 0.4 / 0.7 / 1.0)

- [ ] **Step 1: Write the failing test**

```python
def test_audit_dataset_emits_phases(...):
    events: list[ProgressEvent] = []
    # Reuse or build a minimal catalog + selection plan (prefer existing audit/dataset_builder fixtures).
    audit_dataset(config, catalog, selection_plan, progress=events.append)
    phases = [e.metadata.get("phase") for e in events]
    assert set(phases) >= {"load", "constraints", "leakage", "write"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dataset_builder/test_progress_visibility.py::test_audit_dataset_emits_phases -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Update `audit_dataset` signature and emit four progress events around the existing logic blocks. Wire:

```python
# worker.py
result = audit_dataset(config, catalog, selection_plan, progress=progress)

# build.py
audit_report = audit_dataset(config, catalog, selection_plan, progress=progress)
```

CLI may leave `progress=None`.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/dataset_builder/test_progress_visibility.py::test_audit_dataset_emits_phases tests/dataset_builder/test_build_resume.py -q
```

Expected: PASS (`test_build_resume` mocks must still accept kwargs)

- [ ] **Step 5: Commit**

```bash
git add cv_preprocess/application/audit.py cv_preprocess/jobs/worker.py cv_preprocess/application/build.py tests/dataset_builder/test_progress_visibility.py
git commit -m "feat: emit phased progress from audit_dataset"
```

---

### Task 6: Analyze catalog_write + materialize trainer_export progress

**Files:**
- Modify: `cv_preprocess/application/analyze.py`
- Modify: `cv_preprocess/application/materialize.py`
- Modify: `tests/dataset_builder/test_progress_visibility.py`

**Interfaces:**
- Produces: analyze post-loop `phase=catalog_write`; materialize `phase=trainer_export` before/after `run_trainer_exports`

- [ ] **Step 1: Write the failing tests**

Prefer mocking at boundaries:

```python
def test_materialize_trainer_export_progress_events(monkeypatch, tmp_path):
    events: list[ProgressEvent] = []
    # Arrange materialize with trainer_exports.enabled=True using patterns from
    # tests/dataset_builder/test_trainer_exports.py; monkeypatch run_trainer_exports to return [].
    # Assert any(e.metadata.get("phase") == "trainer_export" for e in events)
```

For analyze, emit before `write_catalog_bundle` and assert via a mocked `write_catalog_bundle` path or the smallest existing analyze fixture.

- [ ] **Step 2: Run tests to verify they fail**

Run the new test names under `test_progress_visibility.py -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

In `analyze.py`, immediately before catalog assembly / `write_catalog_bundle`:

```python
    if progress is not None:
        progress(
            ProgressEvent(
                stage="analyze",
                message="writing catalog bundle",
                fraction=0.99,
                metadata={"phase": "catalog_write"},
            )
        )
```

In `materialize.py`, around trainer exports:

```python
    if trainer_cfg.enabled:
        if progress is not None:
            progress(
                ProgressEvent(
                    stage="materialize",
                    message="running trainer exports",
                    fraction=0.95,
                    metadata={"phase": "trainer_export"},
                )
            )
        export_results = run_trainer_exports(...)
        if progress is not None:
            progress(
                ProgressEvent(
                    stage="materialize",
                    message="trainer exports complete",
                    fraction=0.99,
                    metadata={"phase": "trainer_export", "exports": len(export_results)},
                )
            )
```

Adapt metadata keys to actual `TrainerExportsConfig` fields if listing formats.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/dataset_builder/test_progress_visibility.py tests/dataset_builder/test_trainer_exports.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cv_preprocess/application/analyze.py cv_preprocess/application/materialize.py tests/dataset_builder/test_progress_visibility.py
git commit -m "feat: progress for analyze catalog write and trainer exports"
```

---

### Task 7: Jobs UI — stage/phase header + 5s stale hint

**Files:**
- Create: `frontend/src/jobs/progressDisplay.ts`
- Create: `frontend/src/jobs/progressDisplay.test.ts`
- Modify: `frontend/src/pages/Jobs.tsx`

**Interfaces:**
- Produces: `isProgressStale(latestCreatedAt, nowMs, staleMs=5000)`; Jobs live card shows stage/phase/message, counters, and `処理中…` when active+connected+stale≥5s

- [ ] **Step 1: Write the failing helper test**

Create `frontend/src/jobs/progressDisplay.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { isProgressStale } from "./progressDisplay";

describe("isProgressStale", () => {
  it("is false when updated recently", () => {
    const now = Date.parse("2026-07-30T12:00:05Z");
    expect(isProgressStale("2026-07-30T12:00:03Z", now)).toBe(false);
  });
  it("is true when older than 5s", () => {
    const now = Date.parse("2026-07-30T12:00:10Z");
    expect(isProgressStale("2026-07-30T12:00:03Z", now)).toBe(true);
  });
});
```

- [ ] **Step 2: Run helper test to verify it fails**

Run: `cd frontend && pnpm exec vitest run src/jobs/progressDisplay.test.ts`

Expected: FAIL (module missing)

- [ ] **Step 3: Implement helper + Jobs.tsx**

`frontend/src/jobs/progressDisplay.ts`:

```typescript
export function isProgressStale(
  latestCreatedAt: string | null | undefined,
  nowMs: number,
  staleMs = 5000,
): boolean {
  if (!latestCreatedAt) return false;
  const t = Date.parse(latestCreatedAt);
  if (Number.isNaN(t)) return false;
  return nowMs - t >= staleMs;
}
```

In `Jobs.tsx` Live progress card:

1. Keep/strengthen header: stage + phase pill + message.
2. Stats: show `%`, `current/total`, and last-updated whenever available.
3. Tick once per second (state + interval) and when `isActive && connected && isProgressStale(latest?.created_at, now)`, show muted `処理中…` (distinct from disconnect).

- [ ] **Step 4: Run frontend tests / typecheck**

```bash
cd frontend && pnpm exec vitest run src/jobs/progressDisplay.test.ts
cd frontend && pnpm exec tsc --noEmit
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Jobs.tsx frontend/src/jobs/progressDisplay.ts frontend/src/jobs/progressDisplay.test.ts
git commit -m "feat: clarify Jobs live progress and show stale processing hint"
```

---

### Task 8: Spec status + smoke verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-30-gui-pipeline-progress-design.md` (Status → Implemented)
- Optional: short note in `docs/gui.md` if it already documents Jobs progress

- [ ] **Step 1: Run full relevant pytest suite**

```bash
uv run pytest tests/dataset_builder/test_progress_visibility.py tests/dataset_builder/test_coverage_aware_select.py tests/dataset_builder/test_job_store_locking.py tests/coverage_automation/test_indexer_progress.py -q
```

Expected: all PASS

- [ ] **Step 2: Manual GUI smoke (if server available)**

1. Restart GUI (`./scripts/start-gui.sh`) so worker code reloads.
2. Start a **select** job (or Build that reaches select).
3. Confirm Live progress updates during `features` and `coverage_reservation`.
4. Confirm audit / thin stages show start→done.

- [ ] **Step 3: Update spec status**

Set `**Status:** Implemented` in the design doc.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-07-30-gui-pipeline-progress-design.md docs/gui.md
git commit -m "docs: mark GUI pipeline progress design implemented"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| select `features` incremental progress | Task 2 |
| select `coverage_reservation` incremental progress | Task 3 |
| always-flush greedy/local_search/coverage_reservation | Task 1 |
| scan / plan-split start+done | Task 4 |
| coverage-plan/report fractions | Task 4 |
| audit phased progress | Task 5 |
| analyze `catalog_write` | Task 6 |
| materialize `trainer_export` | Task 6 |
| Jobs UI stage/phase/counters + 5s stale hint | Task 7 |
| Acceptance / smoke | Task 8 |
| Non-goals (no WS redesign, no new protocol) | All tasks |

## Plan self-review notes

- No placeholder “implement later” steps; fixture details point at existing test helpers to copy.
- Optional `progress=None` keeps CLI and older call sites working.
- Frontend stale threshold is exactly **5000 ms** per spec.
