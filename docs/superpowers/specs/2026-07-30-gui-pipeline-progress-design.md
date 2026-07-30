# GUI Pipeline Progress Visibility Design

**Date:** 2026-07-30
**Status:** Approved
**Approach:** Gap-fill progress emission across pipeline stages + Jobs UI clarity (Approach 1)

## Goal

Make long-running pipeline stages show continuous, human-readable progress in the GUI: both a moving percent bar (`current` / `total` / `fraction`) and clear phase/message text, so jobs never look frozen while CPU is still working.

Observed failure mode (select job `c13930…`): progress events exist for milestones, but silent gaps of minutes during `features` and `coverage_reservation` make the Jobs page appear stuck.

## Non-goals

- Redesigning WebSocket transport or wiring `ProgressHub` into the worker subprocess
- New progress APIs or schema breaking changes to `ProgressEvent` / `ProgressRecord`
- Guaranteeing sub-second latency beyond the existing ~1s SQLite poll in the WebSocket handler
- Changing selection / audit / materialize algorithms beyond progress reporting hooks

## Decisions (from brainstorming)

| Topic | Choice |
|-------|--------|
| Scope | **Entire pipeline** (not select-only) |
| UX granularity | **Both** bar motion and phase/message |
| Surfaces | **Backend emit + frontend display** |
| Implementation style | **Gap-fill** existing `ProgressEvent` pattern (analyze-style), not a new reporter framework |

## Progress contract (unchanged shape)

Reuse `cv_preprocess.application.common.ProgressEvent`:

| Field | Role |
|-------|------|
| `stage` | Job stage name (`select`, `analyze`, `audit`, …) |
| `message` | Short human-readable status line |
| `current` / `total` | Item counters for the bar |
| `fraction` | Optional explicit 0–1 progress |
| `metadata.phase` | Fine-grained sub-stage (`features`, `coverage_reservation`, …) |

Persistence and delivery stay as today:

```text
worker ProgressSink (JobProgressWriter)
  → SQLite progress_events (+ JSONL fallback)
  → WS /ws/jobs/{id} (history + 1s poll)
  → Jobs.tsx live progress
```

## Architecture

```text
Silent loops today                    After
─────────────────                     ─────
_clip_features_from_catalog           emit phase=features every N / 0.5s
reserve_coverage_clips                emit phase=coverage_reservation (throttled)
greedy / local_search                 keep emit; ensure writer flushes
audit_dataset (no progress)           add ProgressSink + phase milestones
scan / plan-split (one message)       start + done with fraction
analyze post-loop catalog write       phase=catalog_write
materialize trainer exports           phase=trainer_export
coverage-plan / coverage-report       start + done with fraction

JobProgressWriter._should_flush
  + always-flush phases: greedy, local_search, coverage_reservation

Jobs.tsx
  + clearer stage/phase/message header
  + always show current/total/%/updated time when present
  + active-but-stale hint distinct from WS disconnect
```

## Backend changes

### Priority 1 — select (root cause of freeze)

1. **`_clip_features_from_catalog`** (`application/select.py`)
   - Add optional `progress: ProgressSink | None`.
   - While iterating catalog rows, emit `stage=select`, `phase=features`, updating `current`/`total`/`fraction` (map roughly 0.02→0.05).
   - Throttle emission (every ~500 rows or ≥0.5s), always emit first and last.

2. **`reserve_coverage_clips`** (`selection/coverage_reservation.py`)
   - Add optional progress callback (or `ProgressSink` + label).
   - Each reservation iteration (throttled): `phase=coverage_reservation`, `current` = reserved count, useful `total` (e.g. initial required deficit sum or eligible size), `metadata.remaining_deficit_total`.
   - Wire from `python_backend.greedy_local_search` which already emits start/end around the call.

3. **`JobProgressWriter._should_flush`** (`jobs/progress.py`)
   - Add `greedy`, `local_search`, `coverage_reservation` to the always-flush `phase` set (or ensure interval flush is reliable for these).
   - Do not fail jobs on progress write errors (existing JSONL fallback remains).

### Priority 2 — other silent / thin stages

| Stage | Change |
|-------|--------|
| `scan` | Emit start (`fraction=0`, `phase=start`) and done (`fraction=1`, `phase=complete`) |
| `plan-split` | Same start/done pattern |
| `audit` | Add `progress` parameter; emit phases for load → constraints → leakage → write (`fraction` steps) |
| `analyze` post-loop | Before catalog assembly / `write_catalog_bundle`, emit `phase=catalog_write` |
| `materialize` | Around `run_trainer_exports`, emit `phase=trainer_export` with format name(s) |
| `coverage-plan` / `coverage-report` | Reinforce start/done + fraction if still single-shot |

Build orchestrator stage-transition messages stay as-is; child stages own detailed progress.

### Conventions

- Prefer analyze-style: emit often in loops; let `JobProgressWriter` throttle DB writes.
- Keep existing cancellation checks where present.
- Progress is best-effort; never abort the job solely because a progress emit failed.

## Frontend changes (`frontend/src/pages/Jobs.tsx`)

1. Keep WebSocket + percent derivation (`fraction` else `current/total`).
2. Make **stage + phase + message** a stable, prominent header line.
3. Always show **%**, **current/total**, and **last updated** when available.
4. When job is active and last progress is stale (**≥5 seconds**) but WS is connected, show a **processing** hint (distinct from disconnect).
5. Keep the existing log list; no major redesign.

## Testing

| Area | Assertion |
|------|-----------|
| select features | Multiple progress events during feature build (not only start) |
| coverage_reservation | Progress during reservation loop on a small synthetic candidate set |
| audit | Phased progress events when `progress=` is passed |
| progress writer | Phases `greedy` / `coverage_reservation` flush under throttle rules |
| regression | Existing analyze / coverage-index / materialize progress tests still pass |

Prefer unit tests with in-memory sinks over full GUI e2e for this change.

## Error handling & constraints

- SQLite lock / WSL bind-mount issues: keep JSONL fallback; UI may lag but must not crash the worker.
- Delivery latency remains up to ~1s via WS poll.
- No new dependencies.

## Acceptance criteria

1. Running select shows updating `%` or `current/total` during **features** and **coverage_reservation**, not only at phase boundaries.
2. audit / scan / plan-split show at least start→done (and audit shows intermediate phases).
3. analyze catalog write and materialize trainer export are not silent.
4. Jobs UI shows stage, phase, message, counters, and last-updated clearly while a job is active.
5. Existing progress-emitting stages (analyze, materialize clip copy, coverage-index/run) remain correct.

## Out of scope follow-ups (optional later)

- Push progress from worker into in-process `ProgressHub` (eliminate 1s poll)
- REST fallback polling when WebSocket disconnects
- Shared ProgressReporter helper library across all stages
