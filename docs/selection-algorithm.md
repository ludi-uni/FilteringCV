# Selection Algorithm

Dataset builder selection chooses clips to maximize linguistic coverage under hard constraints. The implementation lives in `cv_preprocess.selection`.

## Overview

1. **Force-includes** from `overrides.jsonl` (if constraints allow)
2. **Greedy loop** — pick highest marginal-utility eligible clip until target duration reached
3. **Reserve pool** — top `reserve_ratio` of remaining clips by score; tail shuffled by seed
4. **Local search** (optional) — swap selected ↔ reserve (`1v1`, `1v2`, `2v1`) until no improvement

For **`unseen_speaker`**, step 2 runs **once per split bucket** after speakers are assigned in `plan-split` (each split has its own duration share). For **`seen_speaker` / `single_speaker`**, selection is over the full pool and clip splits are finalized afterward. See [dataset-builder.md](dataset-builder.md#plan-split-vs-select-protocol-matters).

## Target distribution

For feature family `k` with temperature `α_k`:

```
P_target(f) ∝ P_pool(f)^α_k
```

where `P_pool(f)` is the fraction of eligible clips containing token `f`. Lower `α` flattens the target (less emphasis on frequent tokens).

## Diminishing returns

Per-token utility with weight `w` and scale `τ`:

```
utility(n) = w · log(1 + n/τ)
marginal(n) = utility(n+1) - utility(n)
```

## Per-clip score

For clip `c` and family `k`:

```
score_k(c) = Σ_{f ∈ unique tokens in c} P_target(f) · marginal(n_f)
```

Tokens below `feature_support.min_utterances` / `min_speakers` (pool-wide) are skipped (strong-rescue gate).

Bonuses:
- `speaker_diversity_weight` if clip introduces a new speaker
- `quality_weight · (quality_score / 100)`

## Hard constraints

Enforced via `can_add_clip` (not score penalties):

- Speaker clip/duration caps
- Duplicate group `max_selected` per kind
- `hard_min_quality`, `max_low_quality_ratio`

## Compute backend

Scoring helpers (`score_candidates`, `update_selection_state`) are exposed on `ComputeBackend` for benchmarking and future acceleration. Production selection uses `PythonSelectionBackend` (greedy + local search).

## Determinism

Same catalog + config + `random_seed` → same selected set. Tie-break: higher score, then lexicographically smaller `clip_id`.

## Re-select without re-analyze

Run `cv-preprocess select` after editing overrides or selection weights. Catalog parquet and audio cache are reused.

## Coverage-aware select

**Default: on** (`selection.coverage_constraints.enabled: true`). You do not need `--coverage-aware` for normal runs.

When enabled, selection:

1. Reserves clips for required coverage minima (greedy multi-feature set cover)
2. Fills remaining duration with the existing marginal-utility greedy loop
3. Runs local search that refuses swaps breaking `effective_minimum`
4. Writes coverage audit / missing-features under `{work_dir}/reports/selection/`

Bare coverage targets (`v: 5`) mean `minimum = desired = 5`. Effective targets are capped by eligible clip counts. Speaker selection ranking is unchanged.

Lightweight acoustic redundancy (`selection.acoustic_diversity`, also **on by default**) is a soft penalty below required coverage. Disable with `enabled: false` or `--disable-acoustic-diversity`.

```bash
cv-preprocess select -c config/default.yaml
cv-preprocess select -c config/default.yaml --coverage-audit-output work/reports/selection
```

## Benchmark

```bash
cv-preprocess benchmark-selection --catalog work/catalog/clips.parquet --repeat 3
```

Reports wall/cpu timing for scoring, greedy selection, and catalog aggregates.
