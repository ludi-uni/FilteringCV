# Rust / Native Compute Boundary

FilteringCV v2 ships **Python-only** compute implementations. This document defines the boundary for a future native (`Rust` / PyO3) acceleration layer without implying any stub exists in the repository.

## ComputeBackend protocol

```python
class ComputeBackend(Protocol):
    def count_features(clips: pl.DataFrame) -> pl.DataFrame: ...
    def build_duplicate_groups(clips: pl.DataFrame) -> pl.DataFrame: ...
    def score_candidates(...) -> dict[str, tuple[float, dict, dict]]: ...
    def update_selection_state(state, clip) -> None: ...
```

Location: `cv_preprocess/compute/protocol.py`

## Implementations (shipped)

| Name | Module | Role |
|------|--------|------|
| `PolarsComputeBackend` | `compute/polars_backend.py` | Default for `compute.backend: auto` |
| `PythonComputeBackend` | `compute/python_backend.py` | Pure-Python fallback |

Loader: `cv_preprocess/compute/loader.py` — `resolve_compute_backend("auto")` picks Polars when importable.

## Reserved name (not implemented)

**`NativeComputeBackend`** — planned PyO3 extension for hot paths:

- Feature counting over large catalogs
- Batch candidate scoring during greedy selection
- Duplicate group indexing

There is **no** `NativeComputeBackend` class, empty crate, or stub module in this repository. Do not add a pretend implementation; document and benchmark against the Python/Polars backends only.

## Selection backend (separate)

`SelectionBackend` (`cv_preprocess/selection/protocol.py`) orchestrates greedy + local search. It is distinct from `ComputeBackend`. A future native layer would accelerate primitives inside `ComputeBackend`, not replace selection policy.

## Profiling contract

`run_manifest.json` records:

- `backend` — configured compute backend name
- `stage_timings_sec` — per-stage wall time
- `resources` — wall/cpu/rss snapshot at build end (`compute/profiling.py`)

Use `cv-preprocess benchmark-selection` to compare backends on a fixed catalog.

## Integration checklist (future)

1. Implement `NativeComputeBackend` in a separate optional extra (e.g. `cv-preprocess-native`)
2. Register in `resolve_compute_backend` behind explicit `compute.backend: native`
3. Golden tests: native ≡ polars on fixture catalogs
4. Document memory/ABI requirements in this file

## Non-goals (current delivery)

- Tauri desktop shell
- In-tree Rust crate
- Breaking changes to Parquet catalog schema
