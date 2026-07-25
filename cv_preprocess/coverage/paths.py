"""Resolve coverage automation artifact paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cv_preprocess.config.pipeline import PipelineConfig


@dataclass(frozen=True)
class CoveragePaths:
    output_dir: Path
    index_path: Path
    meta_path: Path
    plan_path: Path
    run_dir: Path


def project_base_from_config_path(config_path: Path) -> Path:
    """Best-effort project root when jobs run without an explicit project_root."""
    parent = Path(config_path).resolve().parent
    if parent.name in {"config", "configs"}:
        return parent.parent
    return Path.cwd().resolve()


def resolve_coverage_paths(
    config: PipelineConfig,
    *,
    base_dir: Path | None = None,
) -> CoveragePaths:
    """Resolve index/plan/run paths under ``coverage.output_dir``."""
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    out = Path(config.coverage.output_dir)
    if not out.is_absolute():
        out = (base / out).resolve()
    else:
        out = out.resolve()
    run_dir = out / str(config.coverage.active_run_dirname or "active-run")
    index_path = out / "clip-index.jsonl"
    return CoveragePaths(
        output_dir=out,
        index_path=index_path,
        meta_path=index_path.with_name(index_path.stem + ".meta.json"),
        plan_path=out / "plan.json",
        run_dir=run_dir,
    )


def accepted_catalog_path(config: PipelineConfig, *, base_dir: Path | None = None) -> Path:
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    work = Path(config.dataset_builder.work_dir)
    if not work.is_absolute():
        work = (base / work).resolve()
    return work / "catalog" / "clips.parquet"
