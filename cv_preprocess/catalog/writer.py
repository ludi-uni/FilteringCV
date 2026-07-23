from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from cv_preprocess.catalog.models import CatalogRef
from cv_preprocess.catalog.schema import CLIPS_COLUMNS, CLIPS_SCHEMA


def _atomic_replace(partial: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(final)


def _column_dtype(schema_type: str) -> pl.DataType:
    if schema_type == "string":
        return pl.Utf8
    if schema_type == "int64":
        return pl.Int64
    if schema_type == "float64":
        return pl.Float64
    if schema_type == "list[string]":
        return pl.List(pl.Utf8)
    return pl.Utf8


def _normalize_clips_df(df: pl.DataFrame) -> pl.DataFrame:
    for col in CLIPS_COLUMNS:
        if col not in df.columns:
            dtype = CLIPS_SCHEMA[col]
            df = df.with_columns(pl.lit(None, dtype=_column_dtype(dtype)).alias(col))
    return df.select(CLIPS_COLUMNS)


def write_clips_parquet(path: Path, rows: list[dict[str, Any]] | pl.DataFrame) -> None:
    path = Path(path)
    if isinstance(rows, pl.DataFrame):
        df = _normalize_clips_df(rows)
    else:
        df = _normalize_clips_df(pl.DataFrame(rows))
    partial = path.with_suffix(path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(partial)
    _atomic_replace(partial, path)


def _write_optional_parquet(path: Path, df: pl.DataFrame | None) -> Path | None:
    if df is None:
        return None
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(partial)
    _atomic_replace(partial, path)
    return path


def write_catalog_bundle(
    work_dir: Path,
    clips_df: pl.DataFrame | list[dict[str, Any]],
    *,
    feature_counts_df: pl.DataFrame | None = None,
    speaker_stats_df: pl.DataFrame | None = None,
    duplicate_groups_df: pl.DataFrame | None = None,
    manifest: dict[str, Any],
) -> CatalogRef:
    work_dir = Path(work_dir)
    catalog_dir = work_dir / "catalog"
    clips_path = catalog_dir / "clips.parquet"
    write_clips_parquet(clips_path, clips_df)

    feature_counts_path = _write_optional_parquet(
        catalog_dir / "feature_counts.parquet", feature_counts_df
    )
    speaker_stats_path = _write_optional_parquet(
        catalog_dir / "speaker_stats.parquet", speaker_stats_df
    )
    duplicate_groups_path = _write_optional_parquet(
        catalog_dir / "duplicate_groups.parquet", duplicate_groups_df
    )

    manifest_path = catalog_dir / "manifest.json"
    partial_manifest = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    partial_manifest.parent.mkdir(parents=True, exist_ok=True)
    partial_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _atomic_replace(partial_manifest, manifest_path)

    return CatalogRef(
        work_dir=work_dir,
        clips_path=clips_path,
        feature_counts_path=feature_counts_path,
        speaker_stats_path=speaker_stats_path,
        duplicate_groups_path=duplicate_groups_path,
        manifest_path=manifest_path,
    )
