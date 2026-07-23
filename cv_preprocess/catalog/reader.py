from __future__ import annotations

from pathlib import Path

import polars as pl

from cv_preprocess.catalog.models import CatalogRef
from cv_preprocess.catalog.schema import CLIPS_COLUMNS


def read_clips(path: Path) -> pl.DataFrame:
    path = Path(path)
    df = pl.read_parquet(path)
    missing = [c for c in CLIPS_COLUMNS if c not in df.columns]
    if missing:
        for col in missing:
            df = df.with_columns(pl.lit(None).alias(col))
    return df.select(CLIPS_COLUMNS)


def load_catalog(work_dir: Path) -> CatalogRef:
    work_dir = Path(work_dir)
    catalog_dir = work_dir / "catalog"
    clips_path = catalog_dir / "clips.parquet"
    feature_counts_path = catalog_dir / "feature_counts.parquet"
    speaker_stats_path = catalog_dir / "speaker_stats.parquet"
    duplicate_groups_path = catalog_dir / "duplicate_groups.parquet"
    manifest_path = catalog_dir / "manifest.json"
    return CatalogRef(
        work_dir=work_dir,
        clips_path=clips_path if clips_path.is_file() else None,
        feature_counts_path=feature_counts_path if feature_counts_path.is_file() else None,
        speaker_stats_path=speaker_stats_path if speaker_stats_path.is_file() else None,
        duplicate_groups_path=duplicate_groups_path if duplicate_groups_path.is_file() else None,
        manifest_path=manifest_path if manifest_path.is_file() else None,
    )
