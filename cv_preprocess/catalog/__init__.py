from __future__ import annotations

from cv_preprocess.catalog.aggregates import (
    build_duplicate_groups,
    build_feature_counts,
    build_speaker_stats,
)
from cv_preprocess.catalog.cache import cached_wav_path, pipeline_cache_key, write_wav_atomic
from cv_preprocess.catalog.feature_index import build_phone_index
from cv_preprocess.catalog.ids import stable_clip_id
from cv_preprocess.catalog.models import CatalogRef, ClipDisposition
from cv_preprocess.catalog.reader import load_catalog, read_clips
from cv_preprocess.catalog.schema import CLIPS_COLUMNS, CLIPS_SCHEMA
from cv_preprocess.catalog.writer import write_catalog_bundle, write_clips_parquet

__all__ = [
    "CLIPS_COLUMNS",
    "CLIPS_SCHEMA",
    "CatalogRef",
    "ClipDisposition",
    "build_duplicate_groups",
    "build_feature_counts",
    "build_phone_index",
    "build_speaker_stats",
    "cached_wav_path",
    "load_catalog",
    "pipeline_cache_key",
    "read_clips",
    "stable_clip_id",
    "write_catalog_bundle",
    "write_clips_parquet",
    "write_wav_atomic",
]
