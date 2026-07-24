"""Count coverage features from accepted clips / index records."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from cv_preprocess.coverage.feature_extractor import extract_coverage_features, unique_preserve_order
from cv_preprocess.coverage.models import ClipIndexRecord
from cv_preprocess.catalog.models import ClipDisposition


def _feature_keys_from_catalog_row(row: Mapping[str, Any]) -> list[str]:
    phonemes = row.get("phonemes")
    phoneme_str = ""
    if isinstance(phonemes, str):
        phoneme_str = phonemes
    elif isinstance(phonemes, list):
        phoneme_str = " ".join(str(p) for p in phonemes)

    text_norm = str(row.get("text_norm") or row.get("normalized_text") or "")
    if phoneme_str or text_norm:
        extracted = extract_coverage_features(normalized_text=text_norm, phoneme_str=phoneme_str)
        return extracted["feature_keys"]

    keys: list[str] = []
    phones = row.get("phones") or []
    if isinstance(phones, list):
        keys.extend(f"phoneme:{p}" for p in unique_preserve_order(str(p) for p in phones))
    return keys


def count_features_per_clip(
    feature_key_lists: Iterable[Sequence[str]],
) -> dict[str, int]:
    """Count each feature at most once per clip."""
    counter: Counter[str] = Counter()
    for keys in feature_key_lists:
        for key in set(keys):
            counter[key] += 1
    return dict(counter)


def count_from_index_records(
    records: Iterable[ClipIndexRecord],
    *,
    clip_ids: set[str] | None = None,
) -> dict[str, int]:
    lists: list[list[str]] = []
    for record in records:
        if clip_ids is not None and record.clip_id not in clip_ids:
            continue
        lists.append(list(record.feature_key_set()))
    return count_features_per_clip(lists)


def load_accepted_clip_ids_from_catalog(clips_parquet: Path) -> list[str]:
    df = pl.read_parquet(clips_parquet)
    if "disposition" not in df.columns or "clip_id" not in df.columns:
        return []
    accepted = {
        ClipDisposition.ELIGIBLE.value,
        ClipDisposition.SELECTED.value,
        ClipDisposition.RESERVE.value,
    }
    return (
        df.filter(pl.col("disposition").is_in(list(accepted)))
        .select("clip_id")
        .to_series()
        .to_list()
    )


def count_from_catalog_parquet(
    clips_parquet: Path,
    *,
    dispositions: set[str] | None = None,
) -> dict[str, int]:
    df = pl.read_parquet(clips_parquet)
    if dispositions is None:
        dispositions = {
            ClipDisposition.ELIGIBLE.value,
            ClipDisposition.SELECTED.value,
        }
    if "disposition" in df.columns:
        df = df.filter(pl.col("disposition").is_in(list(dispositions)))
    keys_lists: list[list[str]] = []
    for row in df.iter_rows(named=True):
        keys_lists.append(_feature_keys_from_catalog_row(row))
    return count_features_per_clip(keys_lists)


def count_from_metadata_jsonl(path: Path) -> dict[str, int]:
    """Legacy preprocess metadata.jsonl: each line is an accepted clip."""
    keys_lists: list[list[str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            keys_lists.append(_feature_keys_from_catalog_row(payload))
    return count_features_per_clip(keys_lists)


def load_accepted_counts(
    *,
    accepted_metadata: Path | None = None,
    catalog_clips: Path | None = None,
    index_records: Sequence[ClipIndexRecord] | None = None,
    accepted_clip_ids: Sequence[str] | None = None,
) -> dict[str, int]:
    if catalog_clips is not None and catalog_clips.is_file():
        return count_from_catalog_parquet(catalog_clips)
    if accepted_metadata is not None and accepted_metadata.is_file():
        suffix = accepted_metadata.suffix.lower()
        if suffix == ".parquet":
            return count_from_catalog_parquet(accepted_metadata)
        return count_from_metadata_jsonl(accepted_metadata)
    if index_records is not None and accepted_clip_ids is not None:
        return count_from_index_records(index_records, clip_ids=set(accepted_clip_ids))
    return {}
