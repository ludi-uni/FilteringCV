from __future__ import annotations

import hashlib
from collections import defaultdict

import polars as pl

from cv_preprocess.catalog.models import ClipDisposition

DuplicateKind = str

DUPLICATE_KINDS: tuple[DuplicateKind, ...] = (
    "exact_audio",
    "same_source_path",
    "same_sentence_id",
    "same_normalized_text",
    "same_speaker_same_text",
)

FEATURE_SPECS: tuple[tuple[str, str], ...] = (
    ("phone", "phonemes"),
    ("biphone", "biphones"),
    ("triphone", "triphones"),
    ("mora", "moras"),
    ("fullcontext", "fullcontext_labels"),
)


def normalized_text_sha256(text_norm: str | None) -> str | None:
    if not text_norm:
        return None
    return hashlib.sha256(text_norm.encode("utf-8")).hexdigest()


def _split_phoneme_tokens(phonemes: str | None) -> list[str]:
    if not phonemes:
        return []
    return [token for token in str(phonemes).replace("\t", " ").split() if token]


def _feature_values(feature_type: str, row: dict[str, object]) -> list[str]:
    if feature_type == "phone":
        return _split_phoneme_tokens(row.get("phonemes"))  # type: ignore[arg-type]
    column = next(col for ftype, col in FEATURE_SPECS if ftype == feature_type)
    values = row.get(column)
    if values is None:
        return []
    if isinstance(values, list):
        return [str(value) for value in values if value is not None and str(value)]
    return []


def build_feature_counts(clips: pl.DataFrame) -> pl.DataFrame:
    """Aggregate feature frequencies across analyzed clips."""
    if clips.is_empty():
        return pl.DataFrame(
            schema={
                "feature_type": pl.Utf8,
                "feature": pl.Utf8,
                "count": pl.Int64,
                "speaker_count": pl.Int64,
                "utterance_count": pl.Int64,
            }
        )

    counts: dict[tuple[str, str], int] = defaultdict(int)
    speakers: dict[tuple[str, str], set[str]] = defaultdict(set)
    utterances: dict[tuple[str, str], set[str]] = defaultdict(set)

    selected = clips.select(
        [
            "clip_id",
            "speaker_id",
            "phonemes",
            *[col for _, col in FEATURE_SPECS if col != "phonemes"],
        ]
    )
    for row in selected.iter_rows(named=True):
        clip_id = str(row["clip_id"])
        speaker_id = str(row.get("speaker_id") or "")
        for feature_type, _ in FEATURE_SPECS:
            for feature in _feature_values(feature_type, row):
                key = (feature_type, feature)
                counts[key] += 1
                speakers[key].add(speaker_id)
                utterances[key].add(clip_id)

    rows = [
        {
            "feature_type": feature_type,
            "feature": feature,
            "count": count,
            "speaker_count": len(speakers[(feature_type, feature)]),
            "utterance_count": len(utterances[(feature_type, feature)]),
        }
        for (feature_type, feature), count in sorted(counts.items())
    ]
    return pl.DataFrame(rows)


def build_speaker_stats(clips: pl.DataFrame) -> pl.DataFrame:
    """Summarize per-speaker clip counts and quality."""
    if clips.is_empty():
        return pl.DataFrame(
            schema={
                "speaker_id": pl.Utf8,
                "n_clips": pl.Int64,
                "n_eligible": pl.Int64,
                "duration_sec_sum": pl.Float64,
                "mean_quality": pl.Float64,
            }
        )

    eligible_value = ClipDisposition.ELIGIBLE.value
    return (
        clips.group_by("speaker_id")
        .agg(
            pl.len().alias("n_clips"),
            (pl.col("disposition") == eligible_value).sum().alias("n_eligible"),
            pl.col("duration_sec").fill_null(0.0).sum().alias("duration_sec_sum"),
            pl.col("quality_score").mean().alias("mean_quality"),
        )
        .sort("speaker_id")
    )


def _stable_group_id(kind: DuplicateKind, key: str) -> str:
    digest = hashlib.sha256(f"{kind}:{key}".encode("utf-8")).hexdigest()
    return digest[:16]


def _groups_from_key_map(
    *,
    kind: DuplicateKind,
    key_to_clip_ids: dict[str, list[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, clip_ids in sorted(key_to_clip_ids.items()):
        unique_ids = sorted(set(clip_ids))
        if len(unique_ids) < 2:
            continue
        rows.append(
            {
                "group_id": _stable_group_id(kind, key),
                "kind": kind,
                "clip_ids": unique_ids,
                "size": len(unique_ids),
            }
        )
    return rows


def _collect_key_groups(
    clips: pl.DataFrame,
    *,
    kind: DuplicateKind,
    key_expr: pl.Expr,
    key_name: str,
) -> list[dict[str, object]]:
    filtered = clips.filter(key_expr.is_not_null())
    if filtered.is_empty():
        return []

    grouped = (
        filtered.with_columns(key_expr.alias(key_name))
        .group_by(key_name)
        .agg(pl.col("clip_id").alias("clip_ids"))
    )
    key_to_clip_ids = {
        str(row[key_name]): [str(clip_id) for clip_id in row["clip_ids"]]
        for row in grouped.iter_rows(named=True)
    }
    return _groups_from_key_map(kind=kind, key_to_clip_ids=key_to_clip_ids)


def build_duplicate_groups(clips: pl.DataFrame) -> pl.DataFrame:
    """Find duplicate clip groups across several identity keys."""
    if clips.is_empty():
        return pl.DataFrame(
            schema={
                "group_id": pl.Utf8,
                "kind": pl.Utf8,
                "clip_ids": pl.List(pl.Utf8),
                "size": pl.Int64,
            }
        )

    work = clips
    if "normalized_text_sha256" not in work.columns:
        work = work.with_columns(
            pl.col("text_norm")
            .map_elements(
                lambda text: normalized_text_sha256(text) if text is not None else None,
                return_dtype=pl.Utf8,
            )
            .alias("normalized_text_sha256")
        )

    rows: list[dict[str, object]] = []
    rows.extend(
        _collect_key_groups(
            work,
            kind="exact_audio",
            key_expr=pl.col("audio_sha256"),
            key_name="audio_sha256",
        )
    )
    rows.extend(
        _collect_key_groups(
            work,
            kind="same_source_path",
            key_expr=pl.col("normalized_relative_source_path"),
            key_name="normalized_relative_source_path",
        )
    )
    rows.extend(
        _collect_key_groups(
            work.filter(pl.col("sentence_id").is_not_null()),
            kind="same_sentence_id",
            key_expr=pl.col("sentence_id"),
            key_name="sentence_id",
        )
    )
    rows.extend(
        _collect_key_groups(
            work.filter(pl.col("normalized_text_sha256").is_not_null()),
            kind="same_normalized_text",
            key_expr=pl.col("normalized_text_sha256"),
            key_name="normalized_text_sha256",
        )
    )

    speaker_text = work.filter(pl.col("text_norm").is_not_null() & pl.col("speaker_id").is_not_null())
    if not speaker_text.is_empty():
        speaker_text = speaker_text.with_columns(
            (pl.col("speaker_id") + "\x00" + pl.col("text_norm")).alias("speaker_text_key")
        )
        rows.extend(
            _collect_key_groups(
                speaker_text,
                kind="same_speaker_same_text",
                key_expr=pl.col("speaker_text_key"),
                key_name="speaker_text_key",
            )
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "group_id": pl.Utf8,
                "kind": pl.Utf8,
                "clip_ids": pl.List(pl.Utf8),
                "size": pl.Int64,
            }
        )
    return pl.DataFrame(rows).sort(["kind", "group_id"])
