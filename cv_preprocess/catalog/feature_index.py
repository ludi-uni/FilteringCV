from __future__ import annotations

from collections import defaultdict

import polars as pl


def build_phone_index(clips_df: pl.DataFrame, *, phoneme_column: str = "phonemes") -> dict[str, list[str]]:
    """Build a minimal inverted index mapping phone tokens to clip_ids."""
    index: dict[str, list[str]] = defaultdict(list)
    if phoneme_column not in clips_df.columns or "clip_id" not in clips_df.columns:
        return {}
    for row in clips_df.select(["clip_id", phoneme_column]).iter_rows(named=True):
        clip_id = row.get("clip_id")
        phonemes = row.get(phoneme_column)
        if not clip_id or not phonemes:
            continue
        tokens = [t for t in str(phonemes).replace("\t", " ").split() if t]
        for token in tokens:
            index[token].append(str(clip_id))
    return {token: sorted(set(ids)) for token, ids in sorted(index.items())}
