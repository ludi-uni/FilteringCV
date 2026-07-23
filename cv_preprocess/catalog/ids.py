from __future__ import annotations

import hashlib


def stable_clip_id(
    source_release: str,
    normalized_relative_source_path: str,
    source_row_index: int,
    audio_sha256: str,
    text_raw: str,
) -> str:
    """Return a deterministic clip identifier from stable input fields."""
    payload = (
        source_release
        + normalized_relative_source_path
        + str(source_row_index)
        + audio_sha256
        + text_raw
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
