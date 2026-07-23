from __future__ import annotations

import polars as pl

from cv_preprocess.catalog.aggregates import build_duplicate_groups


def _clip_row(
    *,
    clip_id: str,
    audio_sha256: str,
    source_path: str,
    sentence_id: str | None = None,
    text_norm: str | None = None,
    speaker_id: str = "spk1",
) -> dict[str, object]:
    return {
        "clip_id": clip_id,
        "audio_sha256": audio_sha256,
        "normalized_relative_source_path": source_path,
        "sentence_id": sentence_id,
        "text_norm": text_norm,
        "speaker_id": speaker_id,
        "disposition": "eligible",
    }


def test_build_duplicate_groups_exact_audio() -> None:
    clips = pl.DataFrame(
        [
            _clip_row(
                clip_id="a",
                audio_sha256="hash1",
                source_path="a.wav",
                text_norm="こんにちは",
            ),
            _clip_row(
                clip_id="b",
                audio_sha256="hash1",
                source_path="b.wav",
                text_norm="おはよう",
            ),
            _clip_row(
                clip_id="c",
                audio_sha256="hash2",
                source_path="c.wav",
                text_norm="さようなら",
            ),
        ]
    )
    groups = build_duplicate_groups(clips)
    exact = groups.filter(pl.col("kind") == "exact_audio")
    assert exact.height == 1
    assert exact.row(0, named=True)["size"] == 2
    assert sorted(exact.row(0, named=True)["clip_ids"]) == ["a", "b"]


def test_build_duplicate_groups_same_normalized_text() -> None:
    clips = pl.DataFrame(
        [
            _clip_row(
                clip_id="a",
                audio_sha256="hash1",
                source_path="a.wav",
                text_norm="同じ文",
            ),
            _clip_row(
                clip_id="b",
                audio_sha256="hash2",
                source_path="b.wav",
                text_norm="同じ文",
            ),
        ]
    )
    groups = build_duplicate_groups(clips)
    same_text = groups.filter(pl.col("kind") == "same_normalized_text")
    assert same_text.height == 1
    assert same_text.row(0, named=True)["size"] == 2


def test_build_duplicate_groups_same_speaker_same_text() -> None:
    clips = pl.DataFrame(
        [
            _clip_row(
                clip_id="a",
                audio_sha256="hash1",
                source_path="a.wav",
                text_norm="同じ",
                speaker_id="spk1",
            ),
            _clip_row(
                clip_id="b",
                audio_sha256="hash2",
                source_path="b.wav",
                text_norm="同じ",
                speaker_id="spk1",
            ),
            _clip_row(
                clip_id="c",
                audio_sha256="hash3",
                source_path="c.wav",
                text_norm="同じ",
                speaker_id="spk2",
            ),
        ]
    )
    groups = build_duplicate_groups(clips)
    same_speaker_text = groups.filter(pl.col("kind") == "same_speaker_same_text")
    assert same_speaker_text.height == 1
    assert sorted(same_speaker_text.row(0, named=True)["clip_ids"]) == ["a", "b"]
