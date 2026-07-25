"""Indexer progress emission and incremental reuse."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from cv_preprocess.application.common import ProgressEvent
from cv_preprocess.config.pipeline import PipelineConfig
from cv_preprocess.coverage.indexer import build_clip_index
from cv_preprocess.coverage.models import CheapQuality, ClipIndexMeta, ClipIndexRecord
from cv_preprocess.io.tsv_loader import ClipRow
from cv_preprocess.reports.serializer import write_json_atomic


def _minimal_config(tmp_path: Path, rows: list[tuple[str, str, str]]) -> PipelineConfig:
    corpus = tmp_path / "corpus"
    clips = corpus / "clips"
    clips.mkdir(parents=True)
    tsv = corpus / "validated.tsv"
    lines = ["client_id\tpath\tsentence"]
    for client_id, rel, sentence in rows:
        audio = clips / rel
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"RIFF....")
        lines.append(f"{client_id}\t{rel}\t{sentence}")
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PipelineConfig.model_validate(
        {
            "input": {
                "corpus_root": str(corpus),
                "clip_tsv": "validated.tsv",
                "audio_subdir": "clips",
            },
            "coverage": {"enabled": True},
            "dataset_builder": {"enabled": True, "work_dir": str(tmp_path / "work")},
        }
    )


def test_build_clip_index_emits_progress(tmp_path: Path) -> None:
    cfg = _minimal_config(
        tmp_path,
        [
            ("s1", "a.wav", "こんにちは"),
            ("s1", "b.wav", "ありがとう"),
        ],
    )
    events: list[ProgressEvent] = []

    fake = ClipIndexRecord(
        clip_id="c1",
        source_path="a.wav",
        client_id="s1",
        sentence="こんにちは",
        normalized_text="こんにちは",
        phonemes=["k", "o"],
        unique_phonemes=["k", "o"],
        moras=[],
        biphones=[],
        positioned_phonemes=[],
        cheap_quality=CheapQuality(decode_ok=False),
        source_row_index=0,
        audio_sha256="abc",
        feature_keys=["phoneme:k"],
    )

    def _fake_build(row, **kwargs):  # noqa: ANN001
        idx = kwargs["source_row_index"]
        return fake.model_copy(
            update={
                "clip_id": f"c{idx}",
                "source_path": row.path,
                "sentence": row.sentence,
                "normalized_text": row.sentence,
                "source_row_index": idx,
            }
        )

    with patch("cv_preprocess.coverage.indexer.build_index_record", side_effect=_fake_build):
        with patch("cv_preprocess.coverage.indexer._sha256_file", return_value="abc"):
            result = build_clip_index(
                cfg,
                output=tmp_path / "clip-index.jsonl",
                decode_audio=False,
                progress=events.append,
            )

    assert result.clip_count == 2
    assert any(e.message == "indexing clips" and e.current is not None for e in events)
    assert any(e.message == "index complete" for e in events)
    assert events[-1].fraction == 1.0


def test_incremental_reuses_without_rebuild(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path, [("s1", "a.wav", "こんにちは")])
    out = tmp_path / "clip-index.jsonl"

    record = ClipIndexRecord(
        clip_id="keep-me",
        source_path="a.wav",
        client_id="s1",
        sentence="こんにちは",
        normalized_text="こんにちは",
        phonemes=["k"],
        unique_phonemes=["k"],
        moras=[],
        biphones=[],
        positioned_phonemes=[],
        cheap_quality=CheapQuality(decode_ok=True),
        source_row_index=0,
        audio_sha256="same-hash",
        feature_keys=["phoneme:k"],
    )
    out.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    write_json_atomic(
        out.with_name("clip-index.meta.json"),
        ClipIndexMeta(
            created_at="t",
            source_fingerprint="fp",
            normalizer_version="n",
            g2p_version="g",
            config_hash="h",
            clip_count=1,
            incremental=False,
        ),
    )

    calls = {"build": 0}

    def _boom(*_a, **_k):  # noqa: ANN001
        calls["build"] += 1
        raise AssertionError("build_index_record should not run on reuse")

    with patch("cv_preprocess.coverage.indexer.build_index_record", side_effect=_boom):
        with patch("cv_preprocess.coverage.indexer._sha256_file", return_value="same-hash"):
            result = build_clip_index(
                cfg,
                output=out,
                incremental=True,
                force=False,
                decode_audio=False,
            )

    assert calls["build"] == 0
    assert result.clip_count == 1
    assert result.meta.incremental is True


def test_reuse_helper_requires_matching_hash(tmp_path: Path) -> None:
    from cv_preprocess.coverage.indexer import _reuse_existing_record

    root = tmp_path
    clips = root / "clips"
    clips.mkdir()
    audio = clips / "a.wav"
    audio.write_bytes(b"x")
    row = ClipRow(client_id="s", path="a.wav", sentence="hi", raw={})
    old = ClipIndexRecord(
        clip_id="id",
        source_path="a.wav",
        client_id="s",
        sentence="hi",
        normalized_text="hi",
        phonemes=[],
        unique_phonemes=[],
        moras=[],
        biphones=[],
        positioned_phonemes=[],
        cheap_quality=CheapQuality(decode_ok=False),
        source_row_index=0,
        audio_sha256="old",
        feature_keys=[],
    )
    existing = {("a.wav", 0): old}
    with patch("cv_preprocess.coverage.indexer._sha256_file", return_value="new"):
        assert (
            _reuse_existing_record(
                row,
                source_row_index=0,
                root=root,
                audio_subdir="clips",
                existing_by_row=existing,
            )
            is None
        )
    with patch("cv_preprocess.coverage.indexer._sha256_file", return_value="old"):
        reused = _reuse_existing_record(
            row,
            source_row_index=0,
            root=root,
            audio_subdir="clips",
            existing_by_row=existing,
        )
        assert reused is old
