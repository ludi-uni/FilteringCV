import json
from pathlib import Path

from cv_preprocess.pipeline.ljspeech_tsv import (
    metadata_jsonl_to_validated_tsv,
    write_ljspeech_validated_tsv,
)


def test_write_ljspeech_validated_tsv_three_columns_no_header(tmp_path: Path) -> None:
    out = tmp_path / "validated.tsv"
    write_ljspeech_validated_tsv(
        out,
        [
            {
                "audio_path": "wavs/cv_ja_000001.wav",
                "text_norm": "あ い う",
                "text_raw": "あいう",
            }
        ],
    )
    raw = out.read_text(encoding="utf-8")
    lines = raw.splitlines()
    assert len(lines) == 1
    parts = lines[0].split("\t")
    assert parts == ["wavs/cv_ja_000001.wav", "あ い う", "あいう"]


def test_write_falls_back_text_raw_to_text_norm(tmp_path: Path) -> None:
    out = tmp_path / "validated.tsv"
    write_ljspeech_validated_tsv(
        out,
        [{"audio_path": "wavs/x.wav", "text_norm": "only"}],
    )
    parts = out.read_text(encoding="utf-8").strip().split("\t")
    assert parts == ["wavs/x.wav", "only", "only"]


def test_metadata_jsonl_round_trip(tmp_path: Path) -> None:
    meta = tmp_path / "metadata.jsonl"
    meta.write_text(
        json.dumps(
            {
                "audio_path": "wavs/a.wav",
                "text_norm": "n",
                "text_raw": "r",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    dst = tmp_path / "validated.tsv"
    n = metadata_jsonl_to_validated_tsv(meta, dst)
    assert n == 1
    assert dst.read_text(encoding="utf-8").strip() == "wavs/a.wav\tn\tr"
