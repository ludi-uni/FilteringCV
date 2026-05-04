from pathlib import Path

import pytest

from cv_preprocess.io.alignment_phoneme_manifest import load_alignment_phoneme_manifest


def test_load_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text(
        '{"source_path": "x.mp3", "phonemes": "k o N n i ch i w a"}\n'
        '{"source_path": "y.mp3", "phonemes": "a i u"}\n',
        encoding="utf-8",
    )
    m = load_alignment_phoneme_manifest(p)
    assert m["x.mp3"] == "k o N n i ch i w a"
    assert m["y.mp3"] == "a i u"


def test_load_duplicate_last_wins(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text(
        '{"source_path": "x.mp3", "phonemes": "a b"}\n'
        '{"source_path": "x.mp3", "phonemes": "c d"}\n',
        encoding="utf-8",
    )
    m = load_alignment_phoneme_manifest(p)
    assert m["x.mp3"] == "c d"


def test_load_invalid_line(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_alignment_phoneme_manifest(p)
