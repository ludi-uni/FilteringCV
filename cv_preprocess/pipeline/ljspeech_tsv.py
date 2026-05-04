from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path


def iter_metadata_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _ljspeech_tsv_row(rec: dict) -> list[str]:
    audio_path = rec.get("audio_path")
    text_norm = rec.get("text_norm")
    if audio_path is None or text_norm is None:
        raise ValueError(
            "metadata record must include audio_path and text_norm "
            f"(got keys: {sorted(rec.keys())})"
        )
    text_raw = rec.get("text_raw")
    if text_raw is None:
        text_raw = text_norm
    return [audio_path, text_norm, text_raw]


def write_ljspeech_validated_tsv(path: Path, records: Iterable[dict]) -> None:
    """Write UTF-8 TSV (no header) for VITS / LJSpeech-style loaders.

    Each row: ``audio_path`` ``text_norm`` ``text_raw`` (tab-separated).
    ``audio_path`` is relative to the corpus output root (e.g. ``wavs/cv_ja_000001.wav``),
    same as ``metadata.jsonl`` field ``audio_path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for rec in records:
            w.writerow(_ljspeech_tsv_row(rec))


def metadata_jsonl_to_validated_tsv(src: Path, dst: Path) -> int:
    """Convert ``metadata.jsonl`` to the same ``validated.tsv`` layout as preprocess."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as inf, dst.open("w", encoding="utf-8", newline="") as outf:
        w = csv.writer(outf, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for line in inf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            w.writerow(_ljspeech_tsv_row(rec))
            n += 1
    return n
