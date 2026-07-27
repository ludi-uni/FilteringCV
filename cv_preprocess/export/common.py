"""Shared helpers for trainer exports."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from cv_preprocess.export.protocol import TextFieldName, PlaceMode, UtteranceRow
from cv_preprocess.pipeline.export import write_wav_16bit

logger = logging.getLogger(__name__)


def load_metadata_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def resolve_text(row: dict[str, Any], text_field: TextFieldName) -> str:
    primary = str(row.get(text_field) or "").strip()
    if primary:
        return primary
    fallback_key = "text_raw" if text_field == "text_norm" else "text_norm"
    return str(row.get(fallback_key) or "").strip()


def utterances_from_metadata(
    rows: list[dict[str, Any]],
    *,
    materialize_root: Path,
    text_field: TextFieldName,
) -> tuple[list[UtteranceRow], list[str], int]:
    """Build utterance list; skip empty text; fail on missing audio."""
    materialize_root = Path(materialize_root)
    out: list[UtteranceRow] = []
    warnings: list[str] = []
    skipped = 0
    for row in rows:
        clip_id = str(row.get("clip_id") or row.get("utt_id") or "").strip()
        if not clip_id:
            warnings.append("skipped metadata row without clip_id/utt_id")
            skipped += 1
            continue
        text = resolve_text(row, text_field)
        if not text:
            warnings.append(f"skipped empty text for clip_id={clip_id}")
            skipped += 1
            continue
        audio_rel = str(row.get("audio_path") or "").strip()
        if not audio_rel:
            raise ValueError(f"metadata row for {clip_id!r} has no audio_path")
        source = materialize_root / audio_rel
        if not source.is_file():
            raise FileNotFoundError(f"audio missing for {clip_id}: {source}")
        speaker = str(row.get("speaker_id") or "speaker").strip() or "speaker"
        out.append(
            UtteranceRow(
                clip_id=clip_id,
                speaker_id=speaker,
                text=text,
                source_audio=source,
                split=str(row["split"]) if row.get("split") is not None else None,
            )
        )
    return out, warnings, skipped


def place_audio(src: Path, dst: Path, mode: PlaceMode, *, resample_hz: int | None) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if resample_hz is None:
        try:
            if mode == "copy":
                shutil.copy2(src, dst)
            elif mode == "hardlink":
                os.link(src, dst)
            elif mode == "symlink":
                os.symlink(Path(src).resolve(), dst)
            else:
                raise ValueError(f"unsupported place mode: {mode!r}")
        except OSError:
            shutil.copy2(src, dst)
        return
    _resample_to_wav(src, dst, int(resample_hz))


def _resample_to_wav(src: Path, dst: Path, sample_rate: int) -> None:
    import soundfile as sf

    y, sr = sf.read(str(src), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    src_sr = int(sr)
    if src_sr != sample_rate:
        import librosa

        y = librosa.resample(y, orig_sr=src_sr, target_sr=sample_rate)
    write_wav_16bit(dst, y, sample_rate)


def sanitize_pipe_text(text: str) -> str:
    return " ".join(text.replace("|", " ").replace("\n", " ").replace("\r", " ").split())
