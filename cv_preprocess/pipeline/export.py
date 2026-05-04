from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def write_wav_16bit(path: Path, y: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_f = np.asarray(y, dtype=np.float32)
    if y_f.ndim > 1:
        y_f = np.mean(y_f, axis=0)
    y_f = np.ascontiguousarray(y_f.reshape(-1))
    sr_i = int(sr)
    if sr_i <= 0:
        raise ValueError(f"Invalid sample rate for WAV: {sr!r}")
    if y_f.size == 0:
        raise ValueError("Cannot write empty WAV")
    y_clip = np.clip(y_f, -1.0, 1.0)
    if not np.isfinite(y_clip).all():
        raise ValueError("Non-finite samples in WAV payload")
    data_i16 = (y_clip * 32767.0).astype(np.int16)
    # 中断で 0 バイトの残骸や特殊ファイルがあると libsndfile が失敗することがある
    path.unlink(missing_ok=True)
    sf.write(
        str(path.resolve()),
        data_i16,
        sr_i,
        subtype="PCM_16",
        format="WAV",
    )


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_reject_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)
