from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(path: Path, *, mono: bool = True) -> tuple[np.ndarray, int]:
    """Decode to float32 mono/stereo and native SR."""
    import librosa

    y, sr = librosa.load(path, sr=None, mono=mono)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    y = np.asarray(y, dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("nan_inf_audio")
    return y, int(sr)
