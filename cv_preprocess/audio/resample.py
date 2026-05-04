from __future__ import annotations

import numpy as np


def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
