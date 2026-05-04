from __future__ import annotations

import warnings

import numpy as np


def normalize_peak(y: np.ndarray, peak_dbfs: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(y)) + 1e-12)
    target = 10.0 ** (peak_dbfs / 20.0)
    g = target / peak
    g = float(np.clip(g, 0.0, 100.0))
    return (y * g).astype(np.float32)


def normalize_loudness(y: np.ndarray, sr: int, integrated_lufs: float = -23.0) -> np.ndarray:
    import pyloudnorm as pyln

    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    # integrated_loudness は極短信号で内部が空スライスの mean になり RuntimeWarning になりやすい
    min_samples = max(1, int(float(sr) * 0.4))
    if y.size < min_samples:
        return normalize_peak(y, peak_dbfs=-1.0)

    try:
        meter = pyln.Meter(sr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loudness = meter.integrated_loudness(y.astype(np.float64))
        if not np.isfinite(loudness):
            return normalize_peak(y, peak_dbfs=-1.0)
        gain_db = integrated_lufs - loudness
        gain_db = float(np.clip(gain_db, -60.0, 60.0))
        gain = 10.0 ** (gain_db / 20.0)
        return (y * gain).astype(np.float32)
    except Exception:
        return normalize_peak(y, peak_dbfs=-1.0)
