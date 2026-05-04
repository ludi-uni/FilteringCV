from __future__ import annotations

import numpy as np
from scipy import signal


def butter_lowpass(y: np.ndarray, sr: int, cutoff_hz: float, order: int = 6) -> np.ndarray:
    nyq = 0.5 * sr
    wn = min(0.99, cutoff_hz / nyq)
    b, a = signal.butter(order, wn, btype="low")
    return signal.filtfilt(b, a, y.astype(np.float64)).astype(np.float32)


def butter_highpass(y: np.ndarray, sr: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    wn = max(cutoff_hz / nyq, 1e-4)
    b, a = signal.butter(order, wn, btype="high")
    return signal.filtfilt(b, a, y.astype(np.float64)).astype(np.float32)
