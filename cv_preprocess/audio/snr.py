from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def estimate_snr_db(
    y: np.ndarray,
    sr: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    noise_percentile: float = 15.0,
    signal_percentile: float = 60.0,
    min_frames: int = 4,
) -> float | None:
    """Frame energy SNR: 10 log10(P_signal / P_noise) with percentile noise/signal floors."""
    if y.size == 0:
        return None
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if y.size < frame:
        return None
    n_frames = 1 + (y.size - frame) // hop
    if n_frames < min_frames:
        return None
    y64 = np.asarray(y, dtype=np.float64)
    vw = sliding_window_view(y64, frame)[::hop, :]
    e_arr = np.mean(vw * vw, axis=1) + 1e-12
    if int(e_arr.size) < min_frames:
        return None
    p_n = float(np.percentile(e_arr, noise_percentile))
    p_s = float(np.percentile(e_arr, signal_percentile))
    if p_n <= 0 or p_s <= p_n:
        return None
    return float(10.0 * np.log10(p_s / p_n))
