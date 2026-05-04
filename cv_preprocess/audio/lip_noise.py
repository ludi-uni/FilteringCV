"""口内音（リップ／唾液のパチッ等）向けの軽量 STFT 抑制。

短いフレームだけエネルギーが局所中央値に対して突出し、かつ中高域に成分がある場合に
帯域内マグニチュードだけを下げる。子音全体を落としにくいよう **連続フレーム数の上限** で制限する。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

from cv_preprocess.audio.binary_runs import short_runs_only as _short_runs_only


def apply_lip_noise_suppress(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    band_low_hz: float = 1400.0,
    band_high_hz: float | None = None,
    spike_ratio: float = 7.0,
    max_burst_frames: int = 6,
    mag_gain: float = 0.52,
    median_kernel_frames: int = 11,
    temporal_smooth_frames: int = 3,
) -> np.ndarray:
    """リップノイズっぽい短バーストを中高域中心に弱める。``mag_gain`` は該当帯の倍率（0〜1）。"""
    import librosa

    y = np.asarray(y, dtype=np.float32).copy()
    if y.size < hop_length + 16:
        return y.astype(np.float32)

    n_fft = int(n_fft)
    hop = int(hop_length)
    win = n_fft

    D = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window="hann",
        center=True,
        dtype=np.complex64,
    )
    mag = np.abs(D).astype(np.float64)
    phase = np.angle(D)
    n_bins, n_frames = mag.shape
    if n_frames < 4:
        return y.astype(np.float32)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    f_hi = float(sr) / 2.0 if band_high_hz is None else float(band_high_hz)
    band_lo = float(band_low_hz)
    bin_lo = int(np.searchsorted(freqs, band_lo, side="left"))
    bin_hi = int(np.searchsorted(freqs, f_hi, side="right"))
    bin_lo = int(np.clip(bin_lo, 0, n_bins - 1))
    bin_hi = int(np.clip(bin_hi, bin_lo + 1, n_bins))

    e_full = np.mean(mag**2, axis=0) + 1e-20
    e_band = np.mean(mag[bin_lo:bin_hi, :] ** 2, axis=0) + 1e-20
    # 中高域にエネルギーが無いバーストは子音以外の可能性が高いので対象外
    hf_frac = e_band / e_full

    k_med = int(median_kernel_frames) | 1
    k_med = int(np.clip(k_med, 3, 65))
    odd_max = n_frames if (n_frames % 2 == 1) else n_frames - 1
    k_med = max(3, min(k_med, odd_max))
    e_med = medfilt(e_full.astype(np.float64), kernel_size=k_med) + 1e-20
    ratio = e_full / e_med

    spike = ratio >= float(spike_ratio)
    hf_frac_med = medfilt(hf_frac.astype(np.float64), kernel_size=k_med) + 1e-20
    hf_ok = hf_frac >= np.maximum(0.22, hf_frac_med * 1.35)
    raw = spike & hf_ok

    max_burst = int(max_burst_frames)
    max_burst = int(np.clip(max_burst, 1, 64))
    keep = _short_runs_only(raw, max_burst)
    if not np.any(keep):
        return y.astype(np.float32)

    k_smooth = int(temporal_smooth_frames) | 1
    k_smooth = int(np.clip(k_smooth, 1, 31))
    g = _smooth_mask(keep.astype(np.float64), k_smooth)
    g = np.clip(g, 0.0, 1.0)

    mg = float(np.clip(mag_gain, 0.08, 0.98))
    w = g[np.newaxis, :]
    mid_att = mag[bin_lo:bin_hi, :] * (1.0 - w * (1.0 - mg))
    parts: list[np.ndarray] = []
    if bin_lo > 0:
        parts.append(mag[:bin_lo, :])
    parts.append(mid_att)
    if bin_hi < n_bins:
        parts.append(mag[bin_hi:, :])
    mag_out = np.concatenate(parts, axis=0)

    D_out = (mag_out * np.exp(1j * phase)).astype(np.complex64)
    y_out = librosa.istft(
        D_out,
        hop_length=hop,
        n_fft=n_fft,
        win_length=win,
        window="hann",
        center=True,
        length=y.size,
    )
    if y_out.size != y.size:
        y_out = np.resize(y_out, y.shape)
    return y_out.astype(np.float32)


def _smooth_mask(x: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return x
    k = np.ones(kernel, dtype=np.float64) / float(kernel)
    return np.convolve(x, k, mode="same")
