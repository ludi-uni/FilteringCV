"""リップノイズ（短い口内クリック）向け: 瞬間検出 + 局所波形補間。

短時間 RMS・ゼロ交差率・スペクトルフラックス・クレスト係数で「短い異常区間」を拾い、
検出区間（パディング付き）だけを前後サンプル間の補間で置き換える。帯域全体を潰しにくい。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import medfilt

from cv_preprocess.audio.binary_runs import short_runs_only as _short_runs_only


def apply_lip_noise_repair(
    y: np.ndarray,
    sr: int,
    *,
    frame_ms: float = 2.5,
    hop_ms: float = 0.65,
    median_kernel_ms: float = 9.0,
    rms_ratio_threshold: float = 5.5,
    zcr_ratio_threshold: float = 2.6,
    crest_factor_threshold: float = 6.0,
    #: ``None`` は内蔵のフラックス閾値（既定 3.5）で ZCR と OR 判定。``0`` 以下でフラックス条件を無効化（ZCR のみ）。
    flux_ratio_threshold: float | None = None,
    max_event_ms: float = 24.0,
    merge_gap_ms: float = 2.5,
    repair_pad_ms: float = 1.8,
    max_repair_ms: float = 28.0,
    interpolation: Literal["linear", "cubic"] = "linear",
    max_repairs_per_clip: int = 96,
    fft_bins: int = 256,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """波形上の短い異常トランジェントを検出し、局所補間で修復する。

    Returns
    -------
    y_out
        修復後のモノラル float32。
    meta
        ``lip_noise_repair_events``（マージ後のイベント数）,
        ``lip_noise_repair_samples``（置換したサンプル数の合計）。
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError("apply_lip_noise_repair expects mono 1-D audio")
    n = int(y.size)
    if n < 32 or sr < 4000:
        return y.copy(), {"lip_noise_repair_events": 0, "lip_noise_repair_samples": 0}

    win = max(16, int(round(frame_ms * sr / 1000.0)))
    hop = max(4, int(round(hop_ms * sr / 1000.0)))
    if n < win + hop:
        return y.copy(), {"lip_noise_repair_events": 0, "lip_noise_repair_samples": 0}

    win_fft = int(np.clip(fft_bins, 64, 1024))
    if win_fft < win:
        win_fft = int(min(1024, max(64, win)))
    win_fft = win_fft - (win_fft % 2)  # even for rfft efficiency

    # (n_frames, win)
    vw = sliding_window_view(y.astype(np.float64), win)[::hop, :]
    n_frames = int(vw.shape[0])
    if n_frames < 5:
        return y.astype(np.float32).copy(), {"lip_noise_repair_events": 0, "lip_noise_repair_samples": 0}

    rms = np.sqrt(np.mean(vw * vw, axis=1) + 1e-20)
    crest = (np.max(np.abs(vw), axis=1) / rms).astype(np.float64)

    sgn = np.sign(vw)
    sgn[sgn == 0] = 1.0
    zc = np.mean(np.abs(np.diff(sgn, axis=1)) > 1e-15, axis=1).astype(np.float64)

    # Spectral flux on same windows (even length for rfft)
    wfft = min(win_fft, win)
    tail = vw[:, -wfft:]
    mag = np.abs(np.fft.rfft(tail, n=wfft, axis=1))
    flux = np.zeros(n_frames, dtype=np.float64)
    if mag.shape[0] >= 2:
        d = np.diff(mag, axis=0)
        fl = np.sum(np.maximum(0.0, d), axis=1)
        flux[1:] = fl

    k_med = int(round(median_kernel_ms / hop_ms)) | 1
    k_med = int(np.clip(k_med, 3, 201))
    odd_cap = n_frames if (n_frames % 2 == 1) else n_frames - 1
    k_med = max(3, min(k_med, odd_cap))

    rms_med = medfilt(rms, kernel_size=k_med) + 1e-20
    zc_med = medfilt(zc, kernel_size=k_med) + 1e-20
    fl_med = medfilt(flux, kernel_size=k_med) + 1e-20

    rr = rms / rms_med
    zr = zc / (zc_med + 1e-20)
    fr = flux / (fl_med + 1e-20)

    base = (rr >= float(rms_ratio_threshold)) & (crest >= float(crest_factor_threshold))
    if flux_ratio_threshold is None:
        flux_thr = 3.5
        transient_shape = (zr >= float(zcr_ratio_threshold)) | (fr >= flux_thr)
    elif float(flux_ratio_threshold) <= 0.0:
        transient_shape = zr >= float(zcr_ratio_threshold)
    else:
        flux_thr = float(flux_ratio_threshold)
        transient_shape = (zr >= float(zcr_ratio_threshold)) | (fr >= flux_thr)
    cand = base & transient_shape

    max_f = max(1, int(round(max_event_ms / hop_ms)))
    cand = _short_runs_only(cand, max_f)

    if not np.any(cand):
        return y.astype(np.float32).copy(), {"lip_noise_repair_events": 0, "lip_noise_repair_samples": 0}

    # Frame indices -> sample intervals [s, e)
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        raw_se: list[tuple[int, int]] = []
    else:
        fi = idx.astype(np.int64, copy=False)
        s_arr = fi * hop
        e_arr = np.minimum(n, fi * hop + win)
        raw_se = list(zip(s_arr.tolist(), e_arr.tolist()))

    merge_gap = max(0, int(round(merge_gap_ms * sr / 1000.0)))
    merged = _merge_intervals(raw_se, gap=merge_gap)

    pad = max(0, int(round(repair_pad_ms * sr / 1000.0)))
    max_w = max(win, int(round(max_repair_ms * sr / 1000.0)))
    max_span = max(win, int(round(max_event_ms * sr / 1000.0)))

    events: list[tuple[int, int]] = []
    for s, e in merged:
        if e - s > max_span:
            sub = np.asarray(y[s:e], dtype=np.float64)
            peak_rel = int(np.argmax(np.abs(sub)))
            c = s + peak_rel
            half = max_span // 2
            s = max(0, c - half)
            e = min(n, s + max_span)
            if e - s < max_span:
                s = max(0, e - max_span)
        s2 = max(0, s - pad)
        e2 = min(n, e + pad)
        if e2 <= s2:
            continue
        if e2 - s2 > max_w:
            c = (s2 + e2) // 2
            h = max_w // 2
            s2 = max(0, c - h)
            e2 = min(n, s2 + max_w)
        events.append((s2, e2))

    events = _merge_intervals(events, gap=0)
    events.sort(key=lambda x: x[0])
    if max_repairs_per_clip > 0 and len(events) > max_repairs_per_clip:
        # Keep strongest by peak |y| inside each interval
        scores = []
        for s, e in events:
            seg = y[s:e]
            scores.append(float(np.max(np.abs(seg)) if seg.size else 0.0))
        order = np.argsort(-np.asarray(scores, dtype=np.float64))[:max_repairs_per_clip]
        events = [events[int(i)] for i in order]
        events.sort(key=lambda x: x[0])

    y_out = y.astype(np.float32).copy()
    total_samples = 0
    for s, e in reversed(events):
        if e <= s or s <= 0 and e >= n:
            continue
        total_samples += _repair_interval_inplace(
            y_out, s, e, interpolation=interpolation
        )

    return y_out, {
        "lip_noise_repair_events": int(len(events)),
        "lip_noise_repair_samples": int(total_samples),
    }


def _merge_intervals(intervals: list[tuple[int, int]], *, gap: int) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda t: t[0])
    out: list[tuple[int, int]] = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce + gap:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


def _repair_interval_inplace(
    y: np.ndarray,
    s: int,
    e: int,
    *,
    interpolation: Literal["linear", "cubic"],
) -> int:
    """Replace y[s:e] in-place. Returns number of samples written."""
    n = int(y.size)
    if e <= s or s < 0 or e > n:
        return 0
    hole = e - s
    if hole <= 0:
        return 0

    if interpolation == "linear":
        left = float(y[s - 1]) if s > 0 else float(y[e] if e < n else 0.0)
        right = float(y[e]) if e < n else float(y[s - 1] if s > 0 else left)
        y[s:e] = np.linspace(left, right, hole + 2, dtype=np.float32)[1:-1]
        return hole

    # cubic: need enough support points; else fall back to linear
    from scipy.interpolate import CubicSpline

    xs: list[float] = []
    vals: list[float] = []
    if s >= 2:
        xs.extend([float(s - 2), float(s - 1)])
        vals.extend([float(y[s - 2]), float(y[s - 1])])
    elif s >= 1:
        xs.append(float(s - 1))
        vals.append(float(y[s - 1]))
    if e < n - 1:
        xs.extend([float(e), float(e + 1)])
        vals.extend([float(y[e]), float(y[e + 1])])
    elif e < n:
        xs.append(float(e))
        vals.append(float(y[e]))

    if len(xs) < 4:
        left = float(y[s - 1]) if s > 0 else float(y[e] if e < n else 0.0)
        right = float(y[e]) if e < n else left
        y[s:e] = np.linspace(left, right, hole + 2, dtype=np.float32)[1:-1]
        return hole

    x_arr = np.asarray(xs, dtype=np.float64)
    v_arr = np.asarray(vals, dtype=np.float64)
    cs = CubicSpline(x_arr, v_arr)
    xi = np.arange(float(s), float(e), dtype=np.float64)
    y[s:e] = cs(xi).astype(np.float32)
    return hole
