from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class EdgeClickResult:
    y: np.ndarray
    removed_leading_ms: float
    removed_trailing_ms: float
    confidence_leading: str
    confidence_trailing: str


_CONSISTENT_MAD = 1.4826


def _frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """hop サンプル刻みの短時間 RMS。``1 + (len - frame) // hop`` 本（末尾窓を含む）。"""
    if x.size < frame:
        return np.array([], dtype=np.float64)
    x64 = np.asarray(x, dtype=np.float64)
    vw = sliding_window_view(x64, frame)[::hop, :]
    return np.sqrt(np.mean(vw * vw, axis=1) + 1e-18).astype(np.float64)


def _mad_threshold(x: np.ndarray, k: float) -> float:
    m = float(np.median(x))
    mad = float(np.median(np.abs(x - m))) + 1e-20
    return m + float(k) * _CONSISTENT_MAD * mad


def _contiguous_pre_peak_low_rms_ms(
    rms: np.ndarray,
    peak_i: int,
    hop: int,
    sr: int,
    thresh: float,
) -> float:
    """RMS ピーク ``peak_i`` より左で、``thresh * 0.45`` 未満が連続する長さ（ms）。"""
    if peak_i <= 0:
        return 0.0
    lo = float(thresh) * 0.45
    k = peak_i - 1
    nfr = 0
    while k >= 0 and float(rms[k]) < lo:
        nfr += 1
        k -= 1
    return 1000.0 * nfr * hop / float(sr)


def _contiguous_pre_burst_silence_ms(
    rms: np.ndarray,
    start_f: int,
    hop: int,
    sr: int,
    nf: float,
    thresh: float,
) -> float:
    """バースト開始フレーム ``start_f`` 直前から遡る連続無音の長さ（ms）。"""
    if start_f <= 0:
        return 0.0
    pre = rms[:start_f]
    if pre.size == 0:
        return 0.0
    pct = float(np.percentile(pre, 12.0))
    sil_floor = float(min(pct, thresh * 0.42))
    sil_floor = max(sil_floor, nf * 0.35)
    thr2 = float(sil_floor) * 1.18
    lt = pre < thr2
    rev = lt[::-1]
    inv = ~rev
    nfr = int(rev.size if not np.any(inv) else np.argmax(inv))
    return 1000.0 * nfr * hop / float(sr)


def _silence_prefix_end_sample(
    tail: np.ndarray,
    frame: int,
    hop: int,
    top_db: float,
) -> int:
    """クリック除去後の ``tail`` 先頭から、``top_db`` 基準で無音とみなせるサンプル数（先頭は削る）。"""
    rms = _frame_rms(tail, frame, hop)
    if rms.size == 0:
        return 0
    ref = float(np.percentile(rms, 93.0)) + 1e-12
    thr = ref * (10.0 ** (-float(top_db) / 20.0))
    ge = rms >= thr
    onset_f: int | None = None
    if ge.size >= 2:
        pairs = ge[1:] & ge[:-1]
        idx = np.flatnonzero(pairs)
        if idx.size > 0:
            onset_f = int(idx[0])
    if onset_f is None:
        idx2 = np.flatnonzero(ge)
        if idx2.size == 0:
            return 0
        onset_f = int(idx2[0])
    return int(onset_f * hop)


def _speech_runs_sustained(
    rms: np.ndarray,
    *,
    mad_k: float,
    min_run_frames: int,
) -> list[tuple[int, int]]:
    """RMS がロバスト閾値以上が ``min_run_frames`` 以上続く区間 [f0,f1]（フレーム、両端含む）。"""
    if rms.size == 0:
        return []
    med = float(np.median(rms))
    mad = float(np.median(np.abs(rms - med))) + 1e-20
    # 平坦な RMS（MAD≈0）で全体が「有声」と化さないよう、相対下限を付ける。
    mad_eff = max(mad, med * 0.11 + 1e-12, float(np.std(rms)) * 0.25 + 1e-12)
    thr = med + float(mad_k) * _CONSISTENT_MAD * mad_eff
    above = rms >= thr
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(above.size)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        if j - i >= min_run_frames:
            runs.append((i, j - 1))
        i = j
    return runs


def _estimate_speech_onset_in_prefix(
    y_prefix: np.ndarray,
    sr: int,
    *,
    frame_ms: float,
    hop_ms: float,
    speech_rms_mad_k: float,
    min_speech_run_ms: float,
) -> int:
    """先頭帯 ``y_prefix``（先頭からの切り出し）だけで持続有声の開始サンプル（0 起算・``y_prefix`` 内）。

    持続有声が取れないときは ``len(y_prefix)``（帯全体をクリック候補スキャン可）を返す。
    発話が帯の先頭から続くときは ``0``。
    """
    n = int(y_prefix.size)
    if n < 32:
        return n
    fl = max(128, int(round(sr * frame_ms / 1000.0)))
    hop = max(32, int(round(sr * hop_ms / 1000.0)))
    rms = _frame_rms(y_prefix, fl, hop)
    if rms.size == 0:
        return n
    frame_ms_eff = 1000.0 * fl / sr
    min_run_frames = max(2, int(round(min_speech_run_ms / max(frame_ms_eff * 0.35, 1e-6))))
    runs = _speech_runs_sustained(rms, mad_k=speech_rms_mad_k, min_run_frames=min_run_frames)
    if not runs:
        return n
    f0, _ = runs[0]
    onset_s = int(f0 * hop)
    return int(np.clip(onset_s, 0, n))


def _smooth_median_1d(x: np.ndarray, kernel: int) -> np.ndarray:
    k = int(kernel) | 1
    if k <= 1 or x.size < k:
        return x
    from scipy.signal import medfilt

    return medfilt(x, kernel_size=k).astype(np.float64, copy=False)


def _short_time_energy(seg: np.ndarray, win: int) -> np.ndarray:
    """各サンプル位置を窓左端とする ``sum(x^2)``（長さ ``len(seg)-win+1``）。"""
    if seg.size < win or win < 2:
        return np.array([], dtype=np.float64)
    sq = (np.asarray(seg, dtype=np.float64) ** 2).astype(np.float64, copy=False)
    vw = sliding_window_view(sq, win)
    return np.sum(vw, axis=1)


def _energy_spike_ok(
    en: np.ndarray,
    center_samp: int,
    n: int,
    win: int,
    neigh_samp: int,
    *,
    spike_ratio: float,
    baseline: str,
    ref_floor: float,
) -> bool:
    """事前計算した短窓エネルギー ``en``（``len = n - win + 1``）で近傍比を判定する。

    ``baseline`` が ``leading`` のときは左側のみ、``trailing`` のときは右側のみで
    基準を取る（発話境界の反対側の高エネルギーで落ちないようにする）。
    """
    if en.size == 0:
        return True
    c = int(np.clip(center_samp, 0, n - win))
    ei = min(c, en.size - 1)
    e_here = float(en[ei])

    lo0 = max(0, c - neigh_samp - win)
    lo1 = max(0, c - win)
    hi0 = min(en.size, c + win)
    hi1 = min(en.size, c + neigh_samp + win)
    sides: list[float] = []
    if baseline in ("both", "leading") and lo1 > lo0:
        sides.append(float(np.median(en[lo0:lo1])))
    if baseline in ("both", "trailing") and hi1 > hi0:
        sides.append(float(np.median(en[hi0:hi1])))
    if not sides:
        return e_here > 1e-12
    base = float(max(np.median(sides), ref_floor, 1e-20))
    return e_here >= base * float(spike_ratio)


def _diff_click_center_in_segment(
    seg: np.ndarray,
    sr: int,
    *,
    diff_mad_k: float,
    diff_median_smooth_samples: int,
    short_energy_ms: float,
    neighbor_energy_ms: float,
    energy_spike_ratio: float,
    max_transient_samples: int,
    energy_baseline: str = "both",
) -> tuple[int | None, str, tuple[np.ndarray, np.ndarray] | None]:
    """seg 内のクリック中心と ``(d, ds)``（``_expand_cut_from_center`` 再利用用）。

    発話境界付近の UI クリック向けに、**最も右（サンプル番号が大きい）**の合格候補を採用する。
    """
    seg = np.asarray(seg, dtype=np.float64)
    n = int(seg.size)
    m = float(np.max(np.abs(seg))) + 1e-12
    norm = seg / m
    d = np.abs(np.diff(norm))
    if d.size < 5:
        return None, "short", None
    T = _mad_threshold(d, diff_mad_k)
    if T <= 0 or not np.any(d > T):
        return None, "below_T", None

    ds = d if diff_median_smooth_samples <= 1 else _smooth_median_1d(d, diff_median_smooth_samples)

    win = max(2, int(round(sr * short_energy_ms / 1000.0)))
    neigh_samp = max(win + 1, int(round(sr * neighbor_energy_ms / 1000.0)))
    en: np.ndarray | None = None
    ref_floor = 1e-20
    if n >= win + neigh_samp * 2:
        en = _short_time_energy(seg, win)
        if en.size > 0:
            ref_floor = float(np.percentile(en, 55.0)) * 0.08 + 1e-20

    # 局所最大は平滑化後の ds で見る（ノイズ抑制）。閾値は raw d の T のまま比較。
    cand: list[int] = []
    for i in range(1, int(ds.size) - 1):
        if float(d[i]) <= T:
            continue
        if ds[i] >= ds[i - 1] and ds[i] >= ds[i + 1]:
            if ds[i] == ds[i - 1] == ds[i + 1]:
                continue
            cand.append(i)
    if not cand:
        i0 = int(np.argmax(d))
        if float(d[i0]) > T:
            cand = [i0]
        else:
            return None, "no_local_max", None

    centers = [int(i) + 1 for i in cand]

    def width_ok(center: int) -> bool:
        pk = center - 1
        if pk < 0 or pk >= int(d.size):
            return False
        lo = pk
        thr_w = max(T * 0.22, float(np.median(d)) * 0.35)
        while lo > 0 and float(d[lo - 1]) > thr_w:
            lo -= 1
        hi = pk
        while hi + 1 < int(d.size) and float(d[hi + 1]) > thr_w:
            hi += 1
        w = hi - lo + 1
        return w <= max(32, int(max_transient_samples))

    scored: list[tuple[float, int]] = []
    for c in centers:
        if not width_ok(c):
            continue
        if en is not None and not _energy_spike_ok(
            en,
            c,
            n,
            win,
            neigh_samp,
            spike_ratio=energy_spike_ratio,
            baseline=energy_baseline,
            ref_floor=ref_floor,
        ):
            continue
        pk = c - 1
        scored.append((float(d[pk]), c))

    if not scored:
        return None, "no_pass_filters", None

    c_best = max(scored, key=lambda t: t[1])[1]
    return c_best, "ok", (d, ds)


def _expand_cut_from_center(
    seg: np.ndarray,
    center: int,
    sr: int,
    *,
    diff_mad_k: float,
    diff_median_smooth_samples: int,
    max_transient_samples: int,
    d_ds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[int, int]:
    """seg 内で [s,e) をミュート／削除対象に拡張（サンプル座標）。"""
    if d_ds is not None:
        d, ds = d_ds
    else:
        norm = seg.astype(np.float64) / (float(np.max(np.abs(seg))) + 1e-12)
        d = np.abs(np.diff(norm))
        ds = d if diff_median_smooth_samples <= 1 else _smooth_median_1d(d, diff_median_smooth_samples)
    T = _mad_threshold(ds, diff_mad_k)
    thr_w = max(T * 0.2, float(np.median(ds)) * 0.45)
    pk = int(np.clip(center - 1, 0, int(ds.size) - 1))
    lo = pk
    while lo > 0 and float(ds[lo - 1]) > thr_w:
        lo -= 1
    hi = pk
    while hi + 1 < int(ds.size) and float(ds[hi + 1]) > thr_w:
        hi += 1
    s = int(lo)
    e = int(hi + 2)
    e = min(int(seg.size), max(s + 1, e))
    if e - s > max_transient_samples:
        h = max_transient_samples // 2
        c = (s + e) // 2
        s = max(0, c - h)
        e = min(int(seg.size), s + max_transient_samples)
    return s, e


def _apply_edge_fade_inplace(y: np.ndarray, s: int, e: int, fade_n: int) -> None:
    """[s,e) をゼロ寄せ。境界 ``e`` 側に ``fade_n`` サンプルで線形に寄せる。"""
    n = int(y.size)
    s = max(0, min(s, n))
    e = max(s, min(e, n))
    if e <= s:
        return
    fn = int(max(0, min(fade_n, e - s)))
    y[s : e - fn] = 0.0
    if fn > 0:
        ramp = np.linspace(1.0, 0.0, fn + 1, dtype=np.float64)[:-1].astype(np.float32)
        y[e - fn : e] *= ramp


def apply_edge_ui_click(
    y: np.ndarray,
    sr: int,
    *,
    lead_scan_ms: float,
    trail_scan_ms: float,
    max_transient_ms: float,
    peak_above_noise_db: float,
    removal: str = "mute_then_trim",
    trail_click_requires_silence_ms: float | None = None,
    lead_click_requires_pre_silence_ms: float | None = None,
    lead_post_click_top_db: float = 34.0,
    # --- diff + energy + VAD（``peak_above_noise_db`` が高いほど検出をやや厳しく寄せる）---
    diff_mad_k: float | None = None,
    diff_median_smooth_samples: int = 5,
    short_energy_ms: float = 3.0,
    neighbor_energy_ms: float = 10.0,
    energy_spike_ratio: float | None = None,
    speech_vad_frame_ms: float = 20.0,
    speech_vad_hop_ms: float = 5.0,
    speech_rms_mad_k: float | None = None,
    min_speech_run_ms: float = 50.0,
    mute_edge_fade_ms: float = 7.0,
) -> EdgeClickResult:
    """先頭・末尾の UI クリックを、差分のロバスト閾値 + 短窓エネルギーで除去する。

    * **先頭**: **先頭 ``lead_scan_ms`` だけ**を切り出し、その中の短時間 RMS で発話開始を推定し、
      その**より前**だけを差分スキャンする（帯内に持続有声が無いときは帯全体をスキャン）。
    * **末尾**: **最後の ``trail_scan_ms``** のみをスキャンし、録音停止付近の孤立スパイクを除去する。
    * ``|diff|``（振幅正規化後）で ``median + k·MAD`` を閾値にし、平滑化後の波形で**局所最大**を候補にする。
    * 短窓エネルギーは先頭では**左近傍のみ**、末尾では**右近傍のみ**と比較し、破裂直後の高エネルギーで落ちないようにする。
    * ``mute`` / ``mute_then_trim`` ではゼロ埋め境界に **``mute_edge_fade_ms``** の短フェードをかける。
    """
    y = np.asarray(y, dtype=np.float32).copy()
    n = int(y.size)
    frame = max(1, int(sr * 0.002))
    hop = max(1, frame // 2)

    db = float(peak_above_noise_db)
    d_k = float(7.5 + db * 0.22) if diff_mad_k is None else float(diff_mad_k)
    e_ratio = float(3.4 + db * 0.11) if energy_spike_ratio is None else float(energy_spike_ratio)
    sp_mad = float(6.2 + db * 0.06) if speech_rms_mad_k is None else float(speech_rms_mad_k)

    lead_n = min(n, int(sr * lead_scan_ms / 1000.0))
    trail_n = min(n, int(sr * trail_scan_ms / 1000.0))
    max_t = max(2, int(sr * max_transient_ms / 1000.0))
    fade_n = max(0, int(round(sr * mute_edge_fade_ms / 1000.0)))

    rem_l_ms = 0.0
    rem_t_ms = 0.0
    conf_l = "none"
    conf_t = "none"

    y_lead = y[:lead_n]
    onset_in_lead = _estimate_speech_onset_in_prefix(
        y_lead,
        sr,
        frame_ms=speech_vad_frame_ms,
        hop_ms=speech_vad_hop_ms,
        speech_rms_mad_k=sp_mad,
        min_speech_run_ms=min_speech_run_ms,
    )
    onset_s = int(onset_in_lead)

    # --- Leading: 発話開始より前 × 先頭 lead_n（推定も lead_n 内のみ）---
    if onset_s <= 0:
        lead_end = 0
    elif onset_s >= lead_n:
        lead_end = min(lead_n, n)
    else:
        lead_end = min(lead_n, int(onset_s), n)
    if lead_end >= 16:
        seg = np.asarray(y[:lead_end], dtype=np.float64)
        center, reason, d_ds = _diff_click_center_in_segment(
            seg,
            sr,
            diff_mad_k=d_k,
            diff_median_smooth_samples=diff_median_smooth_samples,
            short_energy_ms=short_energy_ms,
            neighbor_energy_ms=neighbor_energy_ms,
            energy_spike_ratio=e_ratio,
            max_transient_samples=max_t,
            energy_baseline="leading",
        )
        if center is None:
            conf_l = reason if reason != "ok" else "uncertain"
        else:
            rms_lead = _frame_rms(y[:lead_end], frame, hop)
            thresh = float(np.median(rms_lead) + 1e-8) * (10.0 ** (db / 20.0)) if rms_lead.size else 1.0
            peak_i = min(int(center // hop), int(rms_lead.size) - 1)
            if peak_i < 0:
                peak_i = 0
            do_lead = True
            if lead_click_requires_pre_silence_ms is not None:
                pre_ms = _contiguous_pre_peak_low_rms_ms(rms_lead, peak_i, hop, sr, thresh)
                if pre_ms + 1e-6 < float(lead_click_requires_pre_silence_ms):
                    conf_l = "lead_presilence_short"
                    do_lead = False
            if do_lead:
                s0, e0 = _expand_cut_from_center(
                    seg,
                    center,
                    sr,
                    diff_mad_k=d_k,
                    diff_median_smooth_samples=diff_median_smooth_samples,
                    max_transient_samples=max_t,
                    d_ds=d_ds,
                )
                cut = int(min(e0 + fade_n, lead_end, n))
                if removal in ("mute", "mute_then_trim"):
                    _apply_edge_fade_inplace(y, 0, cut, fade_n)
                    rem_l_ms = 1000.0 * cut / float(sr)
                    conf_l = "high"
                    if removal == "mute_then_trim" and cut > 0:
                        tail = y[int(cut) :]
                        extra = 0
                        if tail.size > 0:
                            extra = _silence_prefix_end_sample(
                                tail, frame, hop, float(lead_post_click_top_db)
                            )
                        y = y[int(cut) + extra :]
                        rem_l_ms += 1000.0 * extra / float(sr)
                elif removal == "fade":
                    w = min(max_t, cut)
                    ramp = np.linspace(1.0, 0.0, w + 1, dtype=np.float64)[:-1].astype(np.float32)
                    y[:w] *= ramp
                    rem_l_ms = 1000.0 * w / float(sr)
                    conf_l = "high"

    # y may have been shortened
    n = int(y.size)

    # --- Trailing: 末尾 ``trail_scan_ms`` 窓のみ（録音停止 UI）。発話本体は差分スパイクで局所除去。
    if trail_n > frame and n > frame:
        trail_start = max(0, n - trail_n)
        if trail_start < n - 8:
            seg_start = trail_start
            seg = np.asarray(y[seg_start:], dtype=np.float64)
            center_rel, reason, d_ds = _diff_click_center_in_segment(
                seg,
                sr,
                diff_mad_k=d_k,
                diff_median_smooth_samples=diff_median_smooth_samples,
                short_energy_ms=short_energy_ms,
                neighbor_energy_ms=neighbor_energy_ms,
                energy_spike_ratio=e_ratio,
                max_transient_samples=max_t,
                energy_baseline="trailing",
            )
            if center_rel is None:
                conf_t = reason if reason != "none" else "none"
            else:
                rms_tail = _frame_rms(y[seg_start:], frame, hop)
                nf = float(np.median(rms_tail[: max(1, int(rms_tail.size * 0.55))]) + 1e-8)
                thresh = nf * (10.0 ** (db / 20.0))
                peak_f = int(np.clip(center_rel // hop, 0, max(0, int(rms_tail.size) - 1)))
                start_f = peak_f
                kf = peak_f
                low_run = 0
                max_f = max(2, max_t // hop)
                while kf >= 0 and (peak_f - kf + 1) <= max_f:
                    if float(rms_tail[kf]) < thresh * 0.42:
                        low_run += 1
                        if low_run >= 2:
                            start_f = kf + low_run
                            break
                    else:
                        low_run = 0
                        start_f = kf
                    kf -= 1
                skip_trail = False
                if trail_click_requires_silence_ms is not None:
                    sil_ms = _contiguous_pre_burst_silence_ms(
                        rms_tail, start_f, hop, sr, nf, thresh
                    )
                    if sil_ms + 1e-6 < float(trail_click_requires_silence_ms):
                        conf_t = "trail_presilence_short"
                        skip_trail = True
                if not skip_trail:
                    s0, e0 = _expand_cut_from_center(
                        seg,
                        center_rel,
                        sr,
                        diff_mad_k=d_k,
                        diff_median_smooth_samples=diff_median_smooth_samples,
                        max_transient_samples=max_t,
                        d_ds=d_ds,
                    )
                    cut_from = seg_start + s0
                    if removal in ("mute", "mute_then_trim"):
                        _apply_edge_fade_inplace(y, cut_from, n, fade_n)
                        removed = n - cut_from
                        y = y[:cut_from]
                        rem_t_ms = 1000.0 * removed / float(sr)
                        conf_t = "high"
                    elif removal == "fade":
                        w = min(max_t, n - cut_from)
                        if w > 0:
                            ramp = np.linspace(0.0, 1.0, w + 1, dtype=np.float64)[1:].astype(np.float32)
                            y[-w:] *= ramp
                        rem_t_ms = 1000.0 * min(w, n - cut_from) / float(sr)
                        conf_t = "high"

    return EdgeClickResult(
        y=y.astype(np.float32),
        removed_leading_ms=rem_l_ms,
        removed_trailing_ms=rem_t_ms,
        confidence_leading=conf_l,
        confidence_trailing=conf_t,
    )
