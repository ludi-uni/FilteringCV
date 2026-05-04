"""差分ベースのデクリック: 1次（および任意で2次）差分のスパイク検出 + 局所補間。

Common Voice 向けの軽量前処理。広帯域をいじらず、急峻な局所不連続だけを
線形／3次スプラインで埋める。破裂音との誤検出緩和に ``require_both`` や
``min_abs_jump`` を使える。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from cv_preprocess.audio.lip_noise_repair import _merge_intervals, _repair_interval_inplace

_CONSISTENT_MAD = 1.4826


def apply_diff_click_repair(
    y: np.ndarray,
    sr: int,
    *,
    mad_k: float = 9.0,
    min_abs_jump: float | None = None,
    use_second_diff: bool = True,
    second_diff_mad_k: float = 8.0,
    require_both: bool = False,
    merge_gap_ms: float = 0.35,
    repair_pad_ms: float = 1.5,
    max_repair_ms: float = 20.0,
    max_repairs_per_clip: int = 256,
    interpolation: Literal["linear", "cubic"] = "cubic",
) -> tuple[np.ndarray, dict[str, float | int]]:
    """|Δy|（と任意で |Δ²y|）がロバスト閾値を超えた近傍だけを補間する。

    Parameters
    ----------
    mad_k
        1 次差分の閾値 ``median(|d1|) + mad_k * 1.4826 * MAD(|d1|)``。
    min_abs_jump
        非 ``None`` のとき ``max(上記, min_abs_jump * max(|y|))`` で下限を付与。
    use_second_diff
        True なら 2 次差分でも候補を取る（``require_both`` で AND にできる）。
    require_both
        True のとき、1 次と 2 次の両方でマークされたサンプルのみ残す。
    merge_gap_ms / repair_pad_ms / max_repair_ms
        区間のマージ・パディング・1 イベントあたりの最大修復幅。
    max_repairs_per_clip
        超過時は区間内の ``max(|d1|)`` が大きい順に残す。0 以下で無制限。
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError("apply_diff_click_repair expects mono 1-D audio")
    n = int(y.size)
    if n < 8 or sr < 4000:
        return y.copy(), {"diff_click_repair_events": 0, "diff_click_repair_samples": 0}

    y64 = y.astype(np.float64, copy=False)
    d1 = np.abs(np.diff(y64))
    med1 = float(np.median(d1))
    mad1 = float(np.median(np.abs(d1 - med1))) + 1e-20
    t1 = med1 + float(mad_k) * _CONSISTENT_MAD * mad1
    if min_abs_jump is not None and float(min_abs_jump) > 0.0:
        peak = float(np.max(np.abs(y64))) + 1e-20
        t1 = max(t1, float(min_abs_jump) * peak)

    mask = np.zeros(n, dtype=bool)
    hi1 = np.flatnonzero(d1 > t1)
    if hi1.size > 0:
        idx1 = np.unique(np.concatenate([hi1, hi1 + 1]))
        idx1 = idx1[(idx1 >= 0) & (idx1 < n)]
        mask[idx1] = True

    if use_second_diff and n >= 3:
        d2 = np.abs(y64[2:] - 2.0 * y64[1:-1] + y64[:-2])
        med2 = float(np.median(d2))
        mad2 = float(np.median(np.abs(d2 - med2))) + 1e-20
        t2 = med2 + float(second_diff_mad_k) * _CONSISTENT_MAD * mad2
        hi2 = np.flatnonzero(d2 > t2)
        if hi2.size > 0:
            spread = hi2[:, None] + np.arange(3, dtype=np.int64)
            idx2 = np.unique(spread.ravel())
            idx2 = idx2[(idx2 >= 0) & (idx2 < n)]
            m2 = np.zeros(n, dtype=bool)
            m2[idx2] = True
            if require_both:
                mask &= m2
            else:
                mask |= m2

    raw = _mask_to_intervals(mask)
    if not raw:
        return y.astype(np.float32).copy(), {"diff_click_repair_events": 0, "diff_click_repair_samples": 0}

    gap = max(0, int(round(merge_gap_ms * sr / 1000.0)))
    merged = _merge_intervals(raw, gap=gap)
    pad = max(0, int(round(repair_pad_ms * sr / 1000.0)))
    max_w = max(4, int(round(max_repair_ms * sr / 1000.0)))

    events: list[tuple[int, int]] = []
    for s, e in merged:
        s2 = max(0, s - pad)
        e2 = min(n, e + pad)
        if e2 <= s2:
            continue
        w = e2 - s2
        if w > max_w:
            c = (s2 + e2) // 2
            h = max_w // 2
            s2 = max(0, c - h)
            e2 = min(n, s2 + max_w)
            if e2 - s2 < max_w:
                s2 = max(0, e2 - max_w)
        events.append((s2, e2))

    events = _merge_intervals(events, gap=0)
    events.sort(key=lambda t: t[0])

    if max_repairs_per_clip > 0 and len(events) > max_repairs_per_clip:

        def _interval_score(s: int, e: int) -> float:
            if e <= s + 1:
                return 0.0
            local = d1[max(0, s - 1) : min(d1.size, e)]
            return float(np.max(local)) if local.size else 0.0

        scores = [_interval_score(s, e) for s, e in events]
        order = np.argsort(-np.asarray(scores, dtype=np.float64))[:max_repairs_per_clip]
        events = [events[int(i)] for i in order]
        events.sort(key=lambda t: t[0])

    y_out = y.astype(np.float32).copy()
    total = 0
    for s, e in reversed(events):
        total += _repair_interval_inplace(y_out, s, e, interpolation=interpolation)

    return y_out, {
        "diff_click_repair_events": int(len(events)),
        "diff_click_repair_samples": int(total),
    }


def _mask_to_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.empty(breaks.size + 1, dtype=np.int64)
    ends = np.empty(breaks.size + 1, dtype=np.int64)
    starts[0] = idx[0]
    if breaks.size:
        starts[1:] = idx[breaks + 1]
        ends[:-1] = idx[breaks] + 1
    ends[-1] = idx[-1] + 1
    return list(zip(starts.tolist(), ends.tolist()))
