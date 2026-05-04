"""ブール配列上の True 連続区間（ラン）のフィルタ。lip ノイズ系で共有。"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def short_runs_only(mask: np.ndarray, max_len: int) -> np.ndarray:
    """True の連続区間の長さが ``max_len`` 以下のものだけ残す（それ以外は False）。"""
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(m, dtype=bool)
    if not np.any(m) or int(max_len) <= 0:
        return out
    labels, nfeat = ndimage.label(m)
    if nfeat <= 0:
        return out
    counts = np.bincount(labels.ravel(), minlength=nfeat + 1)[1 : nfeat + 1]
    keep = np.flatnonzero(counts <= int(max_len)) + 1
    if keep.size == 0:
        return out
    return np.isin(labels, keep)
