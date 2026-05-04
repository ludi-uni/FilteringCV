from __future__ import annotations

from cv_preprocess.pipeline.mfa_g2p_map_suggest import (
    _collect_votes_for_row,
    _pairs_proportional,
    _pairs_zip,
)


def test_pairs_zip_equal_len() -> None:
    assert _pairs_zip(["a", "b"], ["x", "y"]) == [("a", "x"), ("b", "y")]
    assert _pairs_zip(["a"], ["x", "y"]) is None


def test_pairs_proportional_single() -> None:
    assert _pairs_proportional(["m"], ["g"]) == [("m", "g")]


def test_collect_adaptive_zip_when_equal() -> None:
    pairs, m = _collect_votes_for_row(["a", "b"], ["x", "y"], "adaptive")
    assert m == "zip"
    assert pairs == [("a", "x"), ("b", "y")]


def test_collect_adaptive_proportional_when_unequal() -> None:
    pairs, m = _collect_votes_for_row(["a", "b"], ["p", "q", "r"], "adaptive")
    assert m == "proportional"
    assert len(pairs) == 2


def test_zip_only_skips_mismatch() -> None:
    pairs, m = _collect_votes_for_row(["a", "b"], ["x"], "zip_only")
    assert m == "none"
    assert pairs == []


def test_pairs_proportional_index_matches_centered_floor() -> None:
    """``(i+0.5)*m/n`` の floor と整数式 ``(2i+1)*m//(2n)`` が一致する（正の n, m）。"""
    for n in range(1, 12):
        for m in range(1, 15):
            mfa = [f"M{i}" for i in range(n)]
            g2p = [f"G{j}" for j in range(m)]
            pairs = _pairs_proportional(mfa, g2p)
            for i, (mt, gt) in enumerate(pairs):
                j_float = min(m - 1, max(0, int((i + 0.5) * m / n)))
                assert mt == mfa[i]
                assert gt == g2p[j_float]
