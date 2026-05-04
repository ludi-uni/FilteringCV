"""G2P 音素列とアライメント等で得た音素列の比較（足切り用）。"""

from __future__ import annotations


def normalize_phoneme_tokens(s: str) -> list[str]:
    """空白を正規化し、音素（トークン）列に分割する。"""
    return [t for t in s.replace("\t", " ").split() if t]


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    """トークン列の編集距離（挿入・削除・置換いずれもコスト 1）。

    時間 O(nm)、空間 O(min(n, m)) の 2 行 DP（長い列を外側ループに回す）。
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb]


def token_error_rate(g2p: str, aligned: str) -> float:
    """0〜1。完全一致なら 0。片方が空でも定義（もう片方が非空なら 1）。"""
    ta = normalize_phoneme_tokens(g2p)
    tb = normalize_phoneme_tokens(aligned)
    if not ta and not tb:
        return 0.0
    if not ta or not tb:
        return 1.0
    if ta == tb:
        return 0.0
    d = levenshtein_distance(ta, tb)
    return d / max(len(ta), len(tb))


def phoneme_sequences_accept(g2p: str, aligned: str, *, max_token_error_rate: float) -> bool:
    """``max_token_error_rate`` 以下なら合格（0 ならトークン列完全一致のみ）。"""
    if max_token_error_rate < 0 or max_token_error_rate > 1:
        raise ValueError("max_token_error_rate must be in [0, 1]")
    return token_error_rate(g2p, aligned) <= max_token_error_rate


def normalize_compare_chars(s: str) -> list[str]:
    """Unicode 文字単位（結合文字は 1 コードポイント = 1 要素）で比較列にする。"""
    t = (s or "").replace("\r", " ").replace("\n", " ").strip()
    return list(t)


def char_error_rate(reference: str, hypothesis: str) -> float:
    """文字編集距離 ÷ max(文字数)。両方空なら 0。"""
    a = normalize_compare_chars(reference)
    b = normalize_compare_chars(hypothesis)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    if a == b:
        return 0.0
    d = levenshtein_distance(a, b)
    return d / max(len(a), len(b))


def char_sequences_accept(reference: str, hypothesis: str, *, max_char_error_rate: float) -> bool:
    """``max_char_error_rate`` 以下なら合格（Unicode 文字列の編集距離率）。"""
    if max_char_error_rate < 0 or max_char_error_rate > 1:
        raise ValueError("max_char_error_rate must be in [0, 1]")
    return char_error_rate(reference, hypothesis) <= max_char_error_rate
