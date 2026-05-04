"""OpenJTalk カナ列からモーラ数の目安を数える（テキスト対実長の粗い整合ゲート用）。"""

from __future__ import annotations

# 拗音・母音寄りの「前音に付く」小書き（っはモーラ単位として数える）
_SMALL_KANA = frozenset("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")


def mora_count_from_openjtalk_kana(kana: str) -> int:
    """``pyopenjtalk.g2p(..., kana=True)`` の出力想定。空白は無視する。"""
    s = "".join(ch for ch in kana if not ch.isspace())
    if not s:
        return 0
    n = 0
    for ch in s:
        if ch in _SMALL_KANA:
            continue
        n += 1
    return n


def mora_count_for_text(text_norm: str) -> int:
    """正規化済みテキストから OpenJTalk で読みを取り、モーラ数の下限目安を返す。"""
    from cv_preprocess.text.phonemize import g2p_phonemes

    kana = g2p_phonemes(text_norm, kana=True)
    return mora_count_from_openjtalk_kana(kana)
