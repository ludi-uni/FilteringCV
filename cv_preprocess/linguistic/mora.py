"""Mora sequence extraction from Japanese text or OpenJTalk kana."""

from __future__ import annotations

from collections.abc import Sequence

from cv_preprocess.text.mora_estimate import mora_count_from_openjtalk_kana

# Reuse the same small-kana set as mora_estimate for consistent mora boundaries.
_SMALL_KANA = frozenset("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")


def mora_sequence_from_openjtalk_kana(kana: str) -> list[str]:
    """Split OpenJTalk kana output into a mora sequence (small kana attach to previous)."""
    compact = "".join(ch for ch in kana if not ch.isspace())
    if not compact:
        return []

    morae: list[str] = []
    current = ""
    for ch in compact:
        if ch in _SMALL_KANA:
            if current:
                current += ch
            else:
                current = ch
        else:
            if current:
                morae.append(current)
            current = ch
    if current:
        morae.append(current)
    return morae


def mora_sequence_for_text(text: str) -> list[str]:
    """Derive mora sequence from Japanese text via OpenJTalk kana reading."""
    from cv_preprocess.text.phonemize import g2p_phonemes

    kana = g2p_phonemes(text, kana=True)
    return mora_sequence_from_openjtalk_kana(kana)


def mora_count_from_sequence(morae: Sequence[str]) -> int:
    return len(morae)


def extract_mora_bigrams(morae: Sequence[str]) -> list[str]:
    return [f"{left}-{right}" for left, right in zip(morae, morae[1:])]


def verify_mora_count_matches_estimate(kana: str, morae: Sequence[str]) -> bool:
    """Sanity check that sequence length matches the existing mora counter."""
    return len(morae) == mora_count_from_openjtalk_kana(kana)
