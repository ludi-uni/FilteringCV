from __future__ import annotations

from cv_preprocess.text.phonemize import g2p_phonemes, g2p_phonemes_for_dataset


def test_g2p_phonemes_for_dataset_word_separator() -> None:
    plain = g2p_phonemes("今日 は いい 天気 です")
    with_sep = g2p_phonemes_for_dataset("今日 は いい 天気 です", word_separator="|")
    assert "|" in with_sep
    assert " ".join(plain.split()) == " ".join(t for t in with_sep.split() if t != "|")


def test_g2p_phonemes_for_dataset_single_word_unchanged() -> None:
    assert g2p_phonemes_for_dataset("あ", word_separator="|") == g2p_phonemes("あ")
