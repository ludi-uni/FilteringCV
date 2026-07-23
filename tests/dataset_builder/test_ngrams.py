from __future__ import annotations

from cv_preprocess.linguistic.ngrams import (
    extract_biphones,
    extract_phones,
    extract_triphones,
)


def test_extract_phones_unigrams() -> None:
    assert extract_phones("a i u") == ["a", "i", "u"]


def test_extract_biphones() -> None:
    assert extract_biphones("a i u") == ["a-i", "i-u"]


def test_extract_triphones() -> None:
    assert extract_triphones("a i u") == ["a-i-u"]


def test_boundary_tokens_excluded_from_ngrams() -> None:
    phonemes = "sil a pau i sil u"
    assert extract_phones(phonemes) == ["a", "i", "u"]
    assert extract_biphones(phonemes) == ["a-i", "i-u"]
    assert extract_triphones(phonemes) == ["a-i-u"]


def test_boundary_tokens_case_insensitive() -> None:
    phonemes = "SIL a PAU i Sp u"
    assert extract_phones(phonemes) == ["a", "i", "u"]


def test_word_boundary_pipe_excluded() -> None:
    phonemes = "a | i u"
    assert extract_phones(phonemes) == ["a", "i", "u"]
    assert extract_biphones(phonemes) == ["a-i", "i-u"]
