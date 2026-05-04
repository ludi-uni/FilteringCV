from __future__ import annotations

from cv_preprocess.text.phoneme_compare import char_error_rate, char_sequences_accept


def test_char_error_rate_identical() -> None:
    assert char_error_rate("こんにちは", "こんにちは") == 0.0


def test_char_error_rate_one_substitution() -> None:
    r = char_error_rate("abc", "abx")
    assert 0 < r <= 1.0


def test_char_sequences_accept_threshold() -> None:
    assert char_sequences_accept("あいう", "あいう", max_char_error_rate=0.0)
    assert not char_sequences_accept("あいう", "あい", max_char_error_rate=0.0)
    assert char_sequences_accept("あいう", "あい", max_char_error_rate=0.5)
