"""Unit tests for coverage feature extraction."""

from __future__ import annotations

from cv_preprocess.coverage.feature_extractor import (
    extract_biphones_with_boundaries,
    extract_coverage_features,
    extract_positioned_phonemes,
    phonemes_from_g2p_string,
    unique_preserve_order,
)


def test_phoneme_extraction_and_dedup() -> None:
    phones = phonemes_from_g2p_string("k o N n i ch i w a")
    assert phones == ["k", "o", "N", "n", "i", "ch", "i", "w", "a"]
    assert unique_preserve_order(phones) == ["k", "o", "N", "n", "i", "ch", "w", "a"]


def test_biphones_bos_eos() -> None:
    biphones = extract_biphones_with_boundaries("a ts u")
    assert biphones[0] == "BOS-a"
    assert "a-ts" in biphones
    assert "ts-u" in biphones
    assert biphones[-1] == "u-EOS"


def test_positioned_phonemes_word_boundaries() -> None:
    positioned = extract_positioned_phonemes("k o | n i")
    assert "word_initial:k" in positioned
    assert "word_medial:o" in positioned or "word_final:o" in positioned
    assert "word_initial:n" in positioned


def test_coverage_features_bundle() -> None:
    features = extract_coverage_features(
        normalized_text="こんにちは",
        phoneme_str="k o N n i ch i w a",
    )
    assert "phoneme:k" in features["feature_keys"]
    assert any(key.startswith("biphone:") for key in features["feature_keys"])
    assert features["unique_phonemes"] == unique_preserve_order(features["phonemes"])
