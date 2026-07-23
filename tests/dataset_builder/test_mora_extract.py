from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from cv_preprocess.linguistic.features import FeatureSource, extract_linguistic_features
from cv_preprocess.linguistic.fullcontext import extract_fullcontext_labels
from cv_preprocess.linguistic.mora import mora_sequence_for_text, mora_sequence_from_openjtalk_kana


def test_mora_sequence_from_kana_is_deterministic() -> None:
    kana = "こ ん に ち は"
    first = mora_sequence_from_openjtalk_kana(kana)
    second = mora_sequence_from_openjtalk_kana(kana)
    assert first == second == ["こ", "ん", "に", "ち", "は"]


def test_mora_sequence_for_text_is_deterministic() -> None:
    text = "今日は"
    first = mora_sequence_for_text(text)
    second = mora_sequence_for_text(text)
    assert first == second
    assert len(first) > 0


def test_mora_bigram_extraction() -> None:
    features = extract_linguistic_features(
        "こんにちは",
        "k o N n i ch i w a",
        feature_source=FeatureSource.TEXT_G2P,
    )
    assert features.mora_bigrams == [f"{left}-{right}" for left, right in zip(features.morae, features.morae[1:])]


def test_fullcontext_failure_does_not_raise() -> None:
    mock_pyopenjtalk = MagicMock()
    mock_pyopenjtalk.extract_fullcontext.side_effect = RuntimeError("boom")
    with patch.dict(sys.modules, {"pyopenjtalk": mock_pyopenjtalk}):
        labels, warning = extract_fullcontext_labels("テスト")
    assert labels is None
    assert warning is not None
    assert "fullcontext unavailable" in warning


def test_extract_linguistic_features_survives_fullcontext_failure() -> None:
    with patch(
        "cv_preprocess.linguistic.features.extract_fullcontext_labels",
        return_value=(None, "fullcontext unavailable: boom"),
    ):
        features = extract_linguistic_features(
            "こんにちは",
            "k o N n i ch i w a",
            feature_source=FeatureSource.ALIGNED,
            duration_sec=1.0,
        )
    assert features.full_context_labels is None
    assert features.full_context_warning is not None
    assert features.phones
    assert features.mora_count > 0


def test_interrogative_detection() -> None:
    question = extract_linguistic_features(
        "これは何ですか？",
        None,
        feature_source=FeatureSource.TEXT_G2P,
    )
    statement = extract_linguistic_features(
        "これは本です。",
        None,
        feature_source=FeatureSource.TEXT_G2P,
    )
    assert question.utterance_type == "interrogative"
    assert statement.utterance_type == "declarative"
