from cv_preprocess.text.normalize import is_mostly_japanese, normalize_for_tts


def test_normalize_basic() -> None:
    assert "  " not in normalize_for_tts("  あ　い　")


def test_japanese_ratio() -> None:
    assert is_mostly_japanese("今日はいい天気です") is True
    assert is_mostly_japanese("hello") is False
