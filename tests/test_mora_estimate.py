from cv_preprocess.text.mora_estimate import mora_count_from_openjtalk_kana


def test_mora_kana_ky_small_yo() -> None:
    # 「きゃ」は 1 モーラ扱い（小書きは前音に付く）
    assert mora_count_from_openjtalk_kana("きゃ") == 1


def test_mora_kana_simple_word() -> None:
    assert mora_count_from_openjtalk_kana("こ ん に ち は") == 5


def test_mora_kana_no_spaces() -> None:
    assert mora_count_from_openjtalk_kana("コンニチハ") == 5


def test_mora_kana_geminate_counts() -> None:
    # っは別モーラ
    assert mora_count_from_openjtalk_kana("がっこう") == 4


def test_mora_empty() -> None:
    assert mora_count_from_openjtalk_kana("") == 0
    assert mora_count_from_openjtalk_kana("   ") == 0
