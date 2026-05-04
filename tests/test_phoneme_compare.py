from cv_preprocess.text.phoneme_compare import (
    levenshtein_distance,
    normalize_phoneme_tokens,
    phoneme_sequences_accept,
    token_error_rate,
)


def _levenshtein_naive(a: list[str], b: list[str]) -> int:
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[la][lb]


def test_levenshtein_matches_naive_small() -> None:
    alphabet = list("abc")
    for la in range(6):
        for lb in range(6):
            a = [alphabet[i % 3] for i in range(la)]
            b = [alphabet[(i + 1) % 3] for i in range(lb)]
            assert levenshtein_distance(a, b) == _levenshtein_naive(a, b)


def test_levenshtein_empty_and_singleton() -> None:
    assert levenshtein_distance([], []) == 0
    assert levenshtein_distance([], ["x"]) == 1
    assert levenshtein_distance(["a", "b"], ["a", "b", "c"]) == 1


def test_normalize_tokens() -> None:
    assert normalize_phoneme_tokens("  a  b\tc  ") == ["a", "b", "c"]


def test_exact_match() -> None:
    assert phoneme_sequences_accept("a b c", "a b c", max_token_error_rate=0.0) is True
    assert phoneme_sequences_accept("a b c", "a b d", max_token_error_rate=0.0) is False


def test_token_error_rate_one_substitution() -> None:
    r = token_error_rate("a b c", "a b d")
    assert abs(r - 1.0 / 3.0) < 1e-9


def test_max_token_error_rate_allows_small_mismatch() -> None:
    assert phoneme_sequences_accept("a b c", "a b d", max_token_error_rate=0.34) is True
    assert phoneme_sequences_accept("a b c", "a b d", max_token_error_rate=0.32) is False


def test_token_error_rate_identical_after_normalize() -> None:
    assert token_error_rate("  a  b ", "a b") == 0.0
