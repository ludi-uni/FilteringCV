"""Phone n-gram extraction from space-separated phoneme strings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

DEFAULT_BOUNDARY_EXCLUDE: frozenset[str] = frozenset({"sil", "pau", "sp", "SP", "|", ""})
_CASE_INSENSITIVE_BOUNDARY = frozenset({"sil", "pau", "sp"})


def normalize_exclude_tokens(tokens: Iterable[str] | None = None) -> frozenset[str]:
    """Normalize exclude tokens; sil/pau/sp are stored lowercase for case-insensitive match."""
    raw = tokens if tokens is not None else DEFAULT_BOUNDARY_EXCLUDE
    normalized: set[str] = set()
    for token in raw:
        if token.lower() in _CASE_INSENSITIVE_BOUNDARY:
            normalized.add(token.lower())
        else:
            normalized.add(token)
    return frozenset(normalized)


def is_excluded_token(token: str, exclude: frozenset[str]) -> bool:
    if not token:
        return True
    if token in exclude:
        return True
    lowered = token.lower()
    return lowered in _CASE_INSENSITIVE_BOUNDARY and lowered in exclude


def split_phoneme_tokens(phoneme_str: str) -> list[str]:
    return phoneme_str.split()


def filter_phoneme_tokens(
    tokens: Sequence[str],
    *,
    exclude: frozenset[str] | None = None,
) -> list[str]:
    exclude_set = exclude if exclude is not None else normalize_exclude_tokens()
    return [token for token in tokens if not is_excluded_token(token, exclude_set)]


def extract_phones(
    phoneme_str: str,
    *,
    exclude_tokens: Iterable[str] | None = None,
) -> list[str]:
    exclude = normalize_exclude_tokens(exclude_tokens)
    return filter_phoneme_tokens(split_phoneme_tokens(phoneme_str), exclude=exclude)


def extract_biphones(
    phoneme_str: str,
    *,
    exclude_tokens: Iterable[str] | None = None,
) -> list[str]:
    phones = extract_phones(phoneme_str, exclude_tokens=exclude_tokens)
    return [f"{left}-{right}" for left, right in zip(phones, phones[1:])]


def extract_triphones(
    phoneme_str: str,
    *,
    exclude_tokens: Iterable[str] | None = None,
) -> list[str]:
    phones = extract_phones(phoneme_str, exclude_tokens=exclude_tokens)
    return [
        f"{phones[index]}-{phones[index + 1]}-{phones[index + 2]}"
        for index in range(len(phones) - 2)
    ]


def extract_boundary_markers(
    phoneme_str: str,
    *,
    exclude_tokens: Iterable[str] | None = None,
) -> list[str]:
    """Return boundary tokens present in the phoneme string (sil/pau/sp/|), in order."""
    exclude = normalize_exclude_tokens(exclude_tokens)
    markers: list[str] = []
    for token in split_phoneme_tokens(phoneme_str):
        if is_excluded_token(token, exclude):
            lowered = token.lower()
            if lowered in _CASE_INSENSITIVE_BOUNDARY or token == "|":
                label = lowered if lowered in _CASE_INSENSITIVE_BOUNDARY else token
                if label not in markers:
                    markers.append(label)
    return markers


def extract_downweighted_tokens(
    phoneme_str: str,
    *,
    down_weight_tokens: Iterable[str] | None = None,
) -> list[str]:
    if not down_weight_tokens:
        return []
    down_set = normalize_exclude_tokens(down_weight_tokens)
    found: list[str] = []
    for token in split_phoneme_tokens(phoneme_str):
        if is_excluded_token(token, down_set):
            label = token.lower() if token.lower() in _CASE_INSENSITIVE_BOUNDARY else token
            if label not in found:
                found.append(label)
    return found
