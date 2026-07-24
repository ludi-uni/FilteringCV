"""Lightweight linguistic feature extraction for coverage indexing."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from cv_preprocess.linguistic.mora import mora_sequence_for_text
from cv_preprocess.linguistic.ngrams import (
    extract_phones,
    filter_phoneme_tokens,
    normalize_exclude_tokens,
    split_phoneme_tokens,
)

WORD_BOUNDARY = "|"


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def phonemes_from_g2p_string(phoneme_str: str, *, exclude_tokens: Iterable[str] | None = None) -> list[str]:
    return extract_phones(phoneme_str, exclude_tokens=exclude_tokens)


def extract_biphones_with_boundaries(
    phoneme_str: str,
    *,
    exclude_tokens: Iterable[str] | None = None,
    include_bos_eos: bool = True,
) -> list[str]:
    """Biphones over filtered phones, optionally padded with BOS/EOS."""
    phones = extract_phones(phoneme_str, exclude_tokens=exclude_tokens)
    if not phones:
        return []
    sequence = (["BOS"] + phones + ["EOS"]) if include_bos_eos else phones
    if len(sequence) < 2:
        return []
    return [f"{left}-{right}" for left, right in zip(sequence, sequence[1:])]


def extract_positioned_phonemes(
    phoneme_str: str,
    *,
    exclude_tokens: Iterable[str] | None = None,
    word_separator: str = WORD_BOUNDARY,
) -> list[str]:
    """Label phones as word_initial / word_medial / word_final within G2P word groups."""
    exclude = normalize_exclude_tokens(exclude_tokens)
    raw_tokens = split_phoneme_tokens(phoneme_str)
    words: list[list[str]] = [[]]
    for token in raw_tokens:
        if token == word_separator:
            words.append([])
            continue
        if token in {word_separator}:
            continue
        filtered = filter_phoneme_tokens([token], exclude=exclude)
        if filtered:
            words[-1].extend(filtered)
    words = [word for word in words if word]

    positioned: list[str] = []
    for word in words:
        if len(word) == 1:
            positioned.append(f"word_initial:{word[0]}")
            positioned.append(f"word_final:{word[0]}")
            continue
        for index, phone in enumerate(word):
            if index == 0:
                positioned.append(f"word_initial:{phone}")
            elif index == len(word) - 1:
                positioned.append(f"word_final:{phone}")
            else:
                positioned.append(f"word_medial:{phone}")
    return positioned


def extract_moras_for_text(text: str) -> list[str]:
    if not text.strip():
        return []
    return mora_sequence_for_text(text)


def feature_keys_from_parts(
    *,
    unique_phonemes: Sequence[str],
    moras: Sequence[str],
    biphones: Sequence[str],
    positioned_phonemes: Sequence[str],
) -> list[str]:
    keys: list[str] = []
    keys.extend(f"phoneme:{p}" for p in unique_preserve_order(unique_phonemes))
    keys.extend(f"mora:{m}" for m in unique_preserve_order(moras))
    keys.extend(f"biphone:{b}" for b in unique_preserve_order(biphones))
    keys.extend(f"positioned_phoneme:{p}" for p in unique_preserve_order(positioned_phonemes))
    return keys


def extract_coverage_features(
    *,
    normalized_text: str,
    phoneme_str: str,
    exclude_tokens: Iterable[str] | None = None,
    word_separator: str | None = "|",
) -> dict[str, list[str]]:
    phonemes = phonemes_from_g2p_string(phoneme_str, exclude_tokens=exclude_tokens)
    unique_phonemes = unique_preserve_order(phonemes)
    moras = extract_moras_for_text(normalized_text)
    biphones = extract_biphones_with_boundaries(phoneme_str, exclude_tokens=exclude_tokens)
    separator = word_separator if word_separator else WORD_BOUNDARY
    positioned = extract_positioned_phonemes(
        phoneme_str,
        exclude_tokens=exclude_tokens,
        word_separator=separator,
    )
    return {
        "phonemes": phonemes,
        "unique_phonemes": unique_phonemes,
        "moras": moras,
        "biphones": unique_preserve_order(biphones),
        "positioned_phonemes": unique_preserve_order(positioned),
        "feature_keys": feature_keys_from_parts(
            unique_phonemes=unique_phonemes,
            moras=moras,
            biphones=biphones,
            positioned_phonemes=positioned,
        ),
    }
