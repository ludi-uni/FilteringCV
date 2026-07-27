"""Shared coverage feature-key helpers (unified with coverage-run)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from cv_preprocess.coverage.feature_extractor import (
    extract_coverage_features,
    feature_keys_from_parts,
    unique_preserve_order,
)

# Selection linguistic families ↔ coverage feature families
FAMILY_TO_COVERAGE: dict[str, str] = {
    "phone": "phoneme",
    "phoneme": "phoneme",
    "mora": "mora",
    "biphone": "biphone",
    "positioned_phoneme": "positioned_phoneme",
}

COVERAGE_TO_SELECTION: dict[str, str] = {
    "phoneme": "phone",
    "phone": "phone",
    "mora": "mora",
    "biphone": "biphone",
    "positioned_phoneme": "positioned_phoneme",
}


def normalize_feature_key(feature_key: str) -> str:
    """Normalize ``phone:x`` → ``phoneme:x``; leave other families unchanged."""
    family, sep, rest = feature_key.partition(":")
    if not sep:
        return feature_key
    if family == "phone":
        return f"phoneme:{rest}"
    return feature_key


def parse_feature_key(feature_key: str) -> tuple[str, str]:
    key = normalize_feature_key(feature_key)
    family, _, token = key.partition(":")
    return family, token


def format_feature_key(family: str, token: str) -> str:
    family = FAMILY_TO_COVERAGE.get(family, family)
    return f"{family}:{token}"


def coverage_keys_from_clip_parts(
    *,
    normalized_text: str,
    phoneme_str: str,
    exclude_tokens: Iterable[str] | None = None,
) -> list[str]:
    """Build coverage feature keys for a catalog clip (same as coverage-index)."""
    extracted = extract_coverage_features(
        normalized_text=normalized_text,
        phoneme_str=phoneme_str,
        exclude_tokens=exclude_tokens,
    )
    return list(extracted["feature_keys"])


def coverage_keys_from_selection_families(
    features_by_family: Mapping[str, Sequence[str]],
) -> list[str]:
    """Best-effort keys from selection linguistic families (no positioned_phoneme)."""
    phones = list(features_by_family.get("phone") or features_by_family.get("phoneme") or [])
    moras = list(features_by_family.get("mora") or [])
    biphones = list(features_by_family.get("biphone") or [])
    positioned = list(features_by_family.get("positioned_phoneme") or [])
    return feature_keys_from_parts(
        unique_phonemes=unique_preserve_order(phones),
        moras=unique_preserve_order(moras),
        biphones=unique_preserve_order(biphones),
        positioned_phonemes=unique_preserve_order(positioned),
    )
