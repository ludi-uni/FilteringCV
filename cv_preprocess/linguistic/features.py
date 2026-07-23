"""Aggregate linguistic features for dataset builder catalog rows."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from cv_preprocess.linguistic.fullcontext import (
    accent_phrase_length_band,
    extract_fullcontext_labels,
    parse_accent_nucleus_features,
    parse_accent_phrase_lengths,
)
from cv_preprocess.linguistic.mora import extract_mora_bigrams, mora_sequence_for_text
from cv_preprocess.linguistic.ngrams import (
    extract_biphones,
    extract_boundary_markers,
    extract_downweighted_tokens,
    extract_phones,
    extract_triphones,
)


class FeatureSource(str, Enum):
    ALIGNED = "aligned"
    ASR_INFERRED = "asr_inferred"
    TEXT_G2P = "text_g2p"


UtteranceType = Literal["interrogative", "declarative"]


class LinguisticFeatures(BaseModel):
    feature_source: FeatureSource
    phones: list[str] = Field(default_factory=list)
    biphones: list[str] = Field(default_factory=list)
    triphones: list[str] = Field(default_factory=list)
    morae: list[str] = Field(default_factory=list)
    mora_bigrams: list[str] = Field(default_factory=list)
    mora_count: int = 0
    full_context_labels: list[str] | None = None
    full_context_warning: str | None = None
    accent_nucleus_features: list[str] = Field(default_factory=list)
    accent_phrase_length_bands: list[str] = Field(default_factory=list)
    pause_boundary_markers: list[str] = Field(default_factory=list)
    sentence_length_band: str | None = None
    speaking_rate_band: str | None = None
    speaking_rate_mora_per_sec: float | None = None
    utterance_type: UtteranceType | None = None
    downweighted_tokens: list[str] = Field(default_factory=list)


def sentence_length_band(mora_count: int) -> str:
    if mora_count <= 3:
        return "very_short"
    if mora_count <= 7:
        return "short"
    if mora_count <= 15:
        return "medium"
    return "long"


def speaking_rate_band(mora_per_sec: float) -> str:
    if mora_per_sec < 5.0:
        return "slow"
    if mora_per_sec <= 9.0:
        return "normal"
    return "fast"


def detect_utterance_type(text: str) -> UtteranceType:
    stripped = text.rstrip()
    if stripped.endswith("？") or stripped.endswith("?"):
        return "interrogative"
    if stripped.endswith("か"):
        return "interrogative"
    return "declarative"


def extract_linguistic_features(
    text: str,
    phonemes: str | None,
    *,
    feature_source: FeatureSource,
    duration_sec: float | None = None,
    exclude_tokens: list[str] | None = None,
    down_weight_tokens: list[str] | None = None,
) -> LinguisticFeatures:
    """Extract linguistic coverage features from text and optional phoneme string."""
    phoneme_str = phonemes or ""
    morae = mora_sequence_for_text(text) if text.strip() else []
    mora_count = len(morae)

    full_context_labels, full_context_warning = extract_fullcontext_labels(text)
    accent_nucleus_features: list[str] = []
    accent_phrase_length_bands: list[str] = []
    if full_context_labels:
        accent_nucleus_features = parse_accent_nucleus_features(full_context_labels)
        accent_phrase_length_bands = [
            accent_phrase_length_band(length) for length in parse_accent_phrase_lengths(full_context_labels)
        ]

    speaking_rate_mora_per_sec: float | None = None
    rate_band: str | None = None
    if duration_sec is not None and duration_sec > 0 and mora_count > 0:
        speaking_rate_mora_per_sec = mora_count / duration_sec
        rate_band = speaking_rate_band(speaking_rate_mora_per_sec)

    return LinguisticFeatures(
        feature_source=feature_source,
        phones=extract_phones(phoneme_str, exclude_tokens=exclude_tokens),
        biphones=extract_biphones(phoneme_str, exclude_tokens=exclude_tokens),
        triphones=extract_triphones(phoneme_str, exclude_tokens=exclude_tokens),
        morae=morae,
        mora_bigrams=extract_mora_bigrams(morae),
        mora_count=mora_count,
        full_context_labels=full_context_labels,
        full_context_warning=full_context_warning,
        accent_nucleus_features=accent_nucleus_features,
        accent_phrase_length_bands=accent_phrase_length_bands,
        pause_boundary_markers=extract_boundary_markers(phoneme_str, exclude_tokens=exclude_tokens),
        sentence_length_band=sentence_length_band(mora_count) if mora_count > 0 else None,
        speaking_rate_band=rate_band,
        speaking_rate_mora_per_sec=speaking_rate_mora_per_sec,
        utterance_type=detect_utterance_type(text) if text.strip() else None,
        downweighted_tokens=extract_downweighted_tokens(phoneme_str, down_weight_tokens=down_weight_tokens),
    )
