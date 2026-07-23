from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from cv_preprocess.selection.protocol import ClipFeatures


def target_distribution(pool_counts: dict[str, int], alpha: float) -> dict[str, float]:
    """Compute P_target(f) proportional to P_pool(f)^alpha."""
    if not pool_counts:
        return {}
    positive = {token: count for token, count in pool_counts.items() if count > 0}
    if not positive:
        return {}
    if alpha == 0.0:
        uniform = 1.0 / len(positive)
        return dict.fromkeys(positive, uniform)
    powered = {token: float(count) ** alpha for token, count in positive.items()}
    total = sum(powered.values())
    if total <= 0.0:
        uniform = 1.0 / len(positive)
        return dict.fromkeys(positive, uniform)
    return {token: value / total for token, value in powered.items()}


def feature_utility(weight: float, count: float, tau: float) -> float:
    """Diminishing returns utility: weight * log(1 + count / tau)."""
    if weight <= 0.0:
        return 0.0
    safe_tau = tau if tau > 0.0 else 1.0
    safe_count = max(count, 0.0)
    return weight * math.log(1.0 + safe_count / safe_tau)


def marginal_feature_utility(weight: float, current_count: float, tau: float) -> float:
    return feature_utility(weight, current_count + 1.0, tau) - feature_utility(
        weight, current_count, tau
    )


def pool_counts_for_family(
    candidates: Iterable[ClipFeatures],
    family: str,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        for token in candidate.features_by_family.get(family, []):
            counts[token] += 1
    return counts


def speaker_counts_for_family(
    candidates: Iterable[ClipFeatures],
    family: str,
) -> dict[str, set[str]]:
    speakers_by_token: dict[str, set[str]] = {}
    for candidate in candidates:
        for token in candidate.features_by_family.get(family, []):
            speakers_by_token.setdefault(token, set()).add(candidate.speaker_id)
    return speakers_by_token


def is_strong_rescue_eligible(
    token: str,
    *,
    family: str,
    utterance_count: int,
    speaker_count: int,
    min_utterances: int | None,
    min_speakers: int | None,
) -> bool:
    if min_utterances is not None and utterance_count < min_utterances:
        return False
    if min_speakers is not None and speaker_count < min_speakers:
        return False
    return True


def incremental_clip_score(
    clip: ClipFeatures,
    *,
    family: str,
    family_weight: float,
    tau: float,
    alpha: float,
    current_counts: Counter[str],
    pool_counts: Counter[str],
    utterance_counts: Counter[str],
    speaker_sets: dict[str, set[str]],
    min_utterances: int | None = None,
    min_speakers: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Marginal coverage score for one clip within a feature family."""
    if family_weight <= 0.0:
        return 0.0, {}

    targets = target_distribution(dict(pool_counts), alpha)
    contributions: dict[str, float] = {}
    total = 0.0
    seen: set[str] = set()
    for token in clip.features_by_family.get(family, []):
        if token in seen:
            continue
        seen.add(token)
        speaker_count = len(speaker_sets.get(token, set()))
        utterance_count = utterance_counts.get(token, 0)
        if not is_strong_rescue_eligible(
            token,
            family=family,
            utterance_count=utterance_count,
            speaker_count=speaker_count,
            min_utterances=min_utterances,
            min_speakers=min_speakers,
        ):
            continue
        target_p = targets.get(token, 0.0)
        if target_p <= 0.0:
            continue
        current = float(current_counts.get(token, 0))
        marginal = marginal_feature_utility(family_weight, current, tau)
        contribution = target_p * marginal
        contributions[token] = contribution
        total += contribution
    return total, contributions


def total_selection_score(
    clip: ClipFeatures,
    *,
    feature_weights: dict[str, float],
    diminishing_tau: dict[str, float],
    temperatures: dict[str, float],
    current_counts_by_family: dict[str, Counter[str]],
    pool_counts_by_family: dict[str, Counter[str]],
    utterance_counts_by_family: dict[str, Counter[str]],
    speaker_sets_by_family: dict[str, dict[str, set[str]]],
    min_utterances_by_family: dict[str, int],
    min_speakers_by_family: dict[str, int],
    selected_speakers: set[str],
    quality_weight: float = 0.0,
    speaker_diversity_weight: float = 0.0,
) -> tuple[float, dict[str, dict[str, float]], dict[str, float]]:
    positive: dict[str, dict[str, float]] = {}
    penalties: dict[str, float] = {}
    total = 0.0

    for family, weight in feature_weights.items():
        if family in {"quality", "speaker_diversity", "acoustic_diversity"}:
            continue
        tau = diminishing_tau.get(family, 1.0)
        alpha = temperatures.get(family, 1.0)
        family_score, contribs = incremental_clip_score(
            clip,
            family=family,
            family_weight=weight,
            tau=tau,
            alpha=alpha,
            current_counts=current_counts_by_family.get(family, Counter()),
            pool_counts=pool_counts_by_family.get(family, Counter()),
            utterance_counts=utterance_counts_by_family.get(family, Counter()),
            speaker_sets=speaker_sets_by_family.get(family, {}),
            min_utterances=min_utterances_by_family.get(family),
            min_speakers=min_speakers_by_family.get(family),
        )
        if contribs:
            positive[family] = contribs
        total += family_score

    if speaker_diversity_weight > 0.0 and clip.speaker_id not in selected_speakers:
        bonus = speaker_diversity_weight
        positive["speaker_diversity"] = {"new_speaker": bonus}
        total += bonus

    if quality_weight > 0.0 and clip.quality_score is not None:
        bonus = quality_weight * (clip.quality_score / 100.0)
        positive["quality"] = {"score_bonus": bonus}
        total += bonus

    return total, positive, penalties
