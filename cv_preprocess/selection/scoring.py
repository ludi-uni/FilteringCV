from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from cv_preprocess.selection.protocol import ClipFeatures


def target_distribution(pool_counts: dict[str, int] | Counter[str], alpha: float) -> dict[str, float]:
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
    # log(1+(c+1)/τ) - log(1+c/τ) = log((τ+c+1)/(τ+c))
    if weight <= 0.0:
        return 0.0
    safe_tau = tau if tau > 0.0 else 1.0
    safe_count = max(current_count, 0.0)
    return weight * math.log((safe_tau + safe_count + 1.0) / (safe_tau + safe_count))


def unique_tokens(tokens: list[str]) -> list[str]:
    if not tokens:
        return tokens
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def dedupe_clip_features(clip: ClipFeatures) -> dict[str, list[str]]:
    return {
        family: unique_tokens(tokens) for family, tokens in clip.features_by_family.items() if tokens
    }


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


def precompute_family_score_tables(
    *,
    feature_weights: dict[str, float],
    temperatures: dict[str, float],
    pool_counts_by_family: dict[str, Counter[str]],
    utterance_counts_by_family: dict[str, Counter[str]],
    speaker_sets_by_family: dict[str, dict[str, set[str]]],
    min_utterances_by_family: dict[str, int],
    min_speakers_by_family: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Precompute per-family token target weights for strong-rescue-eligible tokens.

    Returns mapping family -> {token: target_p} for tokens that pass support thresholds.
    Pool/target distributions are static during greedy selection.
    """
    tables: dict[str, dict[str, float]] = {}
    for family, weight in feature_weights.items():
        if family in {"quality", "speaker_diversity", "acoustic_diversity"}:
            continue
        if weight <= 0.0:
            continue
        pool_counts = pool_counts_by_family.get(family, Counter())
        alpha = temperatures.get(family, 1.0)
        targets = target_distribution(pool_counts, alpha)
        utterance_counts = utterance_counts_by_family.get(family, Counter())
        speaker_sets = speaker_sets_by_family.get(family, {})
        min_u = min_utterances_by_family.get(family)
        min_s = min_speakers_by_family.get(family)
        eligible_targets: dict[str, float] = {}
        for token, target_p in targets.items():
            if target_p <= 0.0:
                continue
            if not is_strong_rescue_eligible(
                token,
                family=family,
                utterance_count=utterance_counts.get(token, 0),
                speaker_count=len(speaker_sets.get(token, set())),
                min_utterances=min_u,
                min_speakers=min_s,
            ):
                continue
            eligible_targets[token] = target_p
        tables[family] = eligible_targets
    return tables


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
    targets: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Marginal coverage score for one clip within a feature family."""
    if family_weight <= 0.0:
        return 0.0, {}

    if targets is None:
        raw_targets = target_distribution(pool_counts, alpha)
        targets = {}
        for token, target_p in raw_targets.items():
            if target_p <= 0.0:
                continue
            if not is_strong_rescue_eligible(
                token,
                family=family,
                utterance_count=utterance_counts.get(token, 0),
                speaker_count=len(speaker_sets.get(token, set())),
                min_utterances=min_utterances,
                min_speakers=min_speakers,
            ):
                continue
            targets[token] = target_p

    contributions: dict[str, float] = {}
    total = 0.0
    seen: set[str] = set()
    for token in clip.features_by_family.get(family, []):
        if token in seen:
            continue
        seen.add(token)
        target_p = targets.get(token)
        if target_p is None or target_p <= 0.0:
            continue
        current = float(current_counts.get(token, 0))
        marginal = marginal_feature_utility(family_weight, current, tau)
        contribution = target_p * marginal
        contributions[token] = contribution
        total += contribution
    return total, contributions


def score_clip_fast(
    clip: ClipFeatures,
    *,
    feature_weights: dict[str, float],
    diminishing_tau: dict[str, float],
    target_tables: dict[str, dict[str, float]],
    current_counts_by_family: dict[str, Counter[str]],
    selected_speakers: set[str],
    quality_weight: float = 0.0,
    speaker_diversity_weight: float = 0.0,
    features_by_family: dict[str, list[str]] | None = None,
) -> float:
    """Hot-path score without building contribution dictionaries."""
    total = 0.0
    features = features_by_family if features_by_family is not None else clip.features_by_family
    for family, targets in target_tables.items():
        weight = feature_weights.get(family, 0.0)
        if weight <= 0.0:
            continue
        tau = diminishing_tau.get(family, 1.0)
        current_counts = current_counts_by_family.get(family)
        tokens = features.get(family)
        if not tokens:
            continue
        # Callers should pass pre-deduped tokens; still guard if raw clip features are used.
        seen: set[str] | None = None if features_by_family is not None else set()
        for token in tokens:
            if seen is not None:
                if token in seen:
                    continue
                seen.add(token)
            target_p = targets.get(token)
            if target_p is None:
                continue
            current = float(current_counts.get(token, 0)) if current_counts is not None else 0.0
            total += target_p * marginal_feature_utility(weight, current, tau)

    if speaker_diversity_weight > 0.0 and clip.speaker_id not in selected_speakers:
        total += speaker_diversity_weight

    if quality_weight > 0.0 and clip.quality_score is not None:
        total += quality_weight * (clip.quality_score / 100.0)

    return total


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
    target_tables: dict[str, dict[str, float]] | None = None,
) -> tuple[float, dict[str, dict[str, float]], dict[str, float]]:
    positive: dict[str, dict[str, float]] = {}
    penalties: dict[str, float] = {}
    total = 0.0

    if target_tables is None:
        target_tables = precompute_family_score_tables(
            feature_weights=feature_weights,
            temperatures=temperatures,
            pool_counts_by_family=pool_counts_by_family,
            utterance_counts_by_family=utterance_counts_by_family,
            speaker_sets_by_family=speaker_sets_by_family,
            min_utterances_by_family=min_utterances_by_family,
            min_speakers_by_family=min_speakers_by_family,
        )

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
            targets=target_tables.get(family),
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
