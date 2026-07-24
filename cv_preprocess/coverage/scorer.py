"""Candidate scoring for coverage batch planning."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from cv_preprocess.config.coverage import CoverageAutomationConfig
from cv_preprocess.coverage.models import ClipIndexRecord, ScoreBreakdown, SpeakerPassStats
from cv_preprocess.coverage.pass_probability import clip_probability


def rarity_weight(pool_candidate_count: int) -> float:
    return 1.0 / math.sqrt(max(pool_candidate_count, 0) + 1)


def deficit_weight(deficit: int, target: int) -> float:
    if deficit <= 0:
        return 0.0
    return deficit / max(target, 1)


def feature_gain(
    *,
    feature: str,
    deficit: int,
    target: int,
    pool_count: int,
    config: CoverageAutomationConfig,
) -> float:
    if deficit <= 0 or target <= 0:
        return 0.0
    weight = config.target_weight_default
    if config.is_required(feature):
        weight *= config.required_weight_bonus
    return weight * rarity_weight(pool_count) * deficit_weight(deficit, target)


def marginal_coverage_gain(
    record: ClipIndexRecord,
    *,
    deficits: Mapping[str, int],
    targets: Mapping[str, int],
    pool_counts: Mapping[str, int],
    config: CoverageAutomationConfig,
) -> tuple[float, list[str]]:
    matched: list[str] = []
    total = 0.0
    for feature in record.feature_key_set():
        deficit = int(deficits.get(feature, 0))
        if deficit <= 0:
            continue
        target = int(targets.get(feature, 0))
        gain = feature_gain(
            feature=feature,
            deficit=deficit,
            target=target,
            pool_count=int(pool_counts.get(feature, 0)),
            config=config,
        )
        if gain > 0:
            matched.append(feature)
            total += gain
    return total, matched


def estimated_analysis_cost(record: ClipIndexRecord, config: CoverageAutomationConfig) -> float:
    duration = float(record.duration_sec or 0.0)
    return config.analysis_cost.base + duration * config.analysis_cost.duration_weight


def diversity_factor(
    record: ClipIndexRecord,
    *,
    selected_batch: Sequence[ClipIndexRecord],
    config: CoverageAutomationConfig,
) -> tuple[float, float]:
    same_speaker = sum(1 for item in selected_batch if item.client_id == record.client_id)
    if same_speaker >= config.diversity.max_per_speaker_per_batch:
        return 0.0, 0.0

    speaker_factor = 1.0 / (1.0 + same_speaker * config.diversity.speaker_penalty)

    same_text = sum(
        1
        for item in selected_batch
        if item.normalized_text == record.normalized_text and record.normalized_text
    )
    if same_text > 0:
        speaker_factor /= 1.0 + same_text * config.diversity.duplicate_text_penalty

    return speaker_factor, speaker_factor


def score_candidate(
    record: ClipIndexRecord,
    *,
    deficits: Mapping[str, int],
    targets: Mapping[str, int],
    pool_counts: Mapping[str, int],
    selected_batch: Sequence[ClipIndexRecord],
    speaker_stats: Mapping[str, SpeakerPassStats],
    global_attempts: int,
    global_passes: int,
    config: CoverageAutomationConfig,
    rare_rescue_features: set[str] | None = None,
) -> ScoreBreakdown:
    gain, matched = marginal_coverage_gain(
        record,
        deficits=deficits,
        targets=targets,
        pool_counts=pool_counts,
        config=config,
    )
    rare_rescue = False
    if rare_rescue_features and matched:
        rare_rescue = any(feature in rare_rescue_features for feature in matched)

    if gain <= 0:
        return ScoreBreakdown(
            clip_id=record.clip_id,
            score=0.0,
            expected_pass_probability=0.0,
            estimated_cost=estimated_analysis_cost(record, config),
            coverage_gain=0.0,
            diversity_factor=0.0,
            matched_deficits=[],
            rare_rescue=rare_rescue,
        )

    pass_prob = clip_probability(
        record,
        speaker_stats=speaker_stats,
        global_attempts=global_attempts,
        global_passes=global_passes,
        config=config.pass_probability,
    )
    cost = max(estimated_analysis_cost(record, config), 1e-6)
    diversity, speaker_pen = diversity_factor(record, selected_batch=selected_batch, config=config)
    score = (pass_prob * gain * diversity) / cost
    return ScoreBreakdown(
        clip_id=record.clip_id,
        score=float(score),
        expected_pass_probability=float(pass_prob),
        estimated_cost=float(cost),
        coverage_gain=float(gain),
        diversity_factor=float(diversity),
        matched_deficits=matched,
        speaker_penalty=float(speaker_pen),
        rare_rescue=rare_rescue,
        details={"same_speaker_in_batch": sum(1 for s in selected_batch if s.client_id == record.client_id)},
    )
