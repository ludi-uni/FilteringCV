"""Estimate quality-gate pass probability without ML models."""

from __future__ import annotations

from collections.abc import Mapping

from cv_preprocess.config.coverage import PassProbabilityConfig
from cv_preprocess.coverage.models import CheapQuality, ClipIndexRecord, SpeakerPassStats


def bayesian_smooth_rate(
    *,
    passes: int,
    attempts: int,
    global_rate: float,
    prior_strength: float,
) -> float:
    return (passes + prior_strength * global_rate) / (attempts + prior_strength)


def clip_probability(
    record: ClipIndexRecord,
    *,
    speaker_stats: Mapping[str, SpeakerPassStats],
    global_attempts: int,
    global_passes: int,
    config: PassProbabilityConfig,
) -> float:
    if global_attempts > 0:
        global_rate = global_passes / global_attempts
    else:
        global_rate = config.default

    stats = speaker_stats.get(record.client_id)
    if stats is not None and stats.attempts > 0:
        rate = bayesian_smooth_rate(
            passes=stats.passes,
            attempts=stats.attempts,
            global_rate=global_rate,
            prior_strength=config.prior_strength,
        )
    else:
        rate = global_rate

    rate = _apply_cheap_quality_adjustment(rate, record.cheap_quality, record)
    return float(min(config.max_probability, max(config.min_probability, rate)))


def _apply_cheap_quality_adjustment(
    rate: float,
    quality: CheapQuality,
    record: ClipIndexRecord,
) -> float:
    adjusted = rate
    if not quality.decode_ok:
        return 0.0
    if quality.clipping_ratio is not None and quality.clipping_ratio > 0.01:
        adjusted *= 0.5
    if quality.silence_ratio is not None and quality.silence_ratio > 0.6:
        adjusted *= 0.7
    if quality.rms is not None and (quality.rms < 0.005 or quality.rms > 0.4):
        adjusted *= 0.75
    if record.duration_sec is not None:
        if record.duration_sec < 0.4 or record.duration_sec > 20.0:
            adjusted *= 0.8
    if record.down_votes is not None and record.down_votes > 0:
        adjusted *= 0.85
    if record.up_votes is not None and record.up_votes >= 2:
        adjusted = min(1.0, adjusted * 1.05)
    return adjusted
