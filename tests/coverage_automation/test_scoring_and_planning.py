"""Deficit / rarity / pass-probability / scoring / planning unit tests."""

from __future__ import annotations

import math

import pytest

from cv_preprocess.config.coverage import CoverageAutomationConfig, FeatureFamilyTargetConfig
from cv_preprocess.coverage.deficits import compute_deficits, remaining_required_deficits
from cv_preprocess.coverage.models import CheapQuality, ClipIndexRecord, SpeakerPassStats
from cv_preprocess.coverage.pass_probability import bayesian_smooth_rate, clip_probability
from cv_preprocess.coverage.planner import plan_coverage, select_batch
from cv_preprocess.coverage.scorer import rarity_weight, score_candidate


def _record(clip_id: str, features: list[str], *, client_id: str = "s1", duration: float = 1.0, text: str = "a") -> ClipIndexRecord:
    return ClipIndexRecord(
        clip_id=clip_id,
        source_path=f"{clip_id}.mp3",
        client_id=client_id,
        sentence=text,
        normalized_text=text,
        duration_sec=duration,
        feature_keys=features,
        cheap_quality=CheapQuality(decode_ok=True, peak=0.5, rms=0.05, clipping_ratio=0.0, silence_ratio=0.1),
    )


def _config_ab() -> CoverageAutomationConfig:
    return CoverageAutomationConfig(
        enabled=True,
        features={
            "phoneme": FeatureFamilyTargetConfig(
                enabled=True,
                default_target=0,
                targets={"A": 1, "B": 1},
            )
        },
        required_features=["phoneme:A", "phoneme:B"],
        batch={"min_size": 1, "max_size": 10, "safety_factor": 1.0},  # type: ignore[arg-type]
        diversity={"speaker_penalty": 0.5, "duplicate_text_penalty": 1.0, "max_per_speaker_per_batch": 5},  # type: ignore[arg-type]
    )


def test_deficits_under_exact_over() -> None:
    targets = {"phoneme:A": 5, "phoneme:B": 2}
    assert compute_deficits(targets, {"phoneme:A": 3, "phoneme:B": 2}) == {"phoneme:A": 2, "phoneme:B": 0}
    assert compute_deficits(targets, {"phoneme:A": 5, "phoneme:B": 2}) == {"phoneme:A": 0, "phoneme:B": 0}
    assert compute_deficits(targets, {"phoneme:A": 9, "phoneme:B": 9}) == {"phoneme:A": 0, "phoneme:B": 0}


def test_required_optional_deficits() -> None:
    cfg = CoverageAutomationConfig(
        enabled=True,
        features={
            "phoneme": FeatureFamilyTargetConfig(targets={"ts": 10, "dy": 5}, default_target=0),
        },
        required_features=["phoneme:ts"],
        optional_features=["phoneme:dy"],
    )
    deficits = compute_deficits(cfg.iter_active_targets(), {"phoneme:ts": 1, "phoneme:dy": 0})
    required = remaining_required_deficits(cfg, deficits)
    assert "phoneme:ts" in required
    assert "phoneme:dy" not in required


def test_rarity_weight_no_div_zero_and_inverse() -> None:
    assert rarity_weight(0) == 1.0
    assert rarity_weight(3) == 1.0 / math.sqrt(4)
    assert rarity_weight(1) > rarity_weight(100)


def test_pass_probability_smoothing_and_clamps() -> None:
    cfg = CoverageAutomationConfig(enabled=True).pass_probability
    smooth = bayesian_smooth_rate(passes=1, attempts=1, global_rate=0.5, prior_strength=10)
    assert 0.5 < smooth < 1.0

    record = _record("c1", ["phoneme:A"])
    no_hist = clip_probability(
        record,
        speaker_stats={},
        global_attempts=0,
        global_passes=0,
        config=cfg,
    )
    assert cfg.min_probability <= no_hist <= cfg.max_probability

    many = clip_probability(
        record,
        speaker_stats={"s1": SpeakerPassStats(attempts=100, passes=100)},
        global_attempts=200,
        global_passes=100,
        config=cfg,
    )
    assert many == pytest.approx(cfg.max_probability)


def test_score_zero_without_deficit_and_multi_feature_priority() -> None:
    cfg = _config_ab()
    targets = cfg.iter_active_targets()
    deficits = {"phoneme:A": 1, "phoneme:B": 1}
    pool = {"phoneme:A": 10, "phoneme:B": 2}
    none = score_candidate(
        _record("x", ["phoneme:Z"]),
        deficits=deficits,
        targets=targets,
        pool_counts=pool,
        selected_batch=[],
        speaker_stats={},
        global_attempts=0,
        global_passes=0,
        config=cfg,
    )
    assert none.score == 0.0

    single = score_candidate(
        _record("a", ["phoneme:A"]),
        deficits=deficits,
        targets=targets,
        pool_counts=pool,
        selected_batch=[],
        speaker_stats={},
        global_attempts=0,
        global_passes=0,
        config=cfg,
    )
    multi = score_candidate(
        _record("ab", ["phoneme:A", "phoneme:B"]),
        deficits=deficits,
        targets=targets,
        pool_counts=pool,
        selected_batch=[],
        speaker_stats={},
        global_attempts=0,
        global_passes=0,
        config=cfg,
    )
    assert multi.score > single.score


def test_long_and_low_pass_reduce_score() -> None:
    cfg = _config_ab()
    targets = cfg.iter_active_targets()
    deficits = {"phoneme:A": 1}
    pool = {"phoneme:A": 5}
    short = score_candidate(
        _record("s", ["phoneme:A"], duration=1.0),
        deficits=deficits,
        targets=targets,
        pool_counts=pool,
        selected_batch=[],
        speaker_stats={},
        global_attempts=10,
        global_passes=9,
        config=cfg,
    )
    long = score_candidate(
        _record("l", ["phoneme:A"], duration=30.0),
        deficits=deficits,
        targets=targets,
        pool_counts=pool,
        selected_batch=[],
        speaker_stats={},
        global_attempts=10,
        global_passes=9,
        config=cfg,
    )
    assert short.score > long.score

    low = score_candidate(
        _record("low", ["phoneme:A"], client_id="bad"),
        deficits=deficits,
        targets=targets,
        pool_counts=pool,
        selected_batch=[],
        speaker_stats={"bad": SpeakerPassStats(attempts=20, passes=0)},
        global_attempts=20,
        global_passes=0,
        config=cfg,
    )
    assert low.score < short.score


def test_batch_prefers_multi_feature_clip() -> None:
    cfg = _config_ab()
    targets = cfg.iter_active_targets()
    deficits = {"phoneme:A": 1, "phoneme:B": 1}
    candidates = [
        _record("clip1", ["phoneme:A"]),
        _record("clip2", ["phoneme:A"]),
        _record("clip3", ["phoneme:A", "phoneme:B"]),
    ]
    selected, _ = select_batch(
        candidates,
        deficits=deficits,
        targets=targets,
        config=cfg,
        batch_limit=1,
    )
    assert selected[0].clip_id == "clip3"


def test_batch_covers_both_features() -> None:
    cfg = CoverageAutomationConfig(
        enabled=True,
        features={"phoneme": FeatureFamilyTargetConfig(targets={"A": 2, "B": 2}, default_target=0)},
        required_features=["phoneme:A", "phoneme:B"],
        batch={"min_size": 1, "max_size": 4, "safety_factor": 1.0},  # type: ignore[arg-type]
    )
    targets = cfg.iter_active_targets()
    deficits = {"phoneme:A": 2, "phoneme:B": 2}
    candidates = [
        _record("clip1", ["phoneme:A"]),
        _record("clip2", ["phoneme:A"]),
        _record("clip3", ["phoneme:B"]),
        _record("clip4", ["phoneme:B"]),
    ]
    selected, _ = select_batch(candidates, deficits=deficits, targets=targets, config=cfg, batch_limit=4)
    selected_features = {f for clip in selected for f in clip.feature_keys}
    assert "phoneme:A" in selected_features
    assert "phoneme:B" in selected_features


def test_diversity_speaker_and_text_penalty() -> None:
    cfg = _config_ab()
    cfg.diversity.max_per_speaker_per_batch = 1
    targets = cfg.iter_active_targets()
    deficits = {"phoneme:A": 3}
    candidates = [
        _record("c1", ["phoneme:A"], client_id="s1", text="same"),
        _record("c2", ["phoneme:A"], client_id="s1", text="same"),
        _record("c3", ["phoneme:A"], client_id="s2", text="other"),
    ]
    selected, _ = select_batch(candidates, deficits=deficits, targets=targets, config=cfg, batch_limit=3)
    assert len([c for c in selected if c.client_id == "s1"]) <= 1
    assert any(c.client_id == "s2" for c in selected)


def test_plan_complete_when_no_deficit() -> None:
    cfg = _config_ab()
    plan = plan_coverage(
        config=cfg,
        index_records=[_record("c1", ["phoneme:A", "phoneme:B"])],
        accepted_counts={"phoneme:A": 1, "phoneme:B": 1},
    )
    assert plan.batch_size == 0
    assert plan.stop_hints[0].value == "complete"
