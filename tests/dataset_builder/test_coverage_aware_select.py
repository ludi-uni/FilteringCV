"""Unit tests for coverage-aware select (targets, reservation, audit, acoustic)."""

from __future__ import annotations

from cv_preprocess.config.coverage import (
    CoverageAutomationConfig,
    FeatureFamilyTargetConfig,
    FeatureTargetSpec,
)
from cv_preprocess.config.dataset_builder import (
    AcousticDiversityConfig,
    CoverageConstraintsConfig,
    DatasetBuilderConfig,
    SelectionConfig,
)
from cv_preprocess.selection.acoustic_diversity import (
    build_acoustic_diversity_state,
    summarize_acoustic_diversity,
)
from cv_preprocess.selection.constraints import (
    ConstraintConfig,
    preserves_required_coverage,
)
from cv_preprocess.selection.coverage_audit import audit_coverage, classify_feature_status
from cv_preprocess.selection.coverage_reservation import reserve_coverage_clips
from cv_preprocess.selection.coverage_targets import (
    build_selection_coverage_constraints,
    compute_effective_targets,
    rarity_weight,
)
from cv_preprocess.selection.local_search import local_search_improve
from cv_preprocess.selection.protocol import ClipFeatures
from cv_preprocess.selection.python_backend import greedy_local_search


def _clip(
    clip_id: str,
    *,
    speaker: str = "spk1",
    duration: float = 1.0,
    quality: float = 90.0,
    coverage: list[str] | None = None,
    acoustic: dict[str, float | None] | None = None,
    phones: list[str] | None = None,
) -> ClipFeatures:
    phones = phones or []
    return ClipFeatures(
        clip_id=clip_id,
        speaker_id=speaker,
        duration_sec=duration,
        quality_score=quality,
        audio_sha256=clip_id,
        sentence_id=clip_id,
        text_norm=clip_id,
        features_by_family={"phone": phones or [c.split(":")[-1] for c in (coverage or []) if c.startswith("phoneme:")]},
        coverage_keys=list(coverage or []),
        acoustic_metrics=dict(acoustic or {}),
    )


def test_feature_target_spec_int_equals_minimum_and_desired() -> None:
    cfg = FeatureFamilyTargetConfig(targets={"v": 5})
    assert cfg.targets["v"].minimum == 5
    assert cfg.targets["v"].desired == 5


def test_feature_target_spec_mapping() -> None:
    cfg = FeatureFamilyTargetConfig(targets={"v": {"minimum": 2, "desired": 5}})
    assert cfg.targets["v"].minimum == 2
    assert cfg.targets["v"].desired == 5


def test_effective_target_corpus_limited() -> None:
    eff_min, eff_des = compute_effective_targets(
        configured_minimum=5,
        configured_desired=5,
        eligible_clip_count=2,
    )
    assert eff_min == 2
    assert eff_des == 2


def test_reservation_picks_multi_feature_clip() -> None:
    coverage = CoverageAutomationConfig(
        enabled=True,
        features={
            "phoneme": FeatureFamilyTargetConfig(
                required=True,
                targets={"A": 1, "B": 1},
            )
        },
    )
    selection = SelectionConfig(
        coverage_constraints=CoverageConstraintsConfig(
            enabled=True,
            required_families=["phoneme"],
            optional_families=[],
        )
    )
    clips = [
        _clip("c1", coverage=["phoneme:A"], phones=["A"]),
        _clip("c2", coverage=["phoneme:B"], phones=["B"]),
        _clip("c3", coverage=["phoneme:A", "phoneme:B"], phones=["A", "B"]),
    ]
    constraints = build_selection_coverage_constraints(coverage, selection, clips)
    result = reserve_coverage_clips(
        clips,
        constraints,
        constraint_config=ConstraintConfig(),
    )
    assert result.reserved_ids == ["c3"]
    assert result.selected_counts["phoneme:A"] == 1
    assert result.selected_counts["phoneme:B"] == 1


def test_rarity_prefers_scarce_feature() -> None:
    assert rarity_weight(2) > rarity_weight(100)


def test_reservation_then_greedy_keeps_reserved_and_respects_duration() -> None:
    coverage = CoverageAutomationConfig(
        enabled=True,
        features={"phoneme": FeatureFamilyTargetConfig(targets={"A": 1})},
    )
    db = DatasetBuilderConfig(
        selection=SelectionConfig(
            reserve_ratio=0.0,
            feature_weights={"phone": 1.0, "quality": 0.2, "speaker_diversity": 0.0},
            coverage_constraints=CoverageConstraintsConfig(
                enabled=True,
                required_families=["phoneme"],
                optional_families=[],
            ),
            local_search={"enabled": False},  # type: ignore[arg-type]
            duration={"target_hours": None, "tolerance_ratio": 0.0},  # type: ignore[arg-type]
        )
    )
    # Rare coverage clip is lower quality; filler clips are higher quality.
    clips = [
        _clip("rare", coverage=["phoneme:A"], phones=["A"], quality=50.0, duration=1.0),
        _clip("f1", coverage=["phoneme:x"], phones=["x"], quality=99.0, duration=1.0),
        _clip("f2", coverage=["phoneme:y"], phones=["y"], quality=98.0, duration=1.0),
        _clip("f3", coverage=["phoneme:z"], phones=["z"], quality=97.0, duration=1.0),
    ]
    result = greedy_local_search(
        clips,
        config=db,
        target_duration_sec=3.0,
        tolerance_ratio=0.0,
        seed=0,
        coverage_config=coverage,
    )
    assert "rare" in result.selected_ids
    assert len(result.selected_ids) == 3
    total_dur = sum(c.duration_sec for c in clips if c.clip_id in result.selected_ids)
    assert total_dur <= 3.0 + 1e-9


def test_local_search_blocks_coverage_breaking_swap() -> None:
    clips = {
        "c1": _clip("c1", coverage=["phoneme:A"], phones=["A"], quality=50.0),
        "c2": _clip("c2", coverage=["phoneme:B"], phones=["B"], quality=99.0),
    }
    selected, reserve, _ = local_search_improve(
        ["c1"],
        ["c2"],
        clips,
        feature_weights={"phone": 1.0, "quality": 1.0},
        diminishing_tau={"phone": 1.0},
        temperatures={"phone": 1.0},
        candidates=list(clips.values()),
        min_utterances_by_family={},
        min_speakers_by_family={},
        constraint_config=ConstraintConfig(),
        swap_patterns=["1v1"],
        max_iterations=20,
        max_wall_sec=5.0,
        required_coverage_targets={"phoneme:A": 1},
    )
    assert "c1" in selected
    assert "c2" not in selected


def test_local_search_allows_coverage_preserving_upgrade() -> None:
    clips = {
        "c1": _clip("c1", coverage=["phoneme:A"], phones=["A"], quality=50.0),
        "c2": _clip("c2", coverage=["phoneme:A"], phones=["A"], quality=99.0),
    }
    selected, reserve, _ = local_search_improve(
        ["c1"],
        ["c2"],
        clips,
        feature_weights={"phone": 1.0, "quality": 5.0},
        diminishing_tau={"phone": 1.0},
        temperatures={"phone": 1.0},
        candidates=list(clips.values()),
        min_utterances_by_family={},
        min_speakers_by_family={},
        constraint_config=ConstraintConfig(),
        swap_patterns=["1v1"],
        max_iterations=50,
        max_wall_sec=5.0,
        required_coverage_targets={"phoneme:A": 1},
    )
    assert selected[0] == "c2"


def test_corpus_limit_status() -> None:
    status, _ = classify_feature_status(
        configured_minimum=5,
        configured_desired=5,
        effective_minimum=2,
        effective_desired=2,
        selected_count=2,
        index_candidate_count=2,
        eligible_candidate_count=2,
        conflict=False,
    )
    assert status == "corpus_limit_satisfied"


def test_not_present_statuses() -> None:
    s1, _ = classify_feature_status(
        configured_minimum=3,
        configured_desired=3,
        effective_minimum=0,
        effective_desired=0,
        selected_count=0,
        index_candidate_count=0,
        eligible_candidate_count=0,
        conflict=False,
    )
    assert s1 == "not_present_in_index"
    s2, _ = classify_feature_status(
        configured_minimum=3,
        configured_desired=3,
        effective_minimum=0,
        effective_desired=0,
        selected_count=0,
        index_candidate_count=3,
        eligible_candidate_count=0,
        conflict=False,
    )
    assert s2 == "not_present_in_eligible"


def test_selection_constraint_conflict() -> None:
    coverage = CoverageAutomationConfig(
        enabled=True,
        features={"phoneme": FeatureFamilyTargetConfig(targets={"A": 2})},
    )
    selection = SelectionConfig(
        coverage_constraints=CoverageConstraintsConfig(
            enabled=True,
            required_families=["phoneme"],
            optional_families=[],
        )
    )
    # Two candidates, same speaker, max 1 clip per speaker → conflict for second.
    clips = [
        _clip("c1", speaker="s1", coverage=["phoneme:A"], phones=["A"]),
        _clip("c2", speaker="s1", coverage=["phoneme:A"], phones=["A"]),
    ]
    constraints = build_selection_coverage_constraints(coverage, selection, clips)
    result = reserve_coverage_clips(
        clips,
        constraints,
        constraint_config=ConstraintConfig(max_clips_per_speaker=1),
    )
    assert result.selected_counts["phoneme:A"] == 1
    assert "phoneme:A" in result.conflict_features or "phoneme:A" in result.unmet_features

    audit = audit_coverage(
        constraints,
        result.reserved_ids,
        {c.clip_id: c for c in clips},
        conflict_features=result.conflict_features or {"phoneme:A"},
    )
    row = next(r for r in audit.rows if r.feature == "A")
    assert row.status == "selection_constraint_conflict"


def test_acoustic_redundancy_prefers_different_vector() -> None:
    clips = [
        _clip("a", acoustic={"duration": 1.0, "snr": 10.0, "silence_ratio": 0.1, "quality_score": 90}),
        _clip("b", acoustic={"duration": 1.0, "snr": 10.0, "silence_ratio": 0.1, "quality_score": 90}),
        _clip("c", acoustic={"duration": 5.0, "snr": 40.0, "silence_ratio": 0.5, "quality_score": 70}),
    ]
    state = build_acoustic_diversity_state(
        clips,
        AcousticDiversityConfig(
            enabled=True,
            backend="lightweight",
            weight=1.0,
            features=["duration", "snr", "silence_ratio", "quality_score"],
        ),
    )
    state.note_selected("a")
    # b is similar to a → higher redundancy penalty than c
    assert state.redundancy_penalty("b") > state.redundancy_penalty("c")
    summary = summarize_acoustic_diversity(state, ["a", "c"])
    assert summary["enabled"] is True
    assert summary["selected_clip_count"] == 2


def test_preserves_required_coverage_helper() -> None:
    removed = _clip("r", coverage=["phoneme:A"])
    added_bad = _clip("x", coverage=["phoneme:B"])
    added_ok = _clip("y", coverage=["phoneme:A"])
    counts = {"phoneme:A": 1}
    assert not preserves_required_coverage(counts, removed, added_bad, {"phoneme:A": 1})
    assert preserves_required_coverage(counts, removed, added_ok, {"phoneme:A": 1})


def test_coverage_aware_disabled_matches_baseline_selection() -> None:
    db = DatasetBuilderConfig(
        selection=SelectionConfig(
            reserve_ratio=0.0,
            feature_weights={"phone": 1.0, "quality": 0.2},
            coverage_constraints=CoverageConstraintsConfig(enabled=False),
            local_search={"enabled": False},  # type: ignore[arg-type]
        )
    )
    clips = [
        _clip("a", phones=["a"], coverage=["phoneme:a"], quality=90),
        _clip("b", phones=["b"], coverage=["phoneme:b"], quality=80),
        _clip("c", phones=["c"], coverage=["phoneme:c"], quality=70),
    ]
    baseline = greedy_local_search(
        clips,
        config=db,
        target_duration_sec=2.0,
        tolerance_ratio=0.0,
        seed=1,
    )
    again = greedy_local_search(
        clips,
        config=db,
        target_duration_sec=2.0,
        tolerance_ratio=0.0,
        seed=1,
    )
    assert baseline.selected_ids == again.selected_ids


def test_resolve_target_spec_from_coverage_config() -> None:
    cfg = CoverageAutomationConfig(
        features={"phoneme": FeatureFamilyTargetConfig(targets={"v": {"minimum": 2, "desired": 5}})}
    )
    spec = cfg.resolve_target_spec("phoneme:v")
    assert isinstance(spec, FeatureTargetSpec)
    assert spec.minimum == 2
    assert cfg.resolve_target("phoneme:v") == 5


def test_audit_reports_written(tmp_path) -> None:
    coverage = CoverageAutomationConfig(
        enabled=True,
        features={"phoneme": FeatureFamilyTargetConfig(targets={"A": 2})},
    )
    selection = SelectionConfig(
        coverage_constraints=CoverageConstraintsConfig(
            enabled=True,
            required_families=["phoneme"],
            optional_families=[],
        ),
        local_search={"enabled": False},  # type: ignore[arg-type]
        reserve_ratio=0.0,
        feature_weights={"phone": 1.0, "quality": 0.1},
    )
    clips = [
        _clip("c1", coverage=["phoneme:A"], phones=["A"]),
        _clip("c2", coverage=["phoneme:A"], phones=["A"]),
        _clip("c3", coverage=["phoneme:x"], phones=["x"]),
    ]
    db = DatasetBuilderConfig(selection=selection, work_dir=tmp_path)
    result = greedy_local_search(
        clips,
        config=db,
        target_duration_sec=2.0,
        tolerance_ratio=0.0,
        seed=0,
        coverage_config=coverage,
        coverage_audit_output=tmp_path / "selection",
        index_candidate_counts={"phoneme:A": 5, "phoneme:missing": 0},
    )
    assert (tmp_path / "selection" / "coverage-audit.json").is_file()
    assert (tmp_path / "selection" / "coverage-audit.csv").is_file()
    assert (tmp_path / "selection" / "coverage-contributions.jsonl").is_file()
    assert (tmp_path / "selection" / "missing-features.json").is_file()
    assert result.coverage_audit is not None
    assert any(cid in result.selected_ids for cid in ("c1", "c2"))
