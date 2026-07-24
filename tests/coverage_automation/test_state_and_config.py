"""State persistence and stop-condition tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cv_preprocess.config.coverage import CoverageAutomationConfig, FeatureFamilyTargetConfig
from cv_preprocess.coverage.models import CoverageRunState, StopReason
from cv_preprocess.coverage.runner import _decide_stop
from cv_preprocess.coverage.state import load_run_state, save_run_state


def test_atomic_state_roundtrip(tmp_path: Path) -> None:
    state = CoverageRunState(
        run_id="r1",
        iteration=2,
        status=StopReason.RUNNING,
        analyzed_clip_ids=["a", "b"],
        accepted_clip_ids=["a"],
        rejected_clip_ids=["b"],
        current_coverage={"phoneme:ts": 1},
        remaining_deficits={"phoneme:ts": 2},
        config_hash="abc",
        index_fingerprint="def",
        started_at="t0",
        updated_at="t1",
    )
    save_run_state(tmp_path, state)
    assert (tmp_path / "run-state.json").is_file()
    assert not list(tmp_path.glob("*.partial"))
    loaded = load_run_state(tmp_path)
    assert loaded.run_id == "r1"
    assert loaded.analyzed_clip_ids == ["a", "b"]
    assert loaded.status == StopReason.RUNNING


def test_stop_reasons() -> None:
    cfg = CoverageAutomationConfig(
        enabled=True,
        features={"phoneme": FeatureFamilyTargetConfig(targets={"A": 5})},
        required_features=["phoneme:A"],
        limits={"max_iterations": 2, "max_analyzed_clips": 3, "max_audio_hours": 1.0},  # type: ignore[arg-type]
    )
    state = CoverageRunState(run_id="x", iteration=2, analyzed_clip_ids=["1", "2", "3"])
    assert (
        _decide_stop(
            config=cfg,
            state=state,
            required_deficits={},
            plan_stop_hints=[],
            selected_count=0,
        )
        == StopReason.COMPLETE
    )
    state2 = CoverageRunState(run_id="x", iteration=2)
    assert (
        _decide_stop(
            config=cfg,
            state=state2,
            required_deficits={"phoneme:A": 1},
            plan_stop_hints=[],
            selected_count=1,
        )
        == StopReason.ITERATION_LIMIT_REACHED
    )
    state3 = CoverageRunState(run_id="x", iteration=0, analyzed_clip_ids=["1", "2", "3"])
    assert (
        _decide_stop(
            config=cfg,
            state=state3,
            required_deficits={"phoneme:A": 1},
            plan_stop_hints=[],
            selected_count=1,
        )
        == StopReason.ANALYSIS_BUDGET_EXCEEDED
    )
    assert (
        _decide_stop(
            config=cfg,
            state=CoverageRunState(run_id="x"),
            required_deficits={"phoneme:A": 1},
            plan_stop_hints=[StopReason.CANDIDATE_EXHAUSTED],
            selected_count=0,
        )
        == StopReason.CANDIDATE_EXHAUSTED
    )
    assert (
        _decide_stop(
            config=cfg,
            state=CoverageRunState(run_id="x"),
            required_deficits={"phoneme:A": 1},
            plan_stop_hints=[StopReason.UNREACHABLE],
            selected_count=0,
        )
        == StopReason.UNREACHABLE
    )


def test_config_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        CoverageAutomationConfig.model_validate(
            {
                "enabled": True,
                "batch": {"min_size": 10, "max_size": 5, "safety_factor": 1.0},
            }
        )
    with pytest.raises(ValueError):
        CoverageAutomationConfig.model_validate(
            {
                "enabled": True,
                "features": {"unknown_family": {"targets": {"x": 1}}},
            }
        )
    with pytest.raises(ValueError):
        CoverageAutomationConfig.model_validate(
            {
                "enabled": True,
                "required_features": ["phoneme:ts"],
                "features": {"phoneme": {"default_target": 0, "targets": {}}},
            }
        )
