"""Integration tests for coverage automation with mocked analyze."""

from __future__ import annotations

from pathlib import Path

from cv_preprocess.config.coverage import CoverageAutomationConfig, FeatureFamilyTargetConfig
from cv_preprocess.config.pipeline import PipelineConfig
from cv_preprocess.coverage.indexer import write_index_jsonl
from cv_preprocess.coverage.models import (
    AnalysisBatchResult,
    AnalyzedClip,
    CheapQuality,
    ClipIndexMeta,
    ClipIndexRecord,
    RejectedClip,
)
from cv_preprocess.coverage.planner import plan_coverage
from cv_preprocess.coverage.report import write_coverage_reports
from cv_preprocess.coverage.runner import run_coverage
from cv_preprocess.coverage.state import load_run_state, save_run_state
from cv_preprocess.reports.serializer import write_json_atomic


def _rec(
    clip_id: str,
    features: list[str],
    *,
    client_id: str,
    row_index: int,
    text: str = "こんにちは",
) -> ClipIndexRecord:
    return ClipIndexRecord(
        clip_id=clip_id,
        source_path=f"{clip_id}.mp3",
        client_id=client_id,
        sentence=text,
        normalized_text=text,
        duration_sec=1.5,
        feature_keys=features,
        source_row_index=row_index,
        cheap_quality=CheapQuality(decode_ok=True, peak=0.4, rms=0.05, clipping_ratio=0.0, silence_ratio=0.1),
    )


def test_plan_and_run_with_mock_analyze(tmp_path: Path) -> None:
    records = [
        _rec("c1", ["phoneme:A"], client_id="s1", row_index=0),
        _rec("c2", ["phoneme:B"], client_id="s2", row_index=1),
        _rec("c3", ["phoneme:A", "phoneme:B"], client_id="s3", row_index=2),
        _rec("c4", ["phoneme:A"], client_id="s1", row_index=3),
        _rec("c5", ["phoneme:B"], client_id="s2", row_index=4),
    ]
    index_path = tmp_path / "clip-index.jsonl"
    write_index_jsonl(index_path, records)
    write_json_atomic(
        tmp_path / "clip-index.meta.json",
        ClipIndexMeta(
            created_at="t",
            source_fingerprint="fp",
            normalizer_version="n",
            g2p_version="g",
            config_hash="hash",
            clip_count=len(records),
        ),
    )

    coverage = CoverageAutomationConfig(
        enabled=True,
        features={"phoneme": FeatureFamilyTargetConfig(targets={"A": 2, "B": 2}, default_target=0)},
        required_features=["phoneme:A", "phoneme:B"],
        batch={"min_size": 1, "max_size": 2, "safety_factor": 1.0},  # type: ignore[arg-type]
        limits={"max_iterations": 5, "max_analyzed_clips": 20, "max_audio_hours": 10},  # type: ignore[arg-type]
        pass_probability={"default": 0.9, "prior_strength": 1, "min_probability": 0.05, "max_probability": 0.95},  # type: ignore[arg-type]
    )
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": str(tmp_path / "corpus")},
            "dataset_builder": {"enabled": True, "work_dir": str(tmp_path / "work")},
            "coverage": coverage.model_dump(mode="json"),
        }
    )
    # Force exact coverage object (validators already applied)
    object.__setattr__(cfg, "coverage", coverage)

    plan = plan_coverage(config=coverage, index_records=records, accepted_counts={})
    assert plan.batch_size >= 1
    assert plan.selected[0].clip_id == "c3"

    calls: list[list[str]] = []

    def fake_analyze(clips, config, output_dir, **kwargs):  # noqa: ANN001
        ids = [c.clip_id or c.row.path for c in clips]
        calls.append(ids)
        accepted = []
        rejected = []
        for clip in clips:
            cid = clip.clip_id or "x"
            # Reject one A-only clip to force replanning
            if cid == "c1":
                rejected.append(RejectedClip(clip_id=cid, client_id="s1", reason="gate"))
            else:
                accepted.append(AnalyzedClip(clip_id=cid, client_id="s3", duration_sec=1.0))
        return AnalysisBatchResult(accepted=accepted, rejected=rejected)

    run_dir = tmp_path / "run-001"
    # dry-run first
    dry_state = run_coverage(
        cfg,
        index_path=index_path,
        output_dir=run_dir,
        dry_run=True,
        batch_size=2,
        analyze_fn=fake_analyze,
    )
    assert dry_state.status.value == "dry_run"
    assert (run_dir / "dry-run-plan.json").is_file()
    assert not calls

    run_dir2 = tmp_path / "run-002"
    state = run_coverage(
        cfg,
        index_path=index_path,
        output_dir=run_dir2,
        dry_run=False,
        batch_size=2,
        analyze_fn=fake_analyze,
    )
    assert len(calls) >= 2
    assert state.iteration >= 2
    assert (run_dir2 / "coverage-summary.json").is_file()
    assert (run_dir2 / "report.md").is_file()
    assert (run_dir2 / "report.html").is_file()
    assert state.status in {state.status.COMPLETE, state.status}  # enum present

    # resume should refuse mismatched config hash
    save_run_state(
        run_dir2,
        load_run_state(run_dir2).model_copy(update={"config_hash": "other", "status": state.status.RUNNING}),
    )
    try:
        run_coverage(
            cfg,
            index_path=index_path,
            output_dir=run_dir2,
            resume=True,
            analyze_fn=fake_analyze,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised

    write_coverage_reports(run_dir2, state=load_run_state(run_dir2), config=coverage)
