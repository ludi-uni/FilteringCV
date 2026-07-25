"""analyze_project reuses catalog rows produced by coverage pre-pass."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl

from cv_preprocess.application.analyze import analyze_project
from cv_preprocess.catalog.cache import pipeline_cache_key
from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.catalog.reader import read_clips
from cv_preprocess.catalog.writer import write_catalog_bundle
from cv_preprocess.config import PipelineConfig


def _cfg(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "input": {
                "corpus_root": tmp_path,
                "clip_tsv": "validated.tsv",
                "locale_expected": "ja",
            },
            "text": {"require_japanese": True, "phonemize": False},
            "quality_gate": {"min_duration_sec": 0.2, "max_duration_sec": 60.0},
            "audio_pipeline": {
                "target_sample_rate": 22050,
                "steps": [{"type": "resample", "sr": 22050}],
            },
            "dataset_builder": {"enabled": True, "work_dir": tmp_path / "work"},
        }
    )


def test_analyze_reuses_existing_catalog_rows(tmp_path: Path) -> None:
    (tmp_path / "clips").mkdir()
    (tmp_path / "validated.tsv").write_text(
        "client_id\tpath\tsentence\n"
        "spk\ta.wav\tこんにちは\n"
        "spk\tb.wav\tおはよう\n",
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path)
    pipeline_hash = pipeline_cache_key(cfg)

    prior_a = {
        "clip_id": "clip-a",
        "source_release": "test",
        "normalized_relative_source_path": "a.wav",
        "source_row_index": 0,
        "audio_sha256": "x",
        "text_raw": "こんにちは",
        "text_norm": "こんにちは",
        "speaker_id": "spk",
        "sentence_id": None,
        "pipeline_hash": pipeline_hash,
        "split": None,
        "duplicate_group_ids": None,
        "selection_rank": None,
        "selection_utility": None,
        "override_flags": None,
        "analyzed_at": "t",
        "phonemes": None,
        "feature_source": None,
        "biphones": None,
        "triphones": None,
        "moras": None,
        "fullcontext_labels": None,
        "analysis_warnings": None,
        "audio_cache_rel_path": "audio_cache/x.wav",
        "duration_sec": 1.0,
        "quality_score": 0.9,
        "estimated_snr_db": 20.0,
        "silence_ratio": 0.1,
        "reject_reason": None,
        "disposition": ClipDisposition.ELIGIBLE.value,
    }
    write_catalog_bundle(
        cfg.dataset_builder.work_dir,
        pl.DataFrame([prior_a]),
        manifest={"partial_analyze": True, "total_clips": 1, "pipeline_hash": pipeline_hash},
    )

    calls: list[str] = []

    def _fake_analyze_clip(row, **kwargs):  # noqa: ANN001
        calls.append(row.path)
        out = dict(prior_a)
        out["clip_id"] = f"clip-{row.path}"
        out["normalized_relative_source_path"] = row.path
        out["source_row_index"] = kwargs["source_row_index"]
        out["text_raw"] = row.sentence
        out["text_norm"] = row.sentence
        out["disposition"] = ClipDisposition.ELIGIBLE.value
        from cv_preprocess.application.analyze import ClipAnalyzeOutcome

        return ClipAnalyzeOutcome(row=out, disposition=ClipDisposition.ELIGIBLE)

    with patch("cv_preprocess.application.analyze.analyze_clip_with_gates", side_effect=_fake_analyze_clip):
        result = analyze_project(cfg, reuse_existing=True)

    assert calls == ["b.wav"]
    assert any("reused_existing_catalog_rows=1" in w for w in result.warnings)
    clips_df = read_clips(result.catalog.resolved_clips_path())
    assert clips_df.height == 2


def test_build_runs_coverage_before_analyze_when_enabled(tmp_path: Path) -> None:
    from cv_preprocess.application.build import build_dataset
    from cv_preprocess.application.common import (
        AnalyzeResult,
        AuditReport,
        MaterializeResult,
        ScanResult,
        SelectionPlan,
        SplitPlan,
    )
    from cv_preprocess.catalog.models import CatalogRef

    cfg = _cfg(tmp_path)
    object.__setattr__(
        cfg,
        "coverage",
        cfg.coverage.model_copy(update={"enabled": True, "insert_before_analyze": True}),
    )
    order: list[str] = []
    reuse_flags: list[bool] = []

    def _scan(*_a, **_k):
        order.append("scan")
        return ScanResult.model_validate(
            {
                "tsv_path": str(tmp_path / "validated.tsv"),
                "stats": {},
                "rows_after_speaker_filter": 0,
                "rows_after_clip_metadata_filter": 0,
                "merge_filtered_speakers_as_one": False,
                "unique_client_ids_after_filters": 0,
                "unique_client_ids_effective": 0,
                "clip_metadata_filters": {},
                "speaker_filter_list_size": 0,
                "unique_client_ids": 0,
                "sample_client_ids_from_parsed_tsv": [],
                "warnings": [],
                "sample_missing_audio_first10": [],
                "total_missing_audio_sampled": 0,
            }
        )

    def _coverage(*_a, **_k):
        order.append("coverage")
        return {"status": "complete", "analyzed": 0, "accepted": 0, "clip_count": 0}

    catalog = CatalogRef(work_dir=tmp_path / "work")

    def _analyze(*_a, **_k):
        order.append("analyze")
        reuse_flags.append(bool(_k.get("reuse_existing")))
        return AnalyzeResult(catalog=catalog, eligible_count=0, hard_rejected_count=0)

    with (
        patch("cv_preprocess.application.build.scan_project", side_effect=_scan),
        patch("cv_preprocess.application.build._run_coverage_prepass", side_effect=_coverage),
        patch("cv_preprocess.application.build.analyze_project", side_effect=_analyze),
        patch(
            "cv_preprocess.application.build.plan_dataset_split",
            return_value=SplitPlan(catalog=catalog, protocol="unseen_speaker"),
        ),
        patch(
            "cv_preprocess.application.build.select_dataset",
            return_value=SelectionPlan(catalog=catalog),
        ),
        patch(
            "cv_preprocess.application.build.materialize_dataset",
            return_value=MaterializeResult(output_root=str(tmp_path / "out"), selected_count=0),
        ),
        patch(
            "cv_preprocess.application.build.audit_dataset",
            return_value=AuditReport(catalog=catalog, passed=True),
        ),
        patch("cv_preprocess.application.build.load_catalog", return_value=catalog),
    ):
        build_dataset(cfg, force=False)

    assert order == ["scan", "coverage", "analyze"]
    assert reuse_flags == [True]
