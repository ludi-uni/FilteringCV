from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from cv_preprocess.application.build import build_dataset
from cv_preprocess.application.common import AnalyzeResult, ScanResult, SplitPlan
from cv_preprocess.application.select import select_dataset
from cv_preprocess.pipeline.export import write_wav_16bit

from tests.dataset_builder.conftest_helpers import make_synthetic_catalog, selection_pipeline_config


def _seed_work_tree(tmp_path: Path) -> tuple:
    config = selection_pipeline_config(
        tmp_path,
        target_duration_hours=0.002,
        extra={
            "dataset_builder": {
                "materialize": {"output_root": tmp_path / "dataset_out", "mode": "copy"},
            }
        },
    )
    work_dir = config.dataset_builder.work_dir
    rows = [
        {
            "clip_id": "clip_resume",
            "speaker_id": "spk_a",
            "phonemes": "a i u",
            "text_norm": "あいう",
            "duration_sec": 1.0,
        }
    ]
    catalog = make_synthetic_catalog(tmp_path, rows, work_dir=work_dir)
    cache_rel = "audio_cache/test/cl/clip_resume.wav"
    cache_abs = work_dir / cache_rel
    cache_abs.parent.mkdir(parents=True, exist_ok=True)
    write_wav_16bit(cache_abs, np.zeros(8000, dtype=np.float32), 16000)
    clips = pl.read_parquet(catalog.clips_path)
    clips = clips.with_columns(pl.lit(cache_rel).alias("audio_cache_rel_path"))
    clips.write_parquet(catalog.clips_path)

    split_plan = SplitPlan(catalog=catalog, protocol="unseen_speaker")
    select_dataset(config, catalog, split_plan)
    return config, catalog


def test_build_skips_analyze_when_catalog_present(tmp_path: Path) -> None:
    config, catalog = _seed_work_tree(tmp_path)
    analyze_calls: list[int] = []

    def _fake_analyze(cfg, *, progress=None, cancellation=None):
        analyze_calls.append(1)
        return AnalyzeResult(catalog=catalog, eligible_count=1, hard_rejected_count=0)

    def _fake_scan(cfg, *, progress=None):
        return ScanResult.model_validate(
            {
                "tsv_path": str(tmp_path / "validated.tsv"),
                "stats": {},
                "rows_after_speaker_filter": 1,
                "rows_after_clip_metadata_filter": 1,
                "merge_filtered_speakers_as_one": False,
                "unique_client_ids_after_filters": 1,
                "unique_client_ids_effective": 1,
                "clip_metadata_filters": {},
                "speaker_filter_list_size": 0,
                "unique_client_ids": 1,
                "sample_client_ids_from_parsed_tsv": ["spk_a"],
                "warnings": [],
                "sample_missing_audio_first10": [],
                "total_missing_audio_sampled": 0,
            }
        )

    with (
        patch("cv_preprocess.application.build.scan_project", side_effect=_fake_scan),
        patch("cv_preprocess.application.build.analyze_project", side_effect=_fake_analyze),
    ):
        build_dataset(config)
        build_dataset(config)

    assert len(analyze_calls) == 0
    assert (config.dataset_builder.work_dir / "run_manifest.json").is_file()
