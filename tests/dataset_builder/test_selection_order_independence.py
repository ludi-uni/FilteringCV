from __future__ import annotations

import random
from pathlib import Path

from cv_preprocess.application.common import SplitPlan
from cv_preprocess.application.select import select_dataset
from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.selection.python_backend import greedy_local_search
from cv_preprocess.selection.protocol import ClipFeatures

from tests.dataset_builder.conftest_helpers import make_synthetic_catalog, selection_pipeline_config


def _synthetic_rows() -> list[dict]:
    return [
        {
            "clip_id": "clip_z",
            "speaker_id": "spk_a",
            "phonemes": "a b c",
            "text_norm": "あ",
            "duration_sec": 1.0,
        },
        {
            "clip_id": "clip_a",
            "speaker_id": "spk_b",
            "phonemes": "d e f",
            "text_norm": "い",
            "duration_sec": 1.0,
        },
        {
            "clip_id": "clip_m",
            "speaker_id": "spk_c",
            "phonemes": "g h i",
            "text_norm": "う",
            "duration_sec": 1.0,
        },
        {
            "clip_id": "clip_b",
            "speaker_id": "spk_d",
            "phonemes": "j k l",
            "text_norm": "え",
            "duration_sec": 1.0,
        },
    ]


def _clip_features(rows: list[dict], config) -> list[ClipFeatures]:
    from cv_preprocess.application.select import _clip_features_from_catalog
    import polars as pl

    from cv_preprocess.catalog.writer import write_clips_parquet

    df = pl.DataFrame(
        [
            {
                **row,
                "source_release": "synthetic",
                "normalized_relative_source_path": f"{row['clip_id']}.wav",
                "source_row_index": index,
                "audio_sha256": f"sha-{row['clip_id']}",
                "text_raw": row["text_norm"],
                "sentence_id": row["clip_id"],
                "disposition": ClipDisposition.ELIGIBLE.value,
                "reject_reason": None,
                "quality_score": 80.0,
                "estimated_snr_db": 20.0,
                "silence_ratio": 0.1,
                "feature_source": "text_g2p",
                "pipeline_hash": "test",
                "audio_cache_rel_path": None,
                "split": None,
                "duplicate_group_ids": None,
                "selection_rank": None,
                "selection_utility": None,
                "override_flags": None,
                "analyzed_at": "2026-01-01T00:00:00+00:00",
            }
            for index, row in enumerate(rows)
        ]
    )
    return _clip_features_from_catalog(df, config, {}, None)


def test_selection_order_independence(tmp_path: Path) -> None:
    config = selection_pipeline_config(tmp_path, target_duration_hours=0.003)
    catalog = make_synthetic_catalog(tmp_path, _synthetic_rows())
    split_plan = SplitPlan(catalog=catalog, protocol="unseen_speaker")

    baseline = select_dataset(config, catalog, split_plan)

    candidates = _clip_features(_synthetic_rows(), config)
    shuffled = candidates[:]
    random.Random(99).shuffle(shuffled)
    direct = greedy_local_search(
        shuffled,
        config=config.dataset_builder,
        target_duration_sec=0.003 * 3600.0,
        tolerance_ratio=0.5,
        seed=config.dataset_builder.random_seed,
    )

    assert set(baseline.selected_clip_ids) == set(direct.selected_ids)
