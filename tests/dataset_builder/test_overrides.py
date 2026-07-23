from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cv_preprocess.application.common import SplitPlan
from cv_preprocess.application.select import select_dataset
from cv_preprocess.catalog.models import ClipDisposition

from tests.dataset_builder.conftest_helpers import make_synthetic_catalog, selection_pipeline_config


def test_overrides_force_include_and_hard_reject(tmp_path: Path) -> None:
    rows = [
        {
            "clip_id": "keep_me",
            "speaker_id": "spk_a",
            "phonemes": "a i u",
            "text_norm": "あいう",
            "duration_sec": 2.0,
        },
        {
            "clip_id": "reject_me",
            "speaker_id": "spk_b",
            "phonemes": "e o",
            "text_norm": "えお",
            "duration_sec": 2.0,
            "disposition": ClipDisposition.HARD_REJECTED.value,
            "reject_reason": "quality",
        },
        {
            "clip_id": "force_me",
            "speaker_id": "spk_c",
            "phonemes": "ka ki",
            "text_norm": "かき",
            "duration_sec": 2.0,
        },
        {
            "clip_id": "exclude_me",
            "speaker_id": "spk_d",
            "phonemes": "ku ke",
            "text_norm": "くけ",
            "duration_sec": 2.0,
        },
    ]
    config = selection_pipeline_config(tmp_path, target_duration_hours=0.002)
    catalog = make_synthetic_catalog(tmp_path, rows)
    overrides_path = catalog.work_dir / "overrides.jsonl"
    overrides_path.write_text(
        "\n".join(
            [
                json.dumps({"clip_id": "force_me", "action": "force_include"}),
                json.dumps({"clip_id": "exclude_me", "action": "force_exclude"}),
                json.dumps({"clip_id": "reject_me", "action": "hard_reject"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = select_dataset(config, catalog, SplitPlan(catalog=catalog, protocol="unseen_speaker"))
    plan = pl.read_parquet(catalog.work_dir / "plans" / "selection_plan.parquet")

    selected = set(result.selected_clip_ids)
    assert "force_me" in selected
    assert "exclude_me" not in selected
    assert "reject_me" not in selected
    assert ClipDisposition.HARD_REJECTED.value not in set(plan["disposition"].to_list())
    assert "force_me" in set(
        plan.filter(pl.col("disposition") == ClipDisposition.SELECTED.value)["clip_id"].to_list()
    )
