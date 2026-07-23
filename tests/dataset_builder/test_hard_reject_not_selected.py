from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cv_preprocess.application.common import SplitPlan
from cv_preprocess.application.select import select_dataset
from cv_preprocess.catalog.models import ClipDisposition

from tests.dataset_builder.conftest_helpers import make_synthetic_catalog, selection_pipeline_config


def test_hard_rejected_never_selected(tmp_path: Path) -> None:
    rows = [
        {
            "clip_id": "eligible_a",
            "speaker_id": "spk_a",
            "phonemes": "a i u",
            "text_norm": "あいう",
            "duration_sec": 2.0,
        },
        {
            "clip_id": "hard_reject_b",
            "speaker_id": "spk_b",
            "phonemes": "e o",
            "text_norm": "えお",
            "duration_sec": 2.0,
            "disposition": ClipDisposition.HARD_REJECTED.value,
            "reject_reason": "quality",
        },
        {
            "clip_id": "eligible_c",
            "speaker_id": "spk_c",
            "phonemes": "ka ki",
            "text_norm": "かき",
            "duration_sec": 2.0,
        },
    ]
    config = selection_pipeline_config(tmp_path, target_duration_hours=0.01)
    catalog = make_synthetic_catalog(tmp_path, rows)
    overrides_path = catalog.work_dir / "overrides.jsonl"
    overrides_path.write_text(
        json.dumps({"clip_id": "hard_reject_b", "action": "force_include"}) + "\n",
        encoding="utf-8",
    )

    result = select_dataset(config, catalog, SplitPlan(catalog=catalog, protocol="unseen_speaker"))
    plan = pl.read_parquet(catalog.work_dir / "plans" / "selection_plan.parquet")
    selected = set(result.selected_clip_ids)
    hard_rejected_in_plan = set(
        plan.filter(pl.col("disposition") == ClipDisposition.HARD_REJECTED.value)["clip_id"].to_list()
    )

    assert selected.isdisjoint({"hard_reject_b"})
    assert hard_rejected_in_plan == set()
    assert ClipDisposition.HARD_REJECTED.value not in set(plan["disposition"].to_list())
