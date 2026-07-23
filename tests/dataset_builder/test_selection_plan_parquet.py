from __future__ import annotations

from pathlib import Path

import polars as pl

from cv_preprocess.application.select import write_selection_plan_parquet
from cv_preprocess.selection.protocol import SelectionExplanation


def test_write_selection_plan_parquet_handles_late_reserve_reason(tmp_path: Path) -> None:
    """Selected rows with null reserve_reason can exceed Polars infer_schema_length."""
    selected_ids = [f"sel_{i:04d}" for i in range(150)]
    reserve_ids = [f"res_{i:04d}" for i in range(20)]
    explanations = {
        cid: SelectionExplanation(
            selection_score=1.0,
            selected_reason="greedy_marginal_utility",
            rank=i + 1,
        )
        for i, cid in enumerate(selected_ids)
    }
    explanations[reserve_ids[0]] = SelectionExplanation(
        selection_score=0.5,
        reserve_reason="top_reserve_rank",
    )

    path = tmp_path / "selection_plan.parquet"
    write_selection_plan_parquet(
        path,
        selected_ids=selected_ids,
        reserve_ids=reserve_ids,
        explanations=explanations,
        split_plan=None,
    )

    df = pl.read_parquet(path)
    assert df.height == 170
    assert df.schema["reserve_reason"] == pl.Utf8
    assert df.filter(pl.col("clip_id") == reserve_ids[0])["reserve_reason"][0] == "top_reserve_rank"
    assert df.filter(pl.col("disposition") == "selected")["reserve_reason"].null_count() == 150
