from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cv_preprocess.reports.coverage import compute_coverage_summary
from cv_preprocess.reports.models import CatalogReport
from cv_preprocess.reports.rejection import compute_rejection_summary
from cv_preprocess.reports.serializer import write_json_atomic


def test_write_json_atomic_round_trip(tmp_path: Path) -> None:
    clips = pl.DataFrame(
        {
            "clip_id": ["a", "b"],
            "disposition": ["eligible", "hard_rejected"],
            "reject_reason": [None, "missing_audio"],
            "speaker_id": ["spk1", "spk1"],
            "duration_sec": [1.0, None],
            "quality_score": [0.9, None],
            "biphones": [[], []],
            "triphones": [[], []],
            "moras": [["こ", "ん"], []],
            "fullcontext_labels": [None, None],
            "phonemes": [None, None],
        }
    )
    feature_counts = pl.DataFrame(
        {
            "feature_type": ["mora"],
            "feature": ["こ"],
            "count": [1],
            "speaker_count": [1],
            "utterance_count": [1],
        }
    )
    report = CatalogReport(
        coverage=compute_coverage_summary(clips, feature_counts),
        rejection=compute_rejection_summary(clips),
    )
    out_path = tmp_path / "reports" / "catalog_report.json"
    write_json_atomic(out_path, report)
    assert out_path.is_file()
    assert not out_path.with_suffix(out_path.suffix + ".partial").exists()

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["rejection"]["hard_rejected_count"] == 1
    assert loaded["coverage"]["eligible_clips"] == 1
    assert loaded["coverage"]["entries"][0]["feature"] == "こ"
