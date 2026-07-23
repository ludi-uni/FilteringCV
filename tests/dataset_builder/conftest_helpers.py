from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from cv_preprocess.catalog.models import CatalogRef, ClipDisposition
from cv_preprocess.catalog.writer import write_clips_parquet
from cv_preprocess.config import PipelineConfig


def make_synthetic_catalog(
    tmp_path: Path,
    rows: list[dict],
    *,
    work_dir: Path | None = None,
) -> CatalogRef:
    work = work_dir or (tmp_path / "work")
    catalog_dir = work / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    normalized_rows: list[dict] = []
    for index, row in enumerate(rows):
        normalized_rows.append(
            {
                "clip_id": row["clip_id"],
                "source_release": row.get("source_release", "synthetic"),
                "normalized_relative_source_path": row.get(
                    "normalized_relative_source_path", f"clips/{row['clip_id']}.wav"
                ),
                "source_row_index": row.get("source_row_index", index),
                "audio_sha256": row.get("audio_sha256", f"sha-{row['clip_id']}"),
                "text_raw": row.get("text_raw", row.get("text_norm", "")),
                "text_norm": row.get("text_norm", ""),
                "speaker_id": row.get("speaker_id", "spk"),
                "sentence_id": row.get("sentence_id", f"sent-{row['clip_id']}"),
                "disposition": row.get("disposition", ClipDisposition.ELIGIBLE.value),
                "reject_reason": row.get("reject_reason"),
                "duration_sec": float(row.get("duration_sec", 1.0)),
                "quality_score": row.get("quality_score", 80.0),
                "estimated_snr_db": row.get("estimated_snr_db", 20.0),
                "silence_ratio": row.get("silence_ratio", 0.1),
                "phonemes": row.get("phonemes", ""),
                "feature_source": row.get("feature_source", "text_g2p"),
                "pipeline_hash": row.get("pipeline_hash", "test"),
                "audio_cache_rel_path": row.get("audio_cache_rel_path"),
                "split": row.get("split"),
                "duplicate_group_ids": row.get("duplicate_group_ids"),
                "selection_rank": row.get("selection_rank"),
                "selection_utility": row.get("selection_utility"),
                "override_flags": row.get("override_flags"),
                "analyzed_at": row.get("analyzed_at", now),
            }
        )
    clips_path = catalog_dir / "clips.parquet"
    write_clips_parquet(clips_path, pl.DataFrame(normalized_rows))
    return CatalogRef(work_dir=work, clips_path=clips_path)


def selection_pipeline_config(
    tmp_path: Path,
    *,
    target_duration_hours: float = 0.01,
    seed: int = 42,
    extra: dict | None = None,
) -> PipelineConfig:
    payload: dict = {
        "input": {"corpus_root": tmp_path, "clip_tsv": "validated.tsv"},
        "dataset_builder": {
            "enabled": True,
            "work_dir": tmp_path / "work",
            "target_duration_hours": target_duration_hours,
            "random_seed": seed,
            "selection": {
                "reserve_ratio": 0.2,
                "feature_weights": {
                    "phone": 1.0,
                    "speaker_diversity": 0.5,
                    "quality": 0.1,
                },
                "local_search": {"enabled": False},
            },
        },
    }
    if extra:
        payload["dataset_builder"].update(extra.get("dataset_builder", {}))
    return PipelineConfig.model_validate(payload)
