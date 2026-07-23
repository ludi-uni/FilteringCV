from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from cv_preprocess.application.common import SplitPlan
from cv_preprocess.application.materialize import materialize_dataset
from cv_preprocess.application.select import select_dataset
from cv_preprocess.pipeline.export import write_wav_16bit

from tests.dataset_builder.conftest_helpers import make_synthetic_catalog, selection_pipeline_config


def _prepare_catalog_with_audio(tmp_path: Path) -> tuple:
    config = selection_pipeline_config(
        tmp_path,
        target_duration_hours=0.002,
        extra={"dataset_builder": {"materialize": {"mode": "copy"}}},
    )
    work_dir = config.dataset_builder.work_dir
    rows = [
        {
            "clip_id": "clip_a",
            "speaker_id": "spk_a",
            "phonemes": "a i u",
            "text_norm": "あいう",
            "duration_sec": 1.0,
        },
        {
            "clip_id": "clip_b",
            "speaker_id": "spk_b",
            "phonemes": "e o",
            "text_norm": "えお",
            "duration_sec": 1.0,
        },
    ]
    catalog = make_synthetic_catalog(tmp_path, rows, work_dir=work_dir)
    for row in rows:
        cache_rel = f"audio_cache/test/{row['clip_id'][:2]}/{row['clip_id']}.wav"
        cache_abs = work_dir / cache_rel
        cache_abs.parent.mkdir(parents=True, exist_ok=True)
        write_wav_16bit(cache_abs, np.zeros(16000, dtype=np.float32), 16000)
        clips = pl.read_parquet(catalog.clips_path)
        clips = clips.with_columns(
            pl.when(pl.col("clip_id") == row["clip_id"])
            .then(pl.lit(cache_rel))
            .otherwise(pl.col("audio_cache_rel_path"))
            .alias("audio_cache_rel_path")
        )
        clips.write_parquet(catalog.clips_path)

    split_plan = SplitPlan(catalog=catalog, protocol="unseen_speaker")
    selection = select_dataset(config, catalog, split_plan)
    return config, catalog, selection


def test_materialize_copy_mode(tmp_path: Path) -> None:
    config, catalog, selection = _prepare_catalog_with_audio(tmp_path)
    output_root = tmp_path / "out"
    config.dataset_builder.materialize.output_root = output_root
    config.dataset_builder.materialize.mode = "copy"

    result = materialize_dataset(config, catalog, selection)

    assert result.selected_count == 2
    wav_a = output_root / "wavs" / "clip_a.wav"
    wav_b = output_root / "wavs" / "clip_b.wav"
    assert wav_a.is_file()
    assert wav_b.is_file()
    assert not wav_a.is_symlink()
    metadata = (output_root / "metadata.jsonl").read_text(encoding="utf-8")
    assert "clip_a" in metadata
    assert (output_root / "train.jsonl").is_file()


def test_materialize_symlink_falls_back_to_copy(tmp_path: Path) -> None:
    config, catalog, selection = _prepare_catalog_with_audio(tmp_path)
    output_root = tmp_path / "out_symlink"
    config.dataset_builder.materialize.output_root = output_root
    config.dataset_builder.materialize.mode = "symlink"

    with patch("cv_preprocess.application.materialize.os.symlink", side_effect=OSError("no symlink")):
        result = materialize_dataset(config, catalog, selection)

    wav_path = output_root / "wavs" / "clip_a.wav"
    assert result.selected_count == 2
    assert wav_path.is_file()
    assert not wav_path.is_symlink()
