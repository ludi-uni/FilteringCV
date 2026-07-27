"""Tests for piper_plus / Style-Bert-VITS2 trainer exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from cv_preprocess.application.materialize import materialize_dataset
from cv_preprocess.config.dataset_builder import TrainerExportsConfig
from cv_preprocess.export.runner import export_from_materialize_root
from cv_preprocess.pipeline.export import write_wav_16bit

from tests.dataset_builder.test_materialize_modes import _prepare_catalog_with_audio


def test_materialize_writes_trainer_exports(tmp_path: Path) -> None:
    config, catalog, selection = _prepare_catalog_with_audio(tmp_path)
    output_root = tmp_path / "out"
    config.dataset_builder.materialize.output_root = output_root
    config.dataset_builder.materialize.trainer_exports.enabled = True

    result = materialize_dataset(config, catalog, selection)
    assert result.selected_count == 2

    piper_csv = (output_root / "exports" / "piper_plus" / "metadata.csv").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(piper_csv) == 2
    # two speakers → multi-speaker form
    assert piper_csv[0].count("|") == 2
    assert (output_root / "exports" / "piper_plus" / "wav" / "clip_a.wav").is_file()

    esd = (
        output_root / "exports" / "style_bert_vits2" / "Data" / "filteringcv" / "esd.list"
    ).read_text(encoding="utf-8").strip().splitlines()
    assert len(esd) == 2
    parts = esd[0].split("|")
    assert len(parts) == 4
    assert parts[0].endswith(".wav")
    assert parts[2] == "JP"
    assert (
        output_root / "exports" / "style_bert_vits2" / "Data" / "filteringcv" / "raw" / "clip_a.wav"
    ).is_file()


def test_materialize_can_disable_trainer_exports(tmp_path: Path) -> None:
    config, catalog, selection = _prepare_catalog_with_audio(tmp_path)
    output_root = tmp_path / "out"
    config.dataset_builder.materialize.output_root = output_root
    config.dataset_builder.materialize.trainer_exports.enabled = False
    materialize_dataset(config, catalog, selection)
    assert not (output_root / "exports").exists()


def test_piper_single_speaker_two_columns(tmp_path: Path) -> None:
    config, catalog, selection = _prepare_catalog_with_audio(tmp_path)
    # Force both clips to same speaker via re-materialize metadata path: patch catalog speakers
    import polars as pl

    clips = pl.read_parquet(catalog.clips_path)
    clips = clips.with_columns(pl.lit("only_spk").alias("speaker_id"))
    clips.write_parquet(catalog.clips_path)
    # Need re-select so selection still works; reuse existing plan by rewriting metadata via export only
    output_root = tmp_path / "out_single"
    config.dataset_builder.materialize.output_root = output_root
    # Rebuild selection from updated catalog
    from cv_preprocess.application.common import SplitPlan
    from cv_preprocess.application.select import select_dataset

    selection = select_dataset(config, catalog, SplitPlan(catalog=catalog, protocol="unseen_speaker"))
    materialize_dataset(config, catalog, selection)
    lines = (output_root / "exports" / "piper_plus" / "metadata.csv").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert lines
    assert lines[0].count("|") == 1


def test_sbv2_per_speaker_packaging(tmp_path: Path) -> None:
    config, catalog, selection = _prepare_catalog_with_audio(tmp_path)
    output_root = tmp_path / "out"
    config.dataset_builder.materialize.output_root = output_root
    config.dataset_builder.materialize.trainer_exports.style_bert_vits2.packaging = "per_speaker"
    materialize_dataset(config, catalog, selection)
    data_root = output_root / "exports" / "style_bert_vits2" / "Data"
    speakers = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    assert speakers == ["spk_a", "spk_b"]
    assert (data_root / "spk_a" / "esd.list").is_file()
    assert (data_root / "spk_a" / "raw").is_dir()


def test_export_trainer_reexport(tmp_path: Path) -> None:
    config, catalog, selection = _prepare_catalog_with_audio(tmp_path)
    output_root = tmp_path / "out"
    config.dataset_builder.materialize.output_root = output_root
    config.dataset_builder.materialize.trainer_exports.enabled = False
    materialize_dataset(config, catalog, selection)
    assert not (output_root / "exports").exists()

    results = export_from_materialize_root(
        materialize_root=output_root,
        config=TrainerExportsConfig(enabled=True, formats=["piper_plus"]),
        place_mode="copy",
        formats=["piper_plus"],
    )
    assert len(results) == 1
    assert results[0].utterance_count == 2
    assert (output_root / "exports" / "piper_plus" / "metadata.csv").is_file()


def test_optional_resample_changes_rate(tmp_path: Path) -> None:
    config, catalog, selection = _prepare_catalog_with_audio(tmp_path)
    output_root = tmp_path / "out"
    config.dataset_builder.materialize.output_root = output_root
    config.dataset_builder.materialize.trainer_exports.enabled = False
    materialize_dataset(config, catalog, selection)

    results = export_from_materialize_root(
        materialize_root=output_root,
        config=TrainerExportsConfig(
            enabled=True,
            formats=["piper_plus"],
            resample=True,
        ),
        place_mode="copy",
        formats=["piper_plus"],
        resample=True,
    )
    wav = output_root / "exports" / "piper_plus" / "wav" / "clip_a.wav"
    assert wav.is_file()
    info = sf.info(str(wav))
    assert int(info.samplerate) == 22050
    assert results[0].utterance_count == 2
