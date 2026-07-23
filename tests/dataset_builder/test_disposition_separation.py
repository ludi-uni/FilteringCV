from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf

from cv_preprocess.application.analyze import analyze_project
from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.catalog.reader import read_clips
from cv_preprocess.config import PipelineConfig


def _quiet_tone(sr: int, sec: float) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi * 220.0 * sec, int(sr * sec), endpoint=False, dtype=np.float64)
    return (0.08 * np.sin(t)).astype(np.float32)


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")


def _write_tsv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.write_text(
        "client_id\tpath\tsentence\n"
        + "\n".join(f"{cid}\t{rel}\t{text}" for cid, rel, text in rows)
        + "\n",
        encoding="utf-8",
    )


def _analyze_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "input": {
                "corpus_root": tmp_path,
                "clip_tsv": "validated.tsv",
                "locale_expected": "ja",
            },
            "text": {
                "require_japanese": True,
                "phonemize": False,
            },
            "quality_gate": {
                "min_duration_sec": 0.2,
                "max_duration_sec": 60.0,
            },
            "audio_pipeline": {
                "target_sample_rate": 22050,
                "steps": [{"type": "resample", "sr": 22050}],
            },
            "dataset_builder": {
                "enabled": True,
                "work_dir": tmp_path / "work",
                "speaker_constraints": {"max_clips_per_speaker": 1},
            },
            "speakers": {"max_clips_per_speaker": 1},
        }
    )


def test_analyze_disposition_separation(tmp_path: Path) -> None:
    sr = 22050
    clips_dir = tmp_path / "clips"
    good_y = _quiet_tone(sr, 1.0)
    _write_wav(clips_dir / "good.wav", good_y, sr)
    _write_wav(clips_dir / "good2.wav", good_y, sr)
    _write_wav(clips_dir / "short.wav", _quiet_tone(sr, 0.05), sr)
    (clips_dir / "missing.wav").unlink(missing_ok=True)
    _write_tsv(
        tmp_path / "validated.tsv",
        [
            ("speaker_a", "good.wav", "こんにちは"),
            ("speaker_a", "good2.wav", "おはよう"),
            ("speaker_a", "short.wav", "短い"),
            ("speaker_a", "missing.wav", "ない"),
        ],
    )

    cfg = _analyze_config(tmp_path)
    result = analyze_project(cfg)
    clips = read_clips(result.catalog.resolved_clips_path())

    by_path = {
        row["normalized_relative_source_path"]: row
        for row in clips.to_dicts()
    }
    assert by_path["good.wav"]["disposition"] == ClipDisposition.ELIGIBLE.value
    assert by_path["good2.wav"]["disposition"] == ClipDisposition.ELIGIBLE.value
    assert by_path["missing.wav"]["disposition"] == ClipDisposition.HARD_REJECTED.value
    assert by_path["missing.wav"]["reject_reason"] == "missing_audio"
    assert by_path["short.wav"]["disposition"] == ClipDisposition.HARD_REJECTED.value

    dispositions = clips.get_column("disposition").to_list()
    assert ClipDisposition.SELECTED.value not in dispositions
    assert ClipDisposition.RESERVE.value not in dispositions
    assert result.eligible_count == 2
    assert result.hard_rejected_count == 2

    eligible = clips.filter(pl.col("disposition") == ClipDisposition.ELIGIBLE.value)
    assert eligible.filter(pl.col("audio_cache_rel_path").is_not_null()).height == 2
