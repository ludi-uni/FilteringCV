from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cv_preprocess.config import PipelineConfig
from cv_preprocess.pipeline.export import write_wav_16bit
from cv_preprocess.pipeline.secondary import run_secondary


def test_secondary_normalize_and_regate(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    wav_dir = primary / "wavs"
    wav_dir.mkdir(parents=True)
    sr = 22050
    t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
    y = (0.08 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    write_wav_16bit(wav_dir / "cv_ja_000001.wav", y, sr)

    row = {
        "utt_id": "cv_ja_000001",
        "audio_path": "wavs/cv_ja_000001.wav",
        "text_raw": "テスト",
        "text_norm": "テスト",
        "phonemes": None,
        "speaker_id": "sp1",
        "duration_sec": 0.5,
        "silence_ratio": 0.1,
        "estimated_snr_db": 30.0,
        "quality_score": 80.0,
        "quality_tier": "A",
        "trailing_silence_sec": 0.0,
        "split": "train",
        "source_path": "clips/x.mp3",
        "source_release": "test",
        "audio_pipeline_id": "primary",
        "edge_removed_leading_ms": 0.0,
        "edge_removed_trailing_ms": 0.0,
        "edge_click_confidence": None,
        "mora_count": None,
        "min_required_duration_sec": None,
    }
    (primary / "metadata.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    secondary_root = tmp_path / "secondary"
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": tmp_path / "cv"},
            "output": {"root": primary},
            "quality_gate": {
                "min_duration_sec": 0.1,
                "max_duration_sec": 30.0,
                "max_silence_ratio": 0.55,
                "quality_tier_mode": "off",
            },
            "secondary": {
                "output_root": secondary_root,
                "audio_pipeline": {
                    "target_sample_rate": 22050,
                    "audio_pipeline_id": "sec_test",
                    "steps": [
                        {"type": "normalize", "method": "peak", "peak_dbfs": -6.0},
                        {"type": "save_wav", "bit_depth": 16},
                    ],
                },
            },
        }
    )

    rep = run_secondary(cfg, show_progress=False)
    assert rep["accepted"] == 1
    out_m = secondary_root / "metadata.jsonl"
    assert out_m.is_file()
    line = out_m.read_text(encoding="utf-8").strip()
    out_row = json.loads(line)
    assert "primary_quality" in out_row
    assert out_row["primary_quality"]["quality_tier"] == "A"
    assert "secondary_quality" in out_row
    assert out_row["secondary_quality"]["gate_ok"] is True
    assert any(s.get("type") == "normalize" for s in out_row["secondary_corrections"])
    assert (secondary_root / "wavs" / "cv_ja_000001.wav").is_file()
