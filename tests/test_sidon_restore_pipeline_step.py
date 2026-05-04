from __future__ import annotations

import numpy as np

from cv_preprocess.audio.pipeline import run_steps_on_array
from cv_preprocess.config import AudioPipelineConfig, SidonRestoreStep


def test_sidon_restore_step_parse_defaults() -> None:
    st = SidonRestoreStep.model_validate({"type": "sidon_restore"})
    assert st.type == "sidon_restore"
    assert st.enabled is True
    assert "sarulab-speech" in st.hf_repo_id


def test_run_steps_sidon_restore_skipped_no_import_when_disabled() -> None:
    sr = 24_000
    n = int(sr * 0.2)
    y = (0.1 * np.sin(2 * np.pi * 300.0 * np.arange(n, dtype=np.float64) / sr)).astype(np.float32)
    cfg = AudioPipelineConfig(
        target_sample_rate=sr,
        audio_pipeline_id="test",
        steps=[
            SidonRestoreStep(enabled=False),
        ],
    )
    y2, sr2, meta = run_steps_on_array(y, sr, cfg)
    assert sr2 == sr
    assert y2.shape == y.shape
    assert meta["steps_trace"] == [{"type": "sidon_restore", "skipped": True}]
