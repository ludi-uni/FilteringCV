from __future__ import annotations

import numpy as np

from cv_preprocess.audio.pipeline import run_steps_on_array
from cv_preprocess.config import AudioPipelineConfig, TrimSilenceStep


def test_trim_exceeds_max_keep_sets_meta_flag() -> None:
    sr = 1000
    y = np.ones(int(sr * 40), dtype=np.float32) * 0.2
    cfg = AudioPipelineConfig(
        target_sample_rate=sr,
        channels=1,
        audio_pipeline_id="test",
        steps=[
            TrimSilenceStep(
                max_keep_sec=30.0,
                head_tail_db=80.0,
                trim_frame_length=256,
                trim_hop_length=64,
                reject_if_truncated=True,
            ),
        ],
    )
    out, _, meta = run_steps_on_array(y, sr, cfg)
    assert meta.get("trim_exceeds_max_keep_sec") is True
    assert out.shape[0] == int(30 * sr)
