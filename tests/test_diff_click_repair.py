from typing import Literal

import numpy as np
import pytest

from cv_preprocess.audio.diff_click_repair import apply_diff_click_repair
from cv_preprocess.audio.pipeline import run_steps_on_array
from cv_preprocess.config import AudioPipelineConfig, DiffClickRepairStep


def test_diff_click_repair_single_sample_spike() -> None:
    sr = 48000
    n = int(0.5 * sr)
    t = np.arange(n, dtype=np.float64) / sr
    y = (0.08 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    mid = n // 2
    y[mid] += 0.95

    y2, meta = apply_diff_click_repair(y, sr, mad_k=6.0, max_repairs_per_clip=8)
    assert meta["diff_click_repair_events"] >= 1
    assert meta["diff_click_repair_samples"] > 0
    assert float(np.max(np.abs(y2[mid - 4 : mid + 5]))) < float(np.max(np.abs(y[mid - 4 : mid + 5])))
    assert np.isfinite(y2).all()


def test_diff_click_repair_short_clip_noop() -> None:
    y = np.zeros(20, dtype=np.float32)
    y2, meta = apply_diff_click_repair(y, 48000)
    assert y2.shape == y.shape
    assert meta["diff_click_repair_events"] == 0


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_diff_click_repair_interpolation(interpolation: Literal["linear", "cubic"]) -> None:
    sr = 22050
    n = 6000
    rng = np.random.default_rng(3)
    y = (0.02 * rng.standard_normal(n)).astype(np.float32)
    y[3000] += 0.65
    y2, _ = apply_diff_click_repair(y, sr, mad_k=5.0, interpolation=interpolation)
    assert y2.shape == y.shape
    assert np.isfinite(y2).all()


def test_diff_click_repair_pipeline_step() -> None:
    sr = 16000
    y = (0.1 * np.sin(2 * np.pi * 300 * np.arange(8000, dtype=np.float64) / sr)).astype(np.float32)
    y[4002] += 0.8
    cfg = AudioPipelineConfig(
        target_sample_rate=sr,
        steps=[
            DiffClickRepairStep(mad_k=5.0, repair_pad_ms=1.0),
        ],
    )
    y2, out_sr, meta = run_steps_on_array(y, sr, cfg)
    assert out_sr == sr
    assert meta.get("diff_click_repair_events", 0) >= 1
    assert y2.shape == y.shape


def test_diff_click_repair_step_config_require_both_validation() -> None:
    with pytest.raises(ValueError, match="require_both"):
        DiffClickRepairStep(
            use_second_diff=False,
            require_both=True,
        )
