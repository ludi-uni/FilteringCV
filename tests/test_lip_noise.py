from typing import Literal

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from cv_preprocess.audio.lip_noise import apply_lip_noise_suppress
from cv_preprocess.audio.lip_noise_repair import apply_lip_noise_repair
from cv_preprocess.config import LipNoiseRepairStep


def _band_rms(y: np.ndarray, sr: int, lo: float, hi: float) -> float:
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    b = sosfilt(sos, y.astype(np.float64))
    return float(np.sqrt(np.mean(b**2) + 1e-20))


def test_lip_noise_suppress_reduces_short_hf_burst() -> None:
    sr = 22050
    n_body = int(1.2 * sr)
    rng = np.random.default_rng(42)
    body = (0.06 * np.sin(2 * np.pi * 180.0 * np.arange(n_body, dtype=np.float64) / sr)).astype(np.float32)
    body += 0.004 * rng.standard_normal(n_body).astype(np.float32)
    pop_n = 320
    pop = 0.55 * rng.standard_normal(pop_n).astype(np.float32)
    y = np.concatenate([body[: n_body // 2], pop, body[n_body // 2 :]]).astype(np.float32)

    lo, hi = 1600.0, min(9000.0, sr / 2.0 - 400.0)
    rms_before = _band_rms(y[n_body // 2 : n_body // 2 + pop_n + 256], sr, lo, hi)

    y2 = apply_lip_noise_suppress(
        y,
        sr,
        spike_ratio=5.0,
        max_burst_frames=8,
        mag_gain=0.35,
        median_kernel_frames=9,
        temporal_smooth_frames=3,
    )
    rms_after = _band_rms(y2[n_body // 2 : n_body // 2 + pop_n + 256], sr, lo, hi)

    assert rms_after < rms_before * 0.82
    assert np.isfinite(y2).all()


def test_lip_noise_suppress_short_signal_noop() -> None:
    sr = 16000
    y = np.zeros(200, dtype=np.float32)
    y2 = apply_lip_noise_suppress(y, sr)
    assert y2.shape == y.shape


def test_lip_noise_repair_reduces_short_impulse() -> None:
    sr = 22050
    n = int(1.2 * sr)
    rng = np.random.default_rng(0)
    y = (1.2e-4 * rng.standard_normal(n)).astype(np.float32)
    mid = n // 2
    y[mid : mid + 9] += 0.52 * np.hanning(9).astype(np.float32)

    win = 120
    peak_before = float(np.max(np.abs(y[mid - win : mid + win])))

    y2, meta = apply_lip_noise_repair(
        y,
        sr,
        rms_ratio_threshold=3.0,
        zcr_ratio_threshold=1.65,
        crest_factor_threshold=3.8,
        flux_ratio_threshold=None,
        max_repairs_per_clip=8,
    )
    peak_after = float(np.max(np.abs(y2[mid - win : mid + win])))

    assert meta["lip_noise_repair_events"] >= 1
    assert meta["lip_noise_repair_samples"] > 0
    assert peak_after < peak_before * 0.55
    assert np.isfinite(y2).all()


def test_lip_noise_repair_short_clip_returns_meta() -> None:
    y = np.zeros(40, dtype=np.float32)
    y2, meta = apply_lip_noise_repair(y, 22050)
    assert y2.shape == y.shape
    assert meta["lip_noise_repair_events"] == 0


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_lip_noise_repair_interpolation_modes(interpolation: Literal["linear", "cubic"]) -> None:
    sr = 16000
    n = 8000
    rng = np.random.default_rng(1)
    y = (1e-4 * rng.standard_normal(n)).astype(np.float32)
    y[4000:4010] += 0.55 * np.hanning(10).astype(np.float32)
    y2, _ = apply_lip_noise_repair(
        y,
        sr,
        interpolation=interpolation,
        rms_ratio_threshold=2.8,
        crest_factor_threshold=3.5,
        zcr_ratio_threshold=1.6,
        flux_ratio_threshold=2.0,
    )
    assert y2.shape == y.shape
    assert np.isfinite(y2).all()


def test_lip_noise_repair_step_config_flux_disable() -> None:
    s = LipNoiseRepairStep.model_validate({"type": "lip_noise_repair", "flux_ratio_threshold": 0})
    assert s.flux_ratio_threshold == 0
