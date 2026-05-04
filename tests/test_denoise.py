import numpy as np

from cv_preprocess.audio.denoise import apply_denoise
from cv_preprocess.config import DenoiseStep


def test_denoise_none_passthrough() -> None:
    y = np.random.default_rng(1).standard_normal(8000).astype(np.float32)
    out = apply_denoise(y, 16000, DenoiseStep(method="none"))
    assert out.shape == y.shape
    assert np.allclose(out, y.astype(np.float32))


def test_spectral_subtract_reduces_broadband_noise() -> None:
    rng = np.random.default_rng(42)
    sr = 22050
    n = sr
    t = np.arange(n, dtype=np.float64) / sr
    tone = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    hiss = 0.06 * rng.standard_normal(n).astype(np.float32)
    pad_n = int(0.25 * sr)
    pad = 0.05 * rng.standard_normal(pad_n).astype(np.float32)
    y = np.concatenate([pad, tone + hiss]).astype(np.float32)

    step = DenoiseStep(
        method="spectral_subtract",
        noise_lead_ms=200.0,
        noise_trail_ms=200.0,
        noise_quiet_rms_percentile=30.0,
        max_noise_frames=64,
        subtract_alpha=1.35,
        spectral_floor=0.1,
    )
    out = apply_denoise(y, sr, step)
    assert out.shape == y.shape
    # 先頭のノイズ区間のエネルギーが下がる（定常ノイズ前提）
    head = slice(0, pad_n)
    assert float(np.mean(out[head] ** 2)) < float(np.mean(y[head] ** 2)) * 0.85


def test_denoise_step_accepts_sgmse_method() -> None:
    step = DenoiseStep(method="sgmse", sgmse_model_source="speechbrain/sgmse-voicebank")
    assert step.method == "sgmse"


def test_denoise_step_accepts_wpe_deepfilternet_method() -> None:
    step = DenoiseStep(method="wpe_deepfilternet", deepfilternet_model="DeepFilterNet2")
    assert step.method == "wpe_deepfilternet"
    assert step.deepfilternet_model == "DeepFilterNet2"


def test_spectral_subtract_reverb_tail_branch() -> None:
    rng = np.random.default_rng(7)
    sr = 16000
    n = int(0.8 * sr)
    y = (0.08 * rng.standard_normal(n)).astype(np.float32)
    step = DenoiseStep(
        method="spectral_subtract",
        reverb_tail_ms=120.0,
        reverb_tail_frame_percentile=50.0,
        reverb_tail_mix=0.4,
        subtract_alpha=1.0,
        spectral_floor=0.08,
    )
    out = apply_denoise(y, sr, step)
    assert out.shape == y.shape
