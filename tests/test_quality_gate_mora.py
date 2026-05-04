import numpy as np

from cv_preprocess.audio.quality_gate import run_quality_gate
from cv_preprocess.config import QualityGateConfig, SnrEstimatorConfig


def _quiet_tone(sr: int, sec: float) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi * 220.0 * sec, int(sr * sec), endpoint=False, dtype=np.float64)
    return (0.08 * np.sin(t)).astype(np.float32)


def test_gate_rejects_when_audio_too_short_for_mora_count() -> None:
    sr = 22050
    y = _quiet_tone(sr, 0.4)
    gate = QualityGateConfig(
        min_duration_sec=0.1,
        max_duration_sec=60.0,
        min_sec_per_mora=0.1,
        mora_gate_relax=1.0,
    )
    snr = SnrEstimatorConfig()
    r = run_quality_gate(y, sr, text_len=10, gate=gate, snr_cfg=snr, mora_count=10)
    assert not r.ok
    assert r.reason == "gate_text_audio_mora"
    assert r.mora_count == 10
    assert r.min_required_duration_sec == 1.0


def test_gate_accepts_when_duration_meets_mora_floor() -> None:
    sr = 22050
    y = _quiet_tone(sr, 1.2)
    gate = QualityGateConfig(
        min_duration_sec=0.1,
        max_duration_sec=60.0,
        min_sec_per_mora=0.1,
        mora_gate_relax=1.0,
    )
    snr = SnrEstimatorConfig()
    r = run_quality_gate(y, sr, text_len=10, gate=gate, snr_cfg=snr, mora_count=10)
    assert r.ok
    assert r.mora_count == 10
    assert r.min_required_duration_sec == 1.0


def test_gate_mora_disabled_when_min_sec_per_mora_null() -> None:
    sr = 22050
    y = _quiet_tone(sr, 0.1)
    gate = QualityGateConfig(
        min_duration_sec=0.05,
        max_duration_sec=60.0,
        min_sec_per_mora=None,
    )
    snr = SnrEstimatorConfig()
    r = run_quality_gate(y, sr, text_len=100, gate=gate, snr_cfg=snr, mora_count=50)
    assert r.ok
    assert r.mora_count == 50
    assert r.min_required_duration_sec is None


def test_gate_skips_mora_when_mora_count_zero() -> None:
    sr = 22050
    y = _quiet_tone(sr, 0.2)
    gate = QualityGateConfig(
        min_duration_sec=0.1,
        max_duration_sec=60.0,
        min_sec_per_mora=0.2,
    )
    snr = SnrEstimatorConfig()
    r = run_quality_gate(y, sr, text_len=1, gate=gate, snr_cfg=snr, mora_count=0)
    assert r.ok
