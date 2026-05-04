from unittest.mock import patch

import numpy as np

from cv_preprocess.audio.quality_gate import measure_trailing_silence_sec, run_quality_gate
from cv_preprocess.config import QualityGateConfig, SnrEstimatorConfig


def test_run_quality_gate_stereo_channel_first_matches_mono() -> None:
    """(C, T) の float でも sliding_window が落ちず、モノラル平均と同等にゲートできる。"""
    sr = 22050
    y_mono = _quiet_tone(sr, 1.0)
    y_st = np.stack([y_mono * 0.95, y_mono * 1.05], axis=0).astype(np.float32)
    gate = QualityGateConfig(
        min_duration_sec=0.2,
        max_duration_sec=60.0,
        quality_tier_mode="off",
    )
    snr = SnrEstimatorConfig()
    r_mono = run_quality_gate(y_mono, sr, text_len=10, gate=gate, snr_cfg=snr, mora_count=None)
    r_st = run_quality_gate(y_st, sr, text_len=10, gate=gate, snr_cfg=snr, mora_count=None)
    assert r_mono.ok and r_st.ok
    assert abs(r_mono.duration_sec - r_st.duration_sec) < 1e-5
    assert abs(r_mono.silence_ratio - r_st.silence_ratio) < 1e-5


def _quiet_tone(sr: int, sec: float) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi * 220.0 * sec, int(sr * sec), endpoint=False, dtype=np.float64)
    return (0.08 * np.sin(t)).astype(np.float32)


def test_min_snr_skipped_when_estimate_returns_none() -> None:
    """推定 SNR が取れない場合は min_estimated_snr_db で落とさない（帯域補完後などで None になりやすい）。"""
    sr = 22050
    y = _quiet_tone(sr, 1.0)
    gate = QualityGateConfig(
        min_duration_sec=0.2,
        max_duration_sec=60.0,
        min_estimated_snr_db=99.0,
    )
    snr = SnrEstimatorConfig()
    with patch("cv_preprocess.audio.quality_gate.estimate_snr_db", return_value=None):
        r = run_quality_gate(y, sr, text_len=10, gate=gate, snr_cfg=snr, mora_count=None)
    assert r.ok
    assert r.estimated_snr_db is None


def test_quality_tier_annotate_fills_score_and_tier() -> None:
    sr = 22050
    y = _quiet_tone(sr, 1.0)
    gate = QualityGateConfig(
        min_duration_sec=0.2,
        max_duration_sec=60.0,
        quality_tier_mode="annotate",
    )
    snr = SnrEstimatorConfig()
    r = run_quality_gate(y, sr, text_len=10, gate=gate, snr_cfg=snr, mora_count=None)
    assert r.ok
    assert r.quality_tier in ("A", "B", "C")
    assert r.quality_score is not None
    assert 0.0 <= r.quality_score <= 100.0


def test_quality_tier_off_leaves_tier_none() -> None:
    sr = 22050
    y = _quiet_tone(sr, 1.0)
    gate = QualityGateConfig(
        min_duration_sec=0.2,
        max_duration_sec=60.0,
        quality_tier_mode="off",
    )
    snr = SnrEstimatorConfig()
    r = run_quality_gate(y, sr, text_len=10, gate=gate, snr_cfg=snr, mora_count=None)
    assert r.ok
    assert r.quality_tier is None
    assert r.quality_score is None


def test_gate_trailing_silence_hard_reject() -> None:
    """長い末尾無音は max_trailing_silence_sec でハード拒否。"""
    sr = 22050
    n_tail = int(0.9 * sr)
    y = np.concatenate(
        [
            _quiet_tone(sr, 0.4),
            np.zeros(n_tail, dtype=np.float32),
        ]
    ).astype(np.float32)
    gate = QualityGateConfig(
        min_duration_sec=0.2,
        max_duration_sec=60.0,
        max_silence_ratio=0.95,
        max_trailing_silence_sec=0.5,
        quality_tier_mode="off",
    )
    snr = SnrEstimatorConfig()
    r = run_quality_gate(y, sr, text_len=10, gate=gate, snr_cfg=snr, mora_count=None)
    assert not r.ok
    assert r.reason == "gate_trailing_silence"
    assert r.trailing_silence_sec > 0.5


def test_reject_c_drops_tier_c() -> None:
    sr = 22050
    y = _quiet_tone(sr, 1.0)
    gate = QualityGateConfig(
        min_duration_sec=0.2,
        max_duration_sec=60.0,
        quality_tier_mode="reject_c",
        quality_tier_a_min_snr_db=75.0,
        quality_tier_b_min_snr_db=70.0,
    )
    snr = SnrEstimatorConfig()
    r = run_quality_gate(y, sr, text_len=10, gate=gate, snr_cfg=snr, mora_count=None)
    assert not r.ok
    assert r.reason == "gate_quality_tier_c"
    assert r.quality_tier == "C"


def test_reject_b_drops_tier_b() -> None:
    sr = 22050
    y = _quiet_tone(sr, 1.0)
    gate = QualityGateConfig(
        min_duration_sec=0.2,
        max_duration_sec=60.0,
        max_silence_ratio=0.55,
        quality_tier_mode="reject_b",
        quality_tier_a_min_snr_db=75.0,
        quality_tier_b_min_snr_db=3.0,
        quality_tier_a_max_silence_ratio=0.01,
        quality_tier_b_max_silence_ratio=0.55,
        quality_tier_a_max_clipping_ratio=0.001,
    )
    snr = SnrEstimatorConfig()
    r = run_quality_gate(y, sr, text_len=10, gate=gate, snr_cfg=snr, mora_count=None)
    assert not r.ok
    assert r.reason in ("gate_quality_tier_b", "gate_quality_tier_c")


def test_default_yaml_quality_gate_model() -> None:
    from pathlib import Path

    from cv_preprocess.config import load_config

    p = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    cfg = load_config(p)
    assert cfg.quality_gate.quality_tier_mode in ("annotate", "reject_c", "reject_b", "off")
    assert cfg.quality_gate.max_silence_ratio >= cfg.quality_gate.quality_tier_b_max_silence_ratio


def test_measure_trailing_silence_after_tone() -> None:
    sr = 8000
    tone = _quiet_tone(sr, 0.4)
    tail = np.zeros(int(0.35 * sr), dtype=np.float32)
    y = np.concatenate([tone, tail])
    gate = QualityGateConfig(min_duration_sec=0.1, max_duration_sec=60.0)
    snr = SnrEstimatorConfig()
    t = measure_trailing_silence_sec(
        y,
        sr,
        frame_ms=snr.frame_ms,
        hop_ms=snr.hop_ms,
        rms_floor=gate.silence_ratio_rms_floor,
        ref_percentile=gate.silence_ratio_ref_percentile,
    )
    assert t >= 0.28


def test_trailing_cap_downgrades_a_to_b() -> None:
    sr = 8000
    tone = _quiet_tone(sr, 1.2)
    tail = np.zeros(int(0.32 * sr), dtype=np.float32)
    y = np.concatenate([tone, tail])
    gate = QualityGateConfig(
        min_duration_sec=0.1,
        max_duration_sec=60.0,
        quality_tier_mode="annotate",
        quality_tier_a_max_trailing_silence_sec=0.25,
        quality_tier_a_min_snr_db=5.0,
        quality_tier_b_min_snr_db=3.0,
        quality_tier_a_max_silence_ratio=0.45,
        quality_tier_b_max_silence_ratio=0.55,
        max_silence_ratio=0.55,
    )
    snr = SnrEstimatorConfig()
    r = run_quality_gate(y, sr, text_len=4, gate=gate, snr_cfg=snr, mora_count=None)
    assert r.ok
    assert r.trailing_silence_sec >= 0.28
    assert r.quality_tier == "B"
