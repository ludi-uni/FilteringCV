import numpy as np

from cv_preprocess.audio.edge_ui_click import apply_edge_ui_click
from cv_preprocess.audio.trim_silence import trim_silence


def test_trim_silence_accepts_frame_hop() -> None:
    sr = 8000
    y = np.concatenate(
        [
            np.zeros(500, dtype=np.float32),
            0.35 * np.sin(2 * np.pi * 440.0 * np.arange(2000, dtype=np.float64) / sr).astype(np.float32),
        ]
    )
    out = trim_silence(y, sr, top_db=35.0, frame_length=512, hop_length=128)
    assert out.shape[0] < y.shape[0]
    assert out.shape[0] > 100


def test_trim_silence_trailing_only_keeps_leading_zeros() -> None:
    """末尾だけ librosa の右境界で切り、先頭の無音は残す。"""
    sr = 8000
    y = np.concatenate(
        [
            np.zeros(400, dtype=np.float32),
            0.35 * np.sin(2 * np.pi * 440.0 * np.arange(1800, dtype=np.float64) / sr).astype(np.float32),
            np.zeros(700, dtype=np.float32),
        ]
    )
    out = trim_silence(y, sr, top_db=35.0, frame_length=512, hop_length=128, trim_sides="trailing")
    assert out.shape[0] < y.shape[0]
    assert out.shape[0] > 1800
    assert float(np.mean(np.abs(out[:200]))) < 1e-5


def test_trim_silence_nan_input_does_not_return_empty() -> None:
    """NaN が混ざると librosa trim が [0,0] になり空＋パッドだけの無音になる。有限化して維持する。"""
    sr = 16000
    n = 8000
    y = (0.1 * np.sin(2 * np.pi * 220.0 * np.arange(n, dtype=np.float64) / sr)).astype(np.float32)
    y = y.copy()
    y[100:120] = np.nan
    out = trim_silence(y, sr, top_db=40.0, frame_length=1024, hop_length=256, trim_sides="both")
    assert out.size > 0
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) > 1e-4


def test_trim_silence_trailing_spike_falls_back_when_spike_end_zero() -> None:
    """末尾に短いエネルギー帯のみで spike 探索が last_f=-1 になる場合、空配列にならず librosa 境界を使う。"""
    sr = 48000
    n = sr * 2
    y = np.zeros(n, dtype=np.float32)
    y[-400:] = 0.05
    out = trim_silence(
        y,
        sr,
        top_db=44.0,
        frame_length=1024,
        hop_length=256,
        trim_sides="trailing",
        max_trailing_spike_frames=5,
    )
    assert out.size > 0
    assert float(np.max(np.abs(out))) > 1e-6


def test_trim_silence_trailing_spike_strip_long_quiet_tail() -> None:
    """末尾の無音の直前に短い高 RMS 島があり、通常 trailing では切れないケースを spike 無視で切る。"""
    sr = 16000
    body_n = 5000
    body = (0.12 * np.sin(2 * np.pi * 220.0 * np.arange(body_n, dtype=np.float64) / sr)).astype(np.float32)
    quiet_n = int(0.35 * sr)
    quiet = np.zeros(quiet_n, dtype=np.float32)
    spike = np.zeros(400, dtype=np.float32)
    spike[320:380] = 0.04
    y = np.concatenate([body, quiet, spike]).astype(np.float32)
    plain = trim_silence(
        y, sr, top_db=32.0, frame_length=1024, hop_length=256, trim_sides="trailing", max_trailing_spike_frames=0
    )
    cut = trim_silence(
        y, sr, top_db=32.0, frame_length=1024, hop_length=256, trim_sides="trailing", max_trailing_spike_frames=8
    )
    assert cut.shape[0] < plain.shape[0]
    assert cut.shape[0] < y.shape[0]


def test_edge_ui_click_lead_presilence_skips_attack_without_pre_silence() -> None:
    """先頭に短い高エネルギー帯のみ（その左に低 RMS が続かない）: 先頭無音必須時はミュートしない。"""
    sr = 16000
    rng = np.random.default_rng(42)
    n1 = int(0.03 * sr)
    attack = 0.5 * rng.standard_normal(n1).astype(np.float32)
    n2 = 10000
    t = np.arange(n2, dtype=np.float64) / sr
    body = (0.05 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    y = np.concatenate([attack, body])
    er = apply_edge_ui_click(
        y,
        sr,
        lead_scan_ms=400.0,
        trail_scan_ms=120.0,
        max_transient_ms=55.0,
        peak_above_noise_db=16.0,
        removal="mute_then_trim",
        lead_click_requires_pre_silence_ms=18.0,
    )
    assert er.y.shape[0] == y.shape[0]
    assert er.removed_leading_ms == 0.0
    assert er.confidence_leading == "lead_presilence_short"


def test_edge_ui_click_mute_then_trim_strips_leading_zeros() -> None:
    sr = 16000
    # 先頭に短いクリック状の尖り、その後トーン（クリックは周囲より十分大きく）
    click = np.zeros(1200, dtype=np.float32)
    click[80:140] = 1.0
    tone = 0.06 * np.sin(2 * np.pi * 300.0 * np.arange(5000, dtype=np.float64) / sr).astype(np.float32)
    y = np.concatenate([click, tone])
    er = apply_edge_ui_click(
        y,
        sr,
        lead_scan_ms=400.0,
        trail_scan_ms=120.0,
        max_transient_ms=45.0,
        peak_above_noise_db=14.0,
        removal="mute_then_trim",
    )
    assert er.removed_leading_ms > 0.0
    assert er.confidence_leading == "high"
    assert float(np.max(np.abs(er.y[:50]))) > 0.01


def test_edge_ui_click_trailing_short_burst() -> None:
    sr = 16000
    rng = np.random.default_rng(3)
    body_n = 6000
    body = (0.08 * np.sin(2 * np.pi * 220.0 * np.arange(body_n, dtype=np.float64) / sr)).astype(np.float32)
    body += 0.003 * rng.standard_normal(body_n).astype(np.float32)
    click = np.zeros(900, dtype=np.float32)
    click[820:880] = 0.95
    y = np.concatenate([body, click])
    er = apply_edge_ui_click(
        y,
        sr,
        lead_scan_ms=120.0,
        trail_scan_ms=400.0,
        max_transient_ms=50.0,
        peak_above_noise_db=14.0,
        removal="mute_then_trim",
    )
    assert er.removed_trailing_ms > 5.0
    assert er.confidence_trailing == "high"
    assert er.y.shape[0] < y.shape[0]


def test_edge_ui_click_trailing_requires_presilence_skips_when_absent() -> None:
    sr = 16000
    rng = np.random.default_rng(3)
    body_n = 6000
    body = (0.08 * np.sin(2 * np.pi * 220.0 * np.arange(body_n, dtype=np.float64) / sr)).astype(np.float32)
    body += 0.003 * rng.standard_normal(body_n).astype(np.float32)
    click = np.zeros(900, dtype=np.float32)
    click[820:880] = 0.95
    y = np.concatenate([body, click])
    er = apply_edge_ui_click(
        y,
        sr,
        lead_scan_ms=120.0,
        trail_scan_ms=400.0,
        max_transient_ms=50.0,
        peak_above_noise_db=14.0,
        removal="mute_then_trim",
        trail_click_requires_silence_ms=120.0,
    )
    assert er.removed_trailing_ms == 0.0
    assert er.confidence_trailing == "trail_presilence_short"
    assert er.y.shape[0] == y.shape[0]


def test_edge_ui_click_trailing_removes_click_after_silence_when_required() -> None:
    sr = 16000
    rng = np.random.default_rng(7)
    body_n = 5000
    body = (0.08 * np.sin(2 * np.pi * 220.0 * np.arange(body_n, dtype=np.float64) / sr)).astype(np.float32)
    body += 0.003 * rng.standard_normal(body_n).astype(np.float32)
    silence_n = int(0.2 * sr)
    silence = np.zeros(silence_n, dtype=np.float32)
    click = np.zeros(900, dtype=np.float32)
    click[820:880] = 0.95
    y = np.concatenate([body, silence, click])
    er = apply_edge_ui_click(
        y,
        sr,
        lead_scan_ms=120.0,
        trail_scan_ms=600.0,
        max_transient_ms=50.0,
        peak_above_noise_db=14.0,
        removal="mute_then_trim",
        trail_click_requires_silence_ms=80.0,
    )
    assert er.removed_trailing_ms > 5.0
    assert er.confidence_trailing == "high"
    assert er.y.shape[0] < y.shape[0]
