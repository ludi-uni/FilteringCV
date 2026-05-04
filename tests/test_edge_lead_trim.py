import numpy as np

from cv_preprocess.audio.edge_ui_click import apply_edge_ui_click


def test_lead_click_then_silence_then_tone_preserves_weak_onset() -> None:
    """クリック→無音→弱い立ち上がりのトーン: 全体 eps トリムに比べ先頭トーンを削りにくい。"""
    sr = 16000
    click = np.zeros(1200, dtype=np.float32)
    click[80:140] = 1.0
    silence = np.zeros(2400, dtype=np.float32)
    n_tone = 8000
    t = np.arange(n_tone, dtype=np.float64) / sr
    # 先頭 600 サンプルはごく小さい振幅（子音立ち上がり相当）
    ramp_n = 600
    tone = np.zeros(n_tone, dtype=np.float32)
    tone[ramp_n:] = (0.07 * np.sin(2 * np.pi * 300.0 * t[ramp_n:])).astype(np.float32)
    tone[:ramp_n] = (0.0025 * np.sin(2 * np.pi * 300.0 * t[:ramp_n])).astype(np.float32)
    y = np.concatenate([click, silence, tone])

    er = apply_edge_ui_click(
        y,
        sr,
        lead_scan_ms=500.0,
        trail_scan_ms=120.0,
        max_transient_ms=45.0,
        peak_above_noise_db=14.0,
        removal="mute_then_trim",
        lead_post_click_top_db=32.0,
    )
    assert er.removed_leading_ms > 0.0
    assert er.confidence_leading == "high"
    # 弱い立ち上がりが残っている（先頭からすぐにごく小さな正弦が存在）
    assert er.y.size > 4000
    assert float(np.max(np.abs(er.y[:400]))) > 5e-5
