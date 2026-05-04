import numpy as np

from cv_preprocess.audio.pipeline import (
    apply_edge_ui_click_step,
    run_spectral_processing_then_edge_ui_click,
    run_steps_on_array,
)
from cv_preprocess.config import (
    AudioPipelineConfig,
    EdgeUiClickStep,
    LipNoiseRepairStep,
    TrimSilenceStep,
)


def test_apply_edge_ui_click_step_matches_run_steps_edge_only() -> None:
    sr = 16000
    rng = np.random.default_rng(3)
    body_n = 6000
    body = (0.08 * np.sin(2 * np.pi * 220.0 * np.arange(body_n, dtype=np.float64) / sr)).astype(np.float32)
    body += 0.003 * rng.standard_normal(body_n).astype(np.float32)
    click = np.zeros(900, dtype=np.float32)
    click[820:880] = 0.95
    y = np.concatenate([body, click])

    edge = EdgeUiClickStep(
        lead_scan_ms=120.0,
        trail_scan_ms=400.0,
        max_transient_ms=50.0,
        peak_above_noise_db=14.0,
        removal="mute_then_trim",
    )
    cfg = AudioPipelineConfig(target_sample_rate=sr, channels=1, steps=[edge])
    y1, sr1, m1 = run_steps_on_array(y.copy(), sr, cfg)
    y2, m2 = apply_edge_ui_click_step(y, sr, edge)
    assert sr1 == sr
    assert np.allclose(y1, y2, atol=1e-5, rtol=1e-6)
    assert abs(m1["edge_removed_trailing_ms"] - m2["edge_removed_trailing_ms"]) < 1e-3
    assert m1["edge_click_confidence"] == m2["edge_click_confidence"]


def test_run_spectral_processing_then_edge_omits_prior_edge_step() -> None:
    """spectral_cfg 内の edge_ui_click はスキップされ、最後の edge_step だけが効く。"""
    sr = 16000
    rng = np.random.default_rng(3)
    body_n = 6000
    body = (0.08 * np.sin(2 * np.pi * 220.0 * np.arange(body_n, dtype=np.float64) / sr)).astype(np.float32)
    body += 0.003 * rng.standard_normal(body_n).astype(np.float32)
    click = np.zeros(900, dtype=np.float32)
    click[820:880] = 0.95
    y = np.concatenate([body, click])

    embedded_edge = EdgeUiClickStep(
        lead_scan_ms=120.0,
        trail_scan_ms=400.0,
        max_transient_ms=50.0,
        peak_above_noise_db=14.0,
        removal="mute_then_trim",
        trail_click_requires_silence_ms=120.0,
    )
    post_edge = EdgeUiClickStep(
        lead_scan_ms=120.0,
        trail_scan_ms=400.0,
        max_transient_ms=50.0,
        peak_above_noise_db=14.0,
        removal="mute_then_trim",
        trail_click_requires_silence_ms=None,
    )
    spectral = AudioPipelineConfig(
        target_sample_rate=sr,
        channels=1,
        steps=[
            TrimSilenceStep(
                max_keep_sec=30.0,
                head_tail_db=60.0,
                trim_frame_length=1024,
                trim_hop_length=256,
            ),
            embedded_edge,
        ],
    )
    y_out, sr_out, meta = run_spectral_processing_then_edge_ui_click(
        y.copy(), sr, spectral, post_edge, omit_edge_ui_clicks_from_spectral=True
    )
    assert sr_out == sr
    assert meta["post_spectral_edge_removed_trailing_ms"] > 5.0
    assert y_out.shape[0] < y.shape[0]
    assert meta["steps_trace"][-1]["type"] == "edge_ui_click"
    assert meta["steps_trace"][-1].get("phase") == "post_spectral"


def test_post_spectral_edge_deferred_until_after_spectral_block() -> None:
    """post_spectral: true は lip 等のあと、次の非スペクトル（trim）の直前で実行される。"""
    sr = 16000
    rng = np.random.default_rng(3)
    body_n = 6000
    body = (0.08 * np.sin(2 * np.pi * 220.0 * np.arange(body_n, dtype=np.float64) / sr)).astype(np.float32)
    body += 0.003 * rng.standard_normal(body_n).astype(np.float32)
    click = np.zeros(900, dtype=np.float32)
    click[820:880] = 0.95
    y = np.concatenate([body, click])

    edge = EdgeUiClickStep(
        post_spectral=True,
        lead_scan_ms=120.0,
        trail_scan_ms=400.0,
        max_transient_ms=50.0,
        peak_above_noise_db=14.0,
        removal="mute_then_trim",
    )
    cfg = AudioPipelineConfig(
        target_sample_rate=sr,
        channels=1,
        steps=[
            LipNoiseRepairStep(),
            edge,
            TrimSilenceStep(
                max_keep_sec=30.0,
                head_tail_db=55.0,
                trim_frame_length=1024,
                trim_hop_length=256,
            ),
        ],
    )
    y2, out_sr, meta = run_steps_on_array(y, sr, cfg)
    assert out_sr == sr
    assert meta["edge_removed_trailing_ms"] > 5.0
    types = [t["type"] for t in meta["steps_trace"]]
    assert types.index("lip_noise_repair") < types.index("edge_ui_click")
    assert types.index("edge_ui_click") < types.index("trim_silence")
    edge_traces = [t for t in meta["steps_trace"] if t["type"] == "edge_ui_click"]
    assert len(edge_traces) == 1
    assert edge_traces[0]["phase"] == "post_spectral"
    assert y2.shape[0] < y.shape[0]


def test_post_spectral_inline_when_no_spectral_step_yet() -> None:
    """スペクトル系ステップがまだ無いとき post_spectral はキューせず即時適用。"""
    sr = 16000
    rng = np.random.default_rng(3)
    body_n = 6000
    body = (0.08 * np.sin(2 * np.pi * 220.0 * np.arange(body_n, dtype=np.float64) / sr)).astype(np.float32)
    body += 0.003 * rng.standard_normal(body_n).astype(np.float32)
    click = np.zeros(900, dtype=np.float32)
    click[820:880] = 0.95
    y = np.concatenate([body, click])

    edge = EdgeUiClickStep(
        post_spectral=True,
        lead_scan_ms=120.0,
        trail_scan_ms=400.0,
        max_transient_ms=50.0,
        peak_above_noise_db=14.0,
        removal="mute_then_trim",
    )
    cfg = AudioPipelineConfig(
        target_sample_rate=sr,
        channels=1,
        steps=[
            TrimSilenceStep(
                max_keep_sec=30.0,
                head_tail_db=60.0,
                trim_frame_length=1024,
                trim_hop_length=256,
            ),
            edge,
        ],
    )
    y2, out_sr, meta = run_steps_on_array(y, sr, cfg)
    assert out_sr == sr
    assert meta["edge_removed_trailing_ms"] > 5.0
    edge_traces = [t for t in meta["steps_trace"] if t["type"] == "edge_ui_click"]
    assert edge_traces[0]["phase"] == "post_spectral_inline"
    assert y2.shape[0] < y.shape[0]
