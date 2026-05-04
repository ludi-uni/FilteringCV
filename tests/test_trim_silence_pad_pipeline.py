"""trim_silence ステップの pad_head_ms / pad_tail_ms がパイプラインで反映されること。"""
import numpy as np

from cv_preprocess.audio.pipeline import run_steps_on_array
from cv_preprocess.config import AudioPipelineConfig, TrimSilenceStep


def test_trim_silence_pad_does_not_broadcast_on_batch_dim() -> None:
    """(1, T) に対して pad が各軸へ適用され巨大化しないこと（パイプラインと同じ np.pad 契約）。"""
    from cv_preprocess.audio.pipeline import _mono_1d_float32

    sr = 22050
    t = 8000
    y_wrong = np.ones((1, t), dtype=np.float32) * 0.1
    y1d = _mono_1d_float32(y_wrong)
    ph, pt = int(round(75 * sr / 1000)), int(round(120 * sr / 1000))
    out = np.pad(y1d, (ph, pt), mode="constant", constant_values=0.0)
    assert out.ndim == 1
    assert out.shape[0] == t + ph + pt
    # 誤って (ph,pt) を 2 次元に適用すると約 (1+ph+pt)*(t+ph+pt) サンプルになる
    assert out.shape[0] < t * 2


def test_trim_silence_pad_extends_length() -> None:
    sr = 1000
    # 0.5s の矩形波（trim でほぼ全域が非無音のまま）
    y = np.ones(sr // 2, dtype=np.float32) * 0.1
    cfg = AudioPipelineConfig(
        target_sample_rate=sr,
        channels=1,
        audio_pipeline_id="test",
        steps=[
            TrimSilenceStep(
                max_keep_sec=60.0,
                head_tail_db=60.0,
                trim_frame_length=256,
                trim_hop_length=64,
                pad_head_ms=50.0,
                pad_tail_ms=30.0,
            ),
        ],
    )
    out, out_sr, meta = run_steps_on_array(y, sr, cfg)
    assert out_sr == sr
    want = y.size + int(round(50 * sr / 1000.0)) + int(round(30 * sr / 1000.0))
    assert out.shape[0] == want
    assert np.allclose(out[:50], 0.0)
    assert np.allclose(out[-30:], 0.0)
    trace = meta["steps_trace"][-1]
    assert trace["pad_head_ms"] == 50.0
    assert trace["pad_tail_ms"] == 30.0
