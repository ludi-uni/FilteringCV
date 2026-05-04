"""早期ゲート・二段 denoise・MFA 並列解決のユニットテスト。"""

from __future__ import annotations

import numpy as np

from cv_preprocess.audio.quality_gate import run_early_audio_gate
from cv_preprocess.config import (
    AudioPipelineConfig,
    DecodeStep,
    DenoiseStep,
    EarlyAudioGateConfig,
    MfaGateConfig,
    QualityGateConfig,
    ResampleStep,
    SnrEstimatorConfig,
)
from cv_preprocess.pipeline.preprocess_efficiency import (
    apply_denoise_steps_only,
    audio_pipeline_skip_denoise,
    exclusive_single_sgmse_denoise_for_two_pass_batch,
    resolve_mfa_parallelism,
)


def test_exclusive_single_sgmse_only() -> None:
    steps = [DenoiseStep(method="sgmse"), ResampleStep(sr=16000)]
    st = exclusive_single_sgmse_denoise_for_two_pass_batch(steps)
    assert st is not None
    assert st.method == "sgmse"


def test_exclusive_single_sgmse_rejects_mixed_denoise() -> None:
    steps = [DenoiseStep(method="sgmse"), DenoiseStep(method="spectral_subtract")]
    assert exclusive_single_sgmse_denoise_for_two_pass_batch(steps) is None


def test_resolve_mfa_parallelism_auto_jobs() -> None:
    mg = MfaGateConfig(auto_num_jobs=True, num_jobs=2, batch_size=4)
    nj, bs = resolve_mfa_parallelism(mg)
    assert nj >= 1
    assert bs == 4


def test_resolve_mfa_parallelism_scale_batch() -> None:
    mg = MfaGateConfig(
        num_jobs=2,
        batch_size=8,
        auto_num_jobs=False,
        auto_scale_batch_size=True,
        auto_batch_jobs_multiplier=10,
        batch_size_max=50,
    )
    nj, bs = resolve_mfa_parallelism(mg)
    assert nj == 2
    assert bs == 20


def test_audio_pipeline_skip_denoise() -> None:
    cfg = AudioPipelineConfig(
        target_sample_rate=16000,
        steps=[
            ResampleStep(sr=16000),
            DenoiseStep(method="dasheng"),
            DenoiseStep(method="spectral_subtract"),
        ],
    )
    lite = audio_pipeline_skip_denoise(cfg)
    assert len(lite.steps) == len(cfg.steps)
    assert isinstance(lite.steps[0], DecodeStep)
    assert isinstance(lite.steps[1], ResampleStep)
    assert isinstance(lite.steps[2], DenoiseStep)
    assert lite.steps[2].method == "none"
    assert isinstance(lite.steps[3], DenoiseStep)
    assert lite.steps[3].method == "none"


def test_apply_denoise_steps_only_skips_none() -> None:
    y = np.zeros(8000, dtype=np.float32)
    sr = 16000
    steps = [DenoiseStep(method="none"), ResampleStep(sr=16000)]
    y2, sr2, meta = apply_denoise_steps_only(y, sr, steps)
    assert y2.shape == y.shape
    assert sr2 == sr
    assert meta["denoise_trace"] == []


def test_run_early_audio_gate_duration() -> None:
    y = np.zeros(400, dtype=np.float32)
    early = EarlyAudioGateConfig(enabled=True, check_duration=True)
    g = QualityGateConfig(min_duration_sec=1.0, max_duration_sec=30.0)
    snr = SnrEstimatorConfig()
    r = run_early_audio_gate(
        y,
        16000,
        text_len=10,
        mora_count=None,
        main_gate=g,
        snr_cfg=snr,
        early=early,
    )
    assert not r.ok
    assert r.reason == "early_gate_duration"
