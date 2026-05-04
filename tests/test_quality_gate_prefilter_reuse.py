from pathlib import Path

import numpy as np

from cv_preprocess.audio.quality_gate import (
    quality_gate_configs_equivalent,
    quality_gate_run_fingerprint,
    run_quality_gate,
)
from cv_preprocess.config import InputConfig, PipelineConfig, QualityGateConfig, SnrEstimatorConfig
from cv_preprocess.pipeline.preprocess import _maybe_prefilter_final_gate_reuse_pair


def _minimal_cfg(**kwargs: object) -> PipelineConfig:
    return PipelineConfig(input=InputConfig(corpus_root=Path("/tmp")), **kwargs)


def _tone(sr: int, sec: float, amp: float = 0.08) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi * 220.0 * sec, int(sr * sec), endpoint=False, dtype=np.float64)
    return (amp * np.sin(t)).astype(np.float32)


def test_fingerprint_changes_with_audio_or_gate() -> None:
    sr = 16000
    y1 = _tone(sr, 0.5)
    y2 = _tone(sr, 0.5, amp=0.09)
    gate = QualityGateConfig()
    snr = SnrEstimatorConfig()
    fp1 = quality_gate_run_fingerprint(
        y1, sr, 10, gate=gate, snr_cfg=snr, mora_count=None
    )
    fp2 = quality_gate_run_fingerprint(
        y2, sr, 10, gate=gate, snr_cfg=snr, mora_count=None
    )
    assert fp1 != fp2
    g2 = gate.model_copy(update={"max_duration_sec": 99.0})
    fp3 = quality_gate_run_fingerprint(
        y1, sr, 10, gate=g2, snr_cfg=snr, mora_count=None
    )
    assert fp1 != fp3


def test_configs_equivalent_reflects_dump() -> None:
    a = QualityGateConfig()
    b = QualityGateConfig()
    assert quality_gate_configs_equivalent(a, b)
    c = a.model_copy(update={"max_silence_ratio": a.max_silence_ratio + 0.01})
    assert not quality_gate_configs_equivalent(a, c)


def test_maybe_reuse_pair_none_when_prefilter_gate_differs() -> None:
    y = _tone(22050, 0.8)
    sr = 22050
    base = QualityGateConfig(min_duration_sec=0.05, max_duration_sec=60.0)
    strict = base.model_copy(update={"max_silence_ratio": base.max_silence_ratio * 0.5})
    cfg = _minimal_cfg(quality_gate=base, snr=SnrEstimatorConfig())
    gate_pf = run_quality_gate(
        y, sr, text_len=5, gate=strict, snr_cfg=cfg.snr, mora_count=None
    )
    assert gate_pf.ok
    got = _maybe_prefilter_final_gate_reuse_pair(
        gate_pf,
        strict,
        cfg,
        y,
        sr,
        5,
        None,
        False,
        None,
    )
    assert got is None


def test_maybe_reuse_pair_none_when_mora_args_differ() -> None:
    y = _tone(22050, 1.2)
    sr = 22050
    gate = QualityGateConfig(
        min_duration_sec=0.1,
        max_duration_sec=60.0,
        min_sec_per_mora=0.05,
        mora_gate_relax=1.0,
    )
    cfg = _minimal_cfg(quality_gate=gate, snr=SnrEstimatorConfig())
    gate_pf = run_quality_gate(
        y, sr, text_len=10, gate=gate, snr_cfg=cfg.snr, mora_count=8
    )
    assert gate_pf.ok
    # prefilter にモーラ、本番にモーラ無し → 再利用不可
    got = _maybe_prefilter_final_gate_reuse_pair(
        gate_pf,
        gate,
        cfg,
        y,
        sr,
        10,
        8,
        False,
        8,
    )
    assert got is None


def test_maybe_reuse_pair_when_gates_and_mora_match() -> None:
    y = _tone(22050, 1.0)
    sr = 22050
    gate = QualityGateConfig(min_duration_sec=0.1, max_duration_sec=60.0)
    cfg = _minimal_cfg(quality_gate=gate, snr=SnrEstimatorConfig())
    gate_pf = run_quality_gate(
        y, sr, text_len=12, gate=gate, snr_cfg=cfg.snr, mora_count=None
    )
    assert gate_pf.ok
    got = _maybe_prefilter_final_gate_reuse_pair(
        gate_pf,
        gate,
        cfg,
        y,
        sr,
        12,
        None,
        False,
        None,
    )
    assert got is not None
    reused, fp = got
    assert reused is gate_pf
    assert fp == quality_gate_run_fingerprint(
        y, sr, 12, gate=gate, snr_cfg=cfg.snr, mora_count=None
    )
