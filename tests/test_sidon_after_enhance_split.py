from __future__ import annotations

from unittest.mock import patch

import numpy as np

from cv_preprocess.audio.quality_gate import sidon_rescue_after_enhance_split_waveform
from cv_preprocess.config import (
    QualityGateConfig,
    QualityGateSidonAfterEnhanceSplitConfig,
    SnrEstimatorConfig,
)


def test_sidon_after_enhance_split_disabled_passthrough() -> None:
    sr = 24_000
    n = int(sr * 0.6)
    t = np.arange(n, dtype=np.float64) / sr
    y = (0.08 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    gate = QualityGateConfig(
        sidon_after_enhance_split=QualityGateSidonAfterEnhanceSplitConfig(enabled=False),
        min_sec_per_mora=None,
    )
    snr_cfg = SnrEstimatorConfig()
    y2, sr2, meta = sidon_rescue_after_enhance_split_waveform(
        y, sr, text_len=20, mora_count=None, gate=gate, snr_cfg=snr_cfg
    )
    assert sr2 == sr
    assert y2.shape == y.shape
    assert meta["sidon_after_enhance_split"]["enabled"] is False


def test_sidon_after_enhance_split_calls_restore_when_pre_gate_fails() -> None:
    sr = 24_000
    n = int(sr * 0.6)
    y = (0.05 * np.sin(2 * np.pi * 330.0 * np.arange(n, dtype=np.float64) / sr)).astype(np.float32)
    gate = QualityGateConfig(
        sidon_after_enhance_split=QualityGateSidonAfterEnhanceSplitConfig(enabled=True),
        min_duration_sec=99.0,
        max_duration_sec=120.0,
        min_sec_per_mora=None,
    )
    snr_cfg = SnrEstimatorConfig()

    def _identity_restore(
        arr: np.ndarray,
        sample_rate: int,
        _cfg: QualityGateSidonAfterEnhanceSplitConfig,
    ) -> np.ndarray:
        return arr.copy()

    with patch("cv_preprocess.audio.sidon_restore.apply_sidon_restore", side_effect=_identity_restore):
        y2, sr2, meta = sidon_rescue_after_enhance_split_waveform(
            y, sr, text_len=20, mora_count=None, gate=gate, snr_cfg=snr_cfg
        )
    assert sr2 == sr
    block = meta["sidon_after_enhance_split"]
    assert block["enabled"] is True
    assert block["pre_gate_ok"] is False
    assert block["pre_gate_reason"] == "gate_duration"
    assert block.get("applied") is True
    assert block.get("post_gate_ok") is False
