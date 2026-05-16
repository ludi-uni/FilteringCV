"""asr_gate 判定ロジック（mock）。"""

from pathlib import Path

import numpy as np
import yaml

from cv_preprocess.pipeline.preprocess.asr_gate_apply import apply_asr_gate
from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import ClipRow
from cv_preprocess.pipeline.preprocess.types import PendingClip


def _cfg(tmp_path: Path, **asr_over: object) -> PipelineConfig:
    (tmp_path / "ja").mkdir()
    p = tmp_path / "c.yaml"
    base = {
        "input": {"corpus_root": str(tmp_path / "ja")},
        "text": {"phonemize": True},
        "asr_gate": {
            "enabled": True,
            "backend": "mock",
            "mock_mode": "echo",
            "compare_text": True,
            "compare_phonemes": True,
            "max_char_error_rate": 0.2,
            "max_phoneme_error_rate": 0.25,
        },
    }
    ag = dict(base["asr_gate"])
    ag.update(asr_over)
    base["asr_gate"] = ag
    p.write_text(yaml.dump(base), encoding="utf-8")
    return PipelineConfig.from_yaml(p)


def _pending(text: str) -> PendingClip:
    row = ClipRow(
        client_id="c1",
        path="clips/x.wav",
        sentence=text,
        raw={},
        locale="ja",
    )
    y = np.zeros(8000, dtype=np.float32)
    return PendingClip(
        row=row,
        y=y,
        sr=16000,
        text_raw=text,
        text_norm=text,
        phonemes="a",
        excerpt=text[:40],
        ameta={},
    )


def test_apply_asr_gate_mock_echo_accepts(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    p = _pending("こんにちは")
    out = apply_asr_gate(
        cfg,
        [p],
        work_parent=tmp_path,
        rejects_path=tmp_path / "rej.csv",
        reject_fields=["source_path", "client_id", "reason", "sentence_excerpt"],
        reject_reasons={},
    )
    assert len(out) == 1
    assert out[0].asr_hypothesis is not None


def test_apply_asr_gate_updates_phonemes_from_hypothesis(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        use_hypothesis_phonemes=True,
        sync_text_norm_to_hypothesis=True,
        mock_mode="echo",
    )
    p = _pending("あ")
    p.phonemes = "stale"
    out = apply_asr_gate(
        cfg,
        [p],
        work_parent=tmp_path,
        rejects_path=tmp_path / "rej.csv",
        reject_fields=["source_path", "client_id", "reason", "sentence_excerpt"],
        reject_reasons={},
    )
    assert len(out) == 1
    assert out[0].phonemes == "a"
    assert out[0].text_norm == "あ"


def test_apply_asr_gate_mock_mismatch_rejects_text(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, mock_mode="mismatch_char")
    p = _pending("あ")
    rejects: dict[str, int] = {}
    out = apply_asr_gate(
        cfg,
        [p],
        work_parent=tmp_path,
        rejects_path=tmp_path / "rej.csv",
        reject_fields=["source_path", "client_id", "reason", "sentence_excerpt"],
        reject_reasons=rejects,
    )
    assert len(out) == 0
    assert rejects.get("asr_text_mismatch", 0) >= 1
