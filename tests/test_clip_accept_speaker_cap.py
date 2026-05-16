from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cv_preprocess.audio.quality_gate import GateResult
from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import ClipRow
from cv_preprocess.pipeline.preprocess.clip_accept import (
    process_pending_to_acceptance,
    speaker_already_at_max_accepts,
)
from cv_preprocess.pipeline.preprocess.types import PendingClip


def _ok_gate() -> GateResult:
    return GateResult(
        ok=True,
        reason=None,
        duration_sec=1.0,
        silence_ratio=0.01,
        estimated_snr_db=20.0,
        clipping_ratio=0.0,
        dc_offset=0.0,
    )


def _pending(cid: str, path: str) -> PendingClip:
    row = ClipRow(
        client_id=cid,
        path=path,
        sentence="hello",
        raw={},
        locale="ja",
        sentence_id=None,
    )
    return PendingClip(
        row=row,
        y=np.zeros(8000, dtype=np.float32),
        sr=16000,
        text_raw="hello",
        text_norm="hello",
        phonemes=None,
        excerpt="hello",
        ameta={},
    )


def test_speaker_already_at_max_accepts(tmp_path: Path) -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": tmp_path},
            "speakers": {"max_clips_per_speaker": 1},
        }
    )
    row = ClipRow("a", "p.wav", "s", {}, None, None)
    assert not speaker_already_at_max_accepts(cfg, {"a": 0}, row)
    assert speaker_already_at_max_accepts(cfg, {"a": 1}, row)


def test_process_pending_to_acceptance_max_clips_per_speaker(tmp_path: Path) -> None:
    out_root = tmp_path / "out"
    rejects = tmp_path / "rejects.csv"
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": tmp_path},
            "output": {"root": out_root},
            "speakers": {"max_clips_per_speaker": 2},
        }
    )
    counts: dict[str, int] = {}
    accepted: list[dict] = []
    reject_reasons: dict[str, int] = {}
    fields = ["source_path", "client_id", "reason", "sentence_excerpt"]

    with patch(
        "cv_preprocess.pipeline.preprocess.clip_accept.run_quality_gate",
        return_value=_ok_gate(),
    ):
        idx = 0
        for i in range(2):
            idx = process_pending_to_acceptance(
                _pending("sp1", f"a{i}.mp3"),
                cfg=cfg,
                root=tmp_path,
                out_root=out_root,
                lang="ja",
                rejects_path=rejects,
                reject_fields=fields,
                reject_reasons=reject_reasons,
                accepted=accepted,
                accept_idx=idx,
                accepted_count_by_speaker=counts,
            )
        idx = process_pending_to_acceptance(
            _pending("sp1", "over.mp3"),
            cfg=cfg,
            root=tmp_path,
            out_root=out_root,
            lang="ja",
            rejects_path=rejects,
            reject_fields=fields,
            reject_reasons=reject_reasons,
            accepted=accepted,
            accept_idx=idx,
            accepted_count_by_speaker=counts,
        )

    assert len(accepted) == 2
    assert counts == {"sp1": 2}
    assert reject_reasons.get("max_clips_per_speaker") == 1
    assert idx == 2


def test_process_pending_requires_counter_dict_when_cap_set(tmp_path: Path) -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": tmp_path},
            "output": {"root": tmp_path / "out"},
            "speakers": {"max_clips_per_speaker": 1},
        }
    )
    with (
        patch(
            "cv_preprocess.pipeline.preprocess.clip_accept.run_quality_gate",
            return_value=_ok_gate(),
        ),
        pytest.raises(ValueError, match="accepted_count_by_speaker"),
    ):
        process_pending_to_acceptance(
            _pending("sp1", "a.mp3"),
            cfg=cfg,
            root=tmp_path,
            out_root=tmp_path / "out",
            lang="ja",
            rejects_path=tmp_path / "r.csv",
            reject_fields=["source_path", "client_id", "reason", "sentence_excerpt"],
            reject_reasons={},
            accepted=[],
            accept_idx=0,
            accepted_count_by_speaker=None,
        )
