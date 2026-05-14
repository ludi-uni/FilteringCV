"""ASR ゲートの足切り（``PendingClip`` 単位）。``audio.asr_batch`` との循環回避のため分離。"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from cv_preprocess.audio.asr_batch import (
    mock_asr_hypothesis,
    transcribe_batch_paths,
    write_temp_wavs_for_asr_batch,
)
from cv_preprocess.config.pipeline import PipelineConfig
from cv_preprocess.pipeline.export import write_reject_row
from cv_preprocess.pipeline.preprocess.types import PendingClip
from cv_preprocess.text.normalize import normalize_for_tts
from cv_preprocess.text.phoneme_compare import char_error_rate, token_error_rate
from cv_preprocess.text.phonemize import g2p_phonemes

_asr_batch_error_logged: list[bool] = [False]


def _norm_text(s: str, *, enabled: bool) -> str:
    if not enabled:
        return s
    return normalize_for_tts(s)


def apply_asr_gate(
    cfg: PipelineConfig,
    pending: list[PendingClip],
    *,
    work_parent: Path,
    rejects_path: Path,
    reject_fields: list[str],
    reject_reasons: dict[str, int],
) -> list[PendingClip]:
    """``pending`` の波形に対し ASR 仮説を得て閾値判定。通過クリップのみ返し、拒否は ``write_reject_row`` する。"""
    ag = cfg.asr_gate
    if not ag.enabled or not pending:
        return list(pending)

    texts: list[str] = []
    if ag.backend == "mock":
        texts = [mock_asr_hypothesis(p.text_norm, ag) for p in pending]
    elif ag.backend == "nemo_transcribe":
        uid = uuid.uuid4().hex[:12]
        work_dir = work_parent / f"asr_batch_{uid}"
        try:
            clips = [(p.y, int(p.sr)) for p in pending]
            paths = write_temp_wavs_for_asr_batch(clips, ag, work_dir)
            texts = transcribe_batch_paths(ag, paths)
        except Exception as e:
            if not _asr_batch_error_logged[0]:
                _asr_batch_error_logged[0] = True
                err = f"{type(e).__name__}: {e}".replace("\n", " ").strip()
                if len(err) > 400:
                    err = err[:397] + "..."
                print(
                    "[cv-preprocess] asr_gate NeMo バッチ失敗（以降は各クリップ asr_worker_error のみ）: "
                    + err,
                    file=sys.stderr,
                    flush=True,
                )
            for p in pending:
                write_reject_row(
                    rejects_path,
                    {
                        "source_path": p.row.path,
                        "client_id": p.row.client_id,
                        "reason": "asr_worker_error",
                        "sentence_excerpt": p.excerpt,
                    },
                    reject_fields,
                )
                reject_reasons["asr_worker_error"] = reject_reasons.get("asr_worker_error", 0) + 1
            return []
        finally:
            try:
                if work_dir.is_dir():
                    for child in work_dir.glob("*"):
                        try:
                            child.unlink()
                        except OSError:
                            pass
                    work_dir.rmdir()
            except OSError:
                pass
    else:
        raise ValueError(f"unknown asr_gate.backend: {ag.backend!r}")

    if len(texts) != len(pending):
        for p in pending:
            write_reject_row(
                rejects_path,
                {
                    "source_path": p.row.path,
                    "client_id": p.row.client_id,
                    "reason": "asr_worker_error",
                    "sentence_excerpt": p.excerpt,
                },
                reject_fields,
            )
            reject_reasons["asr_worker_error"] = reject_reasons.get("asr_worker_error", 0) + 1
        return []

    survivors: list[PendingClip] = []
    for p, hyp_raw in zip(pending, texts):
        ref_raw = p.text_norm
        if not (ref_raw or "").strip():
            if ag.missing_transcript == "reject":
                write_reject_row(
                    rejects_path,
                    {
                        "source_path": p.row.path,
                        "client_id": p.row.client_id,
                        "reason": "asr_missing_transcript",
                        "sentence_excerpt": p.excerpt,
                    },
                    reject_fields,
                )
                reject_reasons["asr_missing_transcript"] = (
                    reject_reasons.get("asr_missing_transcript", 0) + 1
                )
            else:
                survivors.append(p)
            continue

        hyp_stripped = (hyp_raw or "").strip()
        if not hyp_stripped:
            if ag.decode_failure == "reject":
                write_reject_row(
                    rejects_path,
                    {
                        "source_path": p.row.path,
                        "client_id": p.row.client_id,
                        "reason": "asr_decode_failed",
                        "sentence_excerpt": p.excerpt,
                    },
                    reject_fields,
                )
                reject_reasons["asr_decode_failed"] = reject_reasons.get("asr_decode_failed", 0) + 1
            else:
                survivors.append(p)
            continue

        ref_n = _norm_text(ref_raw, enabled=ag.normalize_reference_text)
        hyp_n = _norm_text(hyp_stripped, enabled=ag.normalize_hypothesis_text)

        cer = char_error_rate(ref_n, hyp_n)
        per: float | None = None
        if ag.compare_phonemes:
            try:
                ref_ph = g2p_phonemes(ref_n, kana=cfg.text.g2p_kana)
                hyp_ph = g2p_phonemes(hyp_n, kana=cfg.text.g2p_kana)
                per = token_error_rate(ref_ph, hyp_ph)
            except Exception:
                if ag.decode_failure == "reject":
                    write_reject_row(
                        rejects_path,
                        {
                            "source_path": p.row.path,
                            "client_id": p.row.client_id,
                            "reason": "asr_decode_failed",
                            "sentence_excerpt": p.excerpt,
                        },
                        reject_fields,
                    )
                    reject_reasons["asr_decode_failed"] = reject_reasons.get("asr_decode_failed", 0) + 1
                else:
                    survivors.append(p)
                continue

        p.asr_hypothesis = hyp_n
        p.asr_confidence = 1.0 if ag.backend == "mock" else None
        p.asr_char_error_rate = cer
        p.asr_phoneme_error_rate = per

        ratio_cap = ag.max_hypothesis_len_ratio
        if ratio_cap is not None and len(ref_n) > 0 and len(hyp_n) > ratio_cap * max(len(ref_n), 1):
            write_reject_row(
                rejects_path,
                {
                    "source_path": p.row.path,
                    "client_id": p.row.client_id,
                    "reason": "asr_duration_outlier",
                    "sentence_excerpt": p.excerpt,
                },
                reject_fields,
            )
            reject_reasons["asr_duration_outlier"] = reject_reasons.get("asr_duration_outlier", 0) + 1
            continue

        min_hc = ag.min_hypothesis_chars
        if min_hc is not None and len(ref_n) > 0 and len(hyp_n) < int(min_hc):
            write_reject_row(
                rejects_path,
                {
                    "source_path": p.row.path,
                    "client_id": p.row.client_id,
                    "reason": "asr_duration_outlier",
                    "sentence_excerpt": p.excerpt,
                },
                reject_fields,
            )
            reject_reasons["asr_duration_outlier"] = reject_reasons.get("asr_duration_outlier", 0) + 1
            continue

        if ag.min_asr_confidence is not None and p.asr_confidence is not None:
            if float(p.asr_confidence) < float(ag.min_asr_confidence):
                write_reject_row(
                    rejects_path,
                    {
                        "source_path": p.row.path,
                        "client_id": p.row.client_id,
                        "reason": "asr_low_confidence",
                        "sentence_excerpt": p.excerpt,
                    },
                    reject_fields,
                )
                reject_reasons["asr_low_confidence"] = reject_reasons.get("asr_low_confidence", 0) + 1
                continue

        if ag.compare_text and cer > float(ag.max_char_error_rate):
            write_reject_row(
                rejects_path,
                {
                    "source_path": p.row.path,
                    "client_id": p.row.client_id,
                    "reason": "asr_text_mismatch",
                    "sentence_excerpt": p.excerpt,
                },
                reject_fields,
            )
            reject_reasons["asr_text_mismatch"] = reject_reasons.get("asr_text_mismatch", 0) + 1
            continue

        if ag.compare_phonemes and per is not None and per > float(ag.max_phoneme_error_rate):
            write_reject_row(
                rejects_path,
                {
                    "source_path": p.row.path,
                    "client_id": p.row.client_id,
                    "reason": "asr_phoneme_mismatch",
                    "sentence_excerpt": p.excerpt,
                },
                reject_fields,
            )
            reject_reasons["asr_phoneme_mismatch"] = reject_reasons.get("asr_phoneme_mismatch", 0) + 1
            continue

        survivors.append(p)

    return survivors
