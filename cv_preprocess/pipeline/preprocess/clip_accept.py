from __future__ import annotations

from pathlib import Path

from cv_preprocess.audio.quality_gate import quality_gate_run_fingerprint, run_quality_gate
from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import ClipRow
from cv_preprocess.pipeline.export import write_reject_row, write_wav_16bit
from cv_preprocess.pipeline.preprocess.helpers import effective_final_quality_gate, infer_release
from cv_preprocess.pipeline.preprocess.types import PendingClip
from cv_preprocess.pipeline.preprocess_efficiency import effective_audio_catalog_for_preprocess


def speaker_already_at_max_accepts(
    cfg: PipelineConfig,
    accepted_count_by_speaker: dict[str, int],
    row: ClipRow,
) -> bool:
    """その ``row`` の話者が、設定上限分すでに採用済みなら真（採用前の短絡判定用）。"""
    cap = cfg.speakers.max_clips_per_speaker
    if cap is None:
        return False
    cid = row.client_id.strip()
    return accepted_count_by_speaker.get(cid, 0) >= int(cap)


def process_pending_to_acceptance(
    pending: PendingClip,
    *,
    cfg: PipelineConfig,
    root: Path,
    out_root: Path,
    lang: str,
    rejects_path: Path,
    reject_fields: list[str],
    reject_reasons: dict[str, int],
    accepted: list[dict],
    accept_idx: int,
    accepted_count_by_speaker: dict[str, int] | None = None,
) -> int:
    """品質ゲート通過時のみ ``accept_idx`` を増やす。拒否時は元の値を返す。

    ``speakers.max_clips_per_speaker`` 設定時は、通過後にその話者の **既採用数** が上限に達していれば
    ``rejects`` に ``max_clips_per_speaker`` で記録して拒否する（``accepted_count_by_speaker`` が必須）。
    """
    row = pending.row
    y, sr = pending.y, pending.sr
    text_raw, text_norm = pending.text_raw, pending.text_norm
    phonemes = pending.phonemes
    excerpt = pending.excerpt
    ameta = pending.ameta

    mora_n: int | None = pending.mora_count
    if mora_n is None and cfg.quality_gate.min_sec_per_mora is not None and lang.split("-")[0] == "ja":
        try:
            from cv_preprocess.text.mora_estimate import mora_count_for_text

            mora_n = mora_count_for_text(text_norm)
        except Exception:
            write_reject_row(
                rejects_path,
                {
                    "source_path": row.path,
                    "client_id": row.client_id,
                    "reason": "mora_estimate_failed",
                    "sentence_excerpt": excerpt,
                },
                reject_fields,
            )
            reject_reasons["mora_estimate_failed"] = reject_reasons.get("mora_estimate_failed", 0) + 1
            return accept_idx

    final_qg = effective_final_quality_gate(cfg)
    fp_final = quality_gate_run_fingerprint(
        y,
        sr,
        len(text_norm),
        gate=final_qg,
        snr_cfg=cfg.snr,
        mora_count=mora_n,
    )
    if (
        not cfg.two_pass_denoise.enabled
        and pending.prefilter_final_gate_reuse is not None
        and pending.prefilter_final_gate_fp == fp_final
    ):
        gate = pending.prefilter_final_gate_reuse
    else:
        gate = run_quality_gate(
            y,
            sr,
            text_len=len(text_norm),
            gate=final_qg,
            snr_cfg=cfg.snr,
            mora_count=mora_n,
        )
    if not gate.ok:
        write_reject_row(
            rejects_path,
            {
                "source_path": row.path,
                "client_id": row.client_id,
                "reason": gate.reason or "gate",
                "sentence_excerpt": excerpt,
            },
            reject_fields,
        )
        reject_reasons[gate.reason or "gate"] = reject_reasons.get(gate.reason or "gate", 0) + 1
        return accept_idx

    sp_cap = cfg.speakers.max_clips_per_speaker
    cid = row.client_id.strip()
    if sp_cap is not None:
        if accepted_count_by_speaker is None:
            raise ValueError(
                "process_pending_to_acceptance(..., accepted_count_by_speaker=...) が必要です。"
                "speakers.max_clips_per_speaker を設定したときは話者ごとの採用数を渡してください。"
            )
        if speaker_already_at_max_accepts(cfg, accepted_count_by_speaker, row):
            write_reject_row(
                rejects_path,
                {
                    "source_path": row.path,
                    "client_id": row.client_id,
                    "reason": "max_clips_per_speaker",
                    "sentence_excerpt": excerpt,
                },
                reject_fields,
            )
            reject_reasons["max_clips_per_speaker"] = reject_reasons.get("max_clips_per_speaker", 0) + 1
            return accept_idx

    accept_idx += 1
    utt_id = f"cv_{lang}_{accept_idx:06d}"
    rel_wav = f"{cfg.output.wav_subdir}/{utt_id}.wav"
    wav_path = out_root / rel_wav
    bit_depth = 16
    for st in effective_audio_catalog_for_preprocess(cfg).steps:
        if st.type == "save_wav":
            bit_depth = st.bit_depth
    if bit_depth == 16:
        write_wav_16bit(wav_path, y, sr)
    else:
        write_wav_16bit(wav_path, y, sr)

    rec = {
        "utt_id": utt_id,
        "audio_path": rel_wav.replace("\\", "/"),
        "text_raw": text_raw,
        "text_norm": text_norm,
        "phonemes": phonemes,
        "speaker_id": row.client_id,
        "duration_sec": gate.duration_sec,
        "silence_ratio": gate.silence_ratio,
        "estimated_snr_db": gate.estimated_snr_db,
        "quality_score": gate.quality_score,
        "quality_tier": gate.quality_tier,
        "trailing_silence_sec": gate.trailing_silence_sec,
        "split": None,
        "source_release": infer_release(root),
        "source_path": row.path,
        "audio_pipeline_id": effective_audio_catalog_for_preprocess(cfg).audio_pipeline_id,
        "edge_removed_leading_ms": ameta.get("edge_removed_leading_ms", 0.0),
        "edge_removed_trailing_ms": ameta.get("edge_removed_trailing_ms", 0.0),
        "edge_click_confidence": ameta.get("edge_click_confidence"),
        "mora_count": gate.mora_count,
        "min_required_duration_sec": gate.min_required_duration_sec,
        "asr_hypothesis": pending.asr_hypothesis,
        "asr_confidence": pending.asr_confidence,
        "asr_char_error_rate": pending.asr_char_error_rate,
        "asr_phoneme_error_rate": pending.asr_phoneme_error_rate,
    }
    accepted.append(rec)
    if sp_cap is not None and accepted_count_by_speaker is not None:
        accepted_count_by_speaker[cid] = accepted_count_by_speaker.get(cid, 0) + 1
    return accept_idx
