from __future__ import annotations

import sys
import traceback

import numpy as np

from cv_preprocess.audio.decode import load_audio
from cv_preprocess.audio.pipeline import run_steps_on_array
from cv_preprocess.audio.quality_gate import run_early_audio_gate, run_quality_gate
from cv_preprocess.audio.resample import resample_audio
from cv_preprocess.io.tsv_loader import ClipRow, iter_clip_audio_paths
from cv_preprocess.pipeline.export import write_reject_row
from cv_preprocess.pipeline.preprocess.helpers import (
    _compute_clip_mora_count_once,
    _maybe_prefilter_final_gate_reuse_pair,
    _mora_gates_needed,
)
from cv_preprocess.pipeline.preprocess.types import PendingClip
from cv_preprocess.text.language_detect import passes_japanese_policy, passes_locale_expected
from cv_preprocess.text.normalize import normalize_for_tts
from cv_preprocess.text.phoneme_compare import phoneme_sequences_accept
from cv_preprocess.text.phonemize import g2p_phonemes


class PreprocessRowMixin:
    """1 クリップ分のテキスト検証〜パス1〜prefilter〜アライナ投入まで（``PreprocessSession`` 用ミックスイン）。"""

    def _preprocess_one_row(self, row: ClipRow) -> None:
        excerpt = row.sentence[:80] if row.sentence else ""
        text_raw = row.sentence
        text_norm = normalize_for_tts(text_raw)

        if not passes_locale_expected(row.locale, self.cfg.input.locale_expected):
            write_reject_row(
                self.rejects_path,
                {
                    "source_path": row.path,
                    "client_id": row.client_id,
                    "reason": "text_locale",
                    "sentence_excerpt": excerpt,
                },
                self.reject_fields,
            )
            self.reject_reasons["text_locale"] = self.reject_reasons.get("text_locale", 0) + 1
            return

        if not passes_japanese_policy(text_norm, self.cfg.text.require_japanese):
            write_reject_row(
                self.rejects_path,
                {
                    "source_path": row.path,
                    "client_id": row.client_id,
                    "reason": "text_not_japanese",
                    "sentence_excerpt": excerpt,
                },
                self.reject_fields,
            )
            self.reject_reasons["text_not_japanese"] = self.reject_reasons.get("text_not_japanese", 0) + 1
            return

        tl = len(text_norm)
        if tl < self.cfg.text.min_text_len or tl > self.cfg.text.max_text_len:
            write_reject_row(
                self.rejects_path,
                {
                    "source_path": row.path,
                    "client_id": row.client_id,
                    "reason": "text_length",
                    "sentence_excerpt": excerpt,
                },
                self.reject_fields,
            )
            self.reject_reasons["text_length"] = self.reject_reasons.get("text_length", 0) + 1
            return

        phonemes: str | None = None
        if self.cfg.text.phonemize:
            try:
                phonemes = g2p_phonemes(text_norm, kana=self.cfg.text.g2p_kana)
            except Exception:
                write_reject_row(
                    self.rejects_path,
                    {
                        "source_path": row.path,
                        "client_id": row.client_id,
                        "reason": "phonemize_failed",
                        "sentence_excerpt": excerpt,
                    },
                    self.reject_fields,
                )
                self.reject_reasons["phonemize_failed"] = self.reject_reasons.get("phonemize_failed", 0) + 1
                return

        if self.alignment_by_path is not None and phonemes is not None:
            aligned_ph = self.alignment_by_path.get(row.path)
            if aligned_ph is None:
                if self.pac.missing_manifest_entry == "reject":
                    write_reject_row(
                        self.rejects_path,
                        {
                            "source_path": row.path,
                            "client_id": row.client_id,
                            "reason": "phoneme_alignment_missing_manifest",
                            "sentence_excerpt": excerpt,
                        },
                        self.reject_fields,
                    )
                    self.reject_reasons["phoneme_alignment_missing_manifest"] = (
                        self.reject_reasons.get("phoneme_alignment_missing_manifest", 0) + 1
                    )
                    return
                elif not phoneme_sequences_accept(
                    phonemes,
                    aligned_ph,
                    max_token_error_rate=self.pac.max_token_error_rate,
                ):
                    write_reject_row(
                        self.rejects_path,
                        {
                            "source_path": row.path,
                            "client_id": row.client_id,
                            "reason": "phoneme_alignment_mismatch",
                            "sentence_excerpt": excerpt,
                        },
                        self.reject_fields,
                    )
                    self.reject_reasons["phoneme_alignment_mismatch"] = (
                        self.reject_reasons.get("phoneme_alignment_mismatch", 0) + 1
                    )
                    return

        mora_early, mora_pref, mora_fin = _mora_gates_needed(self.lang, self.cfg, self.align_prefilter_qg)
        clip_mora_count, mora_reject_reason = _compute_clip_mora_count_once(
            text_norm,
            need_early=mora_early,
            need_pref=mora_pref,
            need_final=mora_fin,
            prefilter_mora_fail_reason=self.prefilter_mora_fail_reason,
        )
        if mora_reject_reason is not None:
            write_reject_row(
                self.rejects_path,
                {
                    "source_path": row.path,
                    "client_id": row.client_id,
                    "reason": mora_reject_reason,
                    "sentence_excerpt": excerpt,
                },
                self.reject_fields,
            )
            self.reject_reasons[mora_reject_reason] = self.reject_reasons.get(mora_reject_reason, 0) + 1
            return

        pfg_reuse = None
        pfg_fp = None

        clip_path = iter_clip_audio_paths(self.root, self.cfg.input.audio_subdir, row)
        if not clip_path.is_file():
            write_reject_row(
                self.rejects_path,
                {
                    "source_path": row.path,
                    "client_id": row.client_id,
                    "reason": "missing_audio",
                    "sentence_excerpt": excerpt,
                },
                self.reject_fields,
            )
            self.reject_reasons["missing_audio"] = self.reject_reasons.get("missing_audio", 0) + 1
            return

        try:
            y, sr = load_audio(clip_path)
        except Exception:
            write_reject_row(
                self.rejects_path,
                {
                    "source_path": row.path,
                    "client_id": row.client_id,
                    "reason": "decode_failed",
                    "sentence_excerpt": excerpt,
                },
                self.reject_fields,
            )
            self.reject_reasons["decode_failed"] = self.reject_reasons.get("decode_failed", 0) + 1
            return

        if not np.isfinite(y).all():
            write_reject_row(
                self.rejects_path,
                {
                    "source_path": row.path,
                    "client_id": row.client_id,
                    "reason": "nan_inf_audio",
                    "sentence_excerpt": excerpt,
                },
                self.reject_fields,
            )
            self.reject_reasons["nan_inf_audio"] = self.reject_reasons.get("nan_inf_audio", 0) + 1
            return

        if self.cfg.early_audio_gate.enabled:
            tsr = int(self.pipeline_for_pass1.target_sample_rate)
            y_chk = (
                resample_audio(np.asarray(y, dtype=np.float32), sr, tsr)
                if sr != tsr
                else np.asarray(y, dtype=np.float32)
            )
            eag = self.cfg.early_audio_gate
            mora_for_early = clip_mora_count if mora_early else None
            eg = run_early_audio_gate(
                y_chk,
                tsr,
                text_len=len(text_norm),
                mora_count=mora_for_early,
                main_gate=self.cfg.quality_gate,
                snr_cfg=self.cfg.snr,
                early=eag,
            )
            if not eg.ok:
                rsn = eg.reason or "early_gate"
                write_reject_row(
                    self.rejects_path,
                    {
                        "source_path": row.path,
                        "client_id": row.client_id,
                        "reason": rsn,
                        "sentence_excerpt": excerpt,
                    },
                    self.reject_fields,
                )
                self.reject_reasons[rsn] = self.reject_reasons.get(rsn, 0) + 1
                return

        try:
            y, sr, ameta = run_steps_on_array(y, sr, self.pipeline_for_pass1)
        except Exception as e:
            err_one_line = f"{type(e).__name__}: {e}".replace("\n", " ").strip()
            if len(err_one_line) > 220:
                err_one_line = err_one_line[:217] + "..."
            if not self.audio_pipeline_error_logged:
                self.audio_pipeline_error_logged = True
                print(
                    "[cv-preprocess] 音声パイプラインで例外（以降の同種失敗は rejects の reason に要約のみ記録）:",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exception(
                    type(e), e, e.__traceback__, limit=12, file=sys.stderr
                )
            write_reject_row(
                self.rejects_path,
                {
                    "source_path": row.path,
                    "client_id": row.client_id,
                    "reason": f"audio_pipeline_failed: {err_one_line}",
                    "sentence_excerpt": excerpt,
                },
                self.reject_fields,
            )
            self.reject_reasons["audio_pipeline_failed"] = self.reject_reasons.get("audio_pipeline_failed", 0) + 1
            return

        if self.cfg.mfa_gate.enabled and self.mfa_prefilter_qg is not None:
            mora_pf = clip_mora_count if mora_pref else None
            gate_pf = run_quality_gate(
                y,
                sr,
                text_len=len(text_norm),
                gate=self.mfa_prefilter_qg,
                snr_cfg=self.cfg.snr,
                mora_count=mora_pf,
            )
            if not gate_pf.ok:
                base_r = gate_pf.reason or "gate"
                pre_r = f"mfa_prefilter_{base_r}"
                write_reject_row(
                    self.rejects_path,
                    {
                        "source_path": row.path,
                        "client_id": row.client_id,
                        "reason": pre_r,
                        "sentence_excerpt": excerpt,
                    },
                    self.reject_fields,
                )
                self.reject_reasons[pre_r] = self.reject_reasons.get(pre_r, 0) + 1
                return
            pq = _maybe_prefilter_final_gate_reuse_pair(
                gate_pf,
                self.mfa_prefilter_qg,
                self.cfg,
                y,
                sr,
                len(text_norm),
                mora_pf,
                mora_fin,
                clip_mora_count,
            )
            if pq is not None:
                pfg_reuse, pfg_fp = pq

        if self.cfg.nfa_gate.enabled and self.nfa_prefilter_qg is not None:
            mora_pf = clip_mora_count if mora_pref else None
            gate_pf = run_quality_gate(
                y,
                sr,
                text_len=len(text_norm),
                gate=self.nfa_prefilter_qg,
                snr_cfg=self.cfg.snr,
                mora_count=mora_pf,
            )
            if not gate_pf.ok:
                base_r = gate_pf.reason or "gate"
                pre_r = f"nfa_prefilter_{base_r}"
                write_reject_row(
                    self.rejects_path,
                    {
                        "source_path": row.path,
                        "client_id": row.client_id,
                        "reason": pre_r,
                        "sentence_excerpt": excerpt,
                    },
                    self.reject_fields,
                )
                self.reject_reasons[pre_r] = self.reject_reasons.get(pre_r, 0) + 1
                return
            pq = _maybe_prefilter_final_gate_reuse_pair(
                gate_pf,
                self.nfa_prefilter_qg,
                self.cfg,
                y,
                sr,
                len(text_norm),
                mora_pf,
                mora_fin,
                clip_mora_count,
            )
            if pq is not None:
                pfg_reuse, pfg_fp = pq

        pending = PendingClip(
            row=row,
            y=y,
            sr=sr,
            text_raw=text_raw,
            text_norm=text_norm,
            phonemes=phonemes,
            excerpt=excerpt,
            ameta=ameta,
            mora_count=clip_mora_count if mora_fin else None,
            prefilter_final_gate_reuse=pfg_reuse,
            prefilter_final_gate_fp=pfg_fp,
        )
        if self.cfg.mfa_gate.enabled:
            pending.mfa_utt_id = f"u{self.mfa_utt_counter:08d}"
            self.mfa_utt_counter += 1
            self.mfa_batch.append(pending)
            if len(self.mfa_batch) >= self.mfa_bs_resolved:
                self.flush_mfa()
        elif self.cfg.nfa_gate.enabled:
            pending.nfa_utt_id = f"u{self.nfa_utt_counter:08d}"
            self.nfa_utt_counter += 1
            self.nfa_batch.append(pending)
            if len(self.nfa_batch) >= self.cfg.nfa_gate.batch_size:
                self.flush_nfa()
        elif self.cfg.asr_gate.enabled:
            self.asr_batch.append(pending)
            if len(self.asr_batch) >= self.cfg.asr_gate.batch_size:
                self.flush_asr()
        else:
            self.flush_finalize_survivors([pending])

