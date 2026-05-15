"""MFA/NFA/ASR バッチ flush と二段 denoise enhance（``PreprocessSession`` 用ミックスイン）。"""

from __future__ import annotations

import sys
import traceback

from tqdm import tqdm

from cv_preprocess.audio.mfa_batch import run_mfa_align_batch
from cv_preprocess.audio.nfa_batch import close_nfa_worker, run_nfa_align_batch
from cv_preprocess.pipeline.export import write_reject_row
from cv_preprocess.pipeline.preprocess.asr_gate_apply import apply_asr_gate
from cv_preprocess.pipeline.preprocess.clip_accept import process_pending_to_acceptance
from cv_preprocess.pipeline.preprocess.two_pass import (
    finalize_two_pass_denoise,
    finalize_two_pass_sgmse_microbatch,
)
from cv_preprocess.pipeline.preprocess.types import PendingClip
from cv_preprocess.text.mfa_token_map import map_mfa_space_separated_to_g2p_tokens
from cv_preprocess.text.normalize import normalize_for_tts
from cv_preprocess.text.phoneme_compare import phoneme_sequences_accept
from cv_preprocess.text.phonemize import g2p_phonemes


class PreprocessFlushMixin:
    def _enhance_phase_is_after_align_complete(self) -> bool:
        c = self.cfg
        return (
            c.two_pass_denoise.enabled
            and str(c.two_pass_denoise.enhance_phase).strip().lower() == "after_align_complete"
        )

    def _two_pass_denoise_exception(self, p_clip: PendingClip, e: Exception) -> None:
        err_one_line = f"{type(e).__name__}: {e}".replace("\n", " ").strip()
        if len(err_one_line) > 220:
            err_one_line = err_one_line[:217] + "..."
        if not self.two_pass_denoise_error_logged:
            self.two_pass_denoise_error_logged = True
            print(
                "[cv-preprocess] 二段 denoise で例外（以降は rejects の reason に要約のみ記録）:",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exception(
                type(e), e, e.__traceback__, limit=12, file=sys.stderr
            )
        write_reject_row(
            self.rejects_path,
            {
                "source_path": p_clip.row.path,
                "client_id": p_clip.row.client_id,
                "reason": f"two_pass_denoise_failed: {err_one_line}",
                "sentence_excerpt": p_clip.excerpt,
            },
            self.reject_fields,
        )
        self.reject_reasons["two_pass_denoise_failed"] = (
            self.reject_reasons.get("two_pass_denoise_failed", 0) + 1
        )

    def _enhance_survivors_and_accept(
        self,
        survivors: list[PendingClip],
        *,
        apply_release: bool,
        progress_bar: bool = False,
    ) -> None:
        cfg = self.cfg
        if not survivors:
            return
        if (
            apply_release
            and cfg.two_pass_denoise.enabled
            and cfg.nfa_gate.enabled
            and cfg.nfa_gate.persistent_worker
            and cfg.nfa_gate.release_persistent_worker_before_two_pass_enhance
        ):
            close_nfa_worker()
        use_micro = cfg.two_pass_denoise.enabled and self.exclusive_sgmse_step is not None
        show_enhance_tqdm = bool(progress_bar and self.show_progress)
        if use_micro:
            micro_bs = max(1, int(cfg.two_pass_denoise.sgmse_micro_batch_max))
            pbar: tqdm | None = None
            if show_enhance_tqdm:
                pbar = tqdm(
                    total=len(survivors),
                    desc="preprocess-enhance",
                    unit="clip",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    mininterval=0.25,
                )
            try:
                for j in range(0, len(survivors), micro_bs):
                    chunk = survivors[j : j + micro_bs]
                    try:
                        fins = finalize_two_pass_sgmse_microbatch(
                            chunk, cfg, self.exclusive_sgmse_step
                        )
                    except Exception:
                        for p in chunk:
                            try:
                                p_fin = finalize_two_pass_denoise(p, cfg)
                            except Exception as e2:
                                self._two_pass_denoise_exception(p, e2)
                                continue
                            self.accept_idx = process_pending_to_acceptance(
                                p_fin,
                                cfg=cfg,
                                root=self.root,
                                out_root=self.out_root,
                                lang=self.lang,
                                rejects_path=self.rejects_path,
                                reject_fields=self.reject_fields,
                                reject_reasons=self.reject_reasons,
                                accepted=self.accepted,
                                accept_idx=self.accept_idx,
                            )
                    else:
                        for p_fin in fins:
                            self.accept_idx = process_pending_to_acceptance(
                                p_fin,
                                cfg=cfg,
                                root=self.root,
                                out_root=self.out_root,
                                lang=self.lang,
                                rejects_path=self.rejects_path,
                                reject_fields=self.reject_fields,
                                reject_reasons=self.reject_reasons,
                                accepted=self.accepted,
                                accept_idx=self.accept_idx,
                            )
                    if pbar is not None:
                        pbar.update(len(chunk))
            finally:
                if pbar is not None:
                    pbar.close()
            return

        iter_survivors: list[PendingClip] | tqdm = survivors
        if show_enhance_tqdm:
            iter_survivors = tqdm(
                survivors,
                desc="preprocess-enhance",
                unit="clip",
                total=len(survivors),
                file=sys.stderr,
                dynamic_ncols=True,
                mininterval=0.25,
            )
        for p in iter_survivors:
            try:
                p_fin = finalize_two_pass_denoise(p, cfg)
            except Exception as e:
                self._two_pass_denoise_exception(p, e)
                continue
            self.accept_idx = process_pending_to_acceptance(
                p_fin,
                cfg=cfg,
                root=self.root,
                out_root=self.out_root,
                lang=self.lang,
                rejects_path=self.rejects_path,
                reject_fields=self.reject_fields,
                reject_reasons=self.reject_reasons,
                accepted=self.accepted,
                accept_idx=self.accept_idx,
            )

    def flush_finalize_survivors(self, survivors: list[PendingClip]) -> None:
        if not survivors:
            return
        if self.cfg.asr_gate.enabled:
            survivors = apply_asr_gate(
                self.cfg,
                survivors,
                work_parent=self.out_root.resolve(),
                rejects_path=self.rejects_path,
                reject_fields=self.reject_fields,
                reject_reasons=self.reject_reasons,
            )
            self.asr_batches_flushed += 1
        if not survivors:
            return
        if self.cfg.two_pass_denoise.enabled and self._enhance_phase_is_after_align_complete():
            self.deferred_enhance_queue.extend(survivors)
            return
        self._enhance_survivors_and_accept(survivors, apply_release=True)

    def flush_mfa(self) -> None:
        cfg = self.cfg
        if not self.mfa_batch:
            return
        mg = cfg.mfa_gate
        items = [(p.mfa_utt_id, p.y, p.sr, p.text_norm) for p in self.mfa_batch]
        results = run_mfa_align_batch(self.mg_for_mfa, items, work_parent=self.out_root.resolve())
        survivors: list[PendingClip] = []
        for p, res in zip(self.mfa_batch, results):
            if not res.ok:
                write_reject_row(
                    self.rejects_path,
                    {
                        "source_path": p.row.path,
                        "client_id": p.row.client_id,
                        "reason": "mfa_align_failed",
                        "sentence_excerpt": p.excerpt,
                    },
                    self.reject_fields,
                )
                self.reject_reasons["mfa_align_failed"] = self.reject_reasons.get("mfa_align_failed", 0) + 1
                continue
            if mg.compare_phones_to_g2p:
                g2p_s = (p.phonemes or "").strip()
                mfa_raw = (res.phone_string or "").strip()
                mfa_s = map_mfa_space_separated_to_g2p_tokens(mfa_raw, self.mfa_g2p_token_map)
                if not phoneme_sequences_accept(
                    g2p_s,
                    mfa_s,
                    max_token_error_rate=mg.max_token_error_rate_vs_g2p,
                ):
                    write_reject_row(
                        self.rejects_path,
                        {
                            "source_path": p.row.path,
                            "client_id": p.row.client_id,
                            "reason": "mfa_phoneme_mismatch",
                            "sentence_excerpt": p.excerpt,
                        },
                        self.reject_fields,
                    )
                    self.reject_reasons["mfa_phoneme_mismatch"] = (
                        self.reject_reasons.get("mfa_phoneme_mismatch", 0) + 1
                    )
                    continue
            survivors.append(p)
        self.flush_finalize_survivors(survivors)
        self.mfa_batch.clear()
        self.mfa_batches_flushed += 1

    def flush_nfa(self) -> None:
        cfg = self.cfg
        if not self.nfa_batch:
            return
        ng = cfg.nfa_gate
        items = [(p.nfa_utt_id, p.y, p.sr, p.text_norm) for p in self.nfa_batch]
        results = run_nfa_align_batch(ng, items, work_parent=self.out_root.resolve())
        survivors: list[PendingClip] = []
        for p, res in zip(self.nfa_batch, results):
            if not res.ok:
                write_reject_row(
                    self.rejects_path,
                    {
                        "source_path": p.row.path,
                        "client_id": p.row.client_id,
                        "reason": "nfa_align_failed",
                        "sentence_excerpt": p.excerpt,
                    },
                    self.reject_fields,
                )
                self.reject_reasons["nfa_align_failed"] = self.reject_reasons.get("nfa_align_failed", 0) + 1
                continue
            if ng.compare_pred_text_to_norm:
                hyp = (res.pred_text or "").strip()
                if not hyp:
                    write_reject_row(
                        self.rejects_path,
                        {
                            "source_path": p.row.path,
                            "client_id": p.row.client_id,
                            "reason": "nfa_pred_text_missing",
                            "sentence_excerpt": p.excerpt,
                        },
                        self.reject_fields,
                    )
                    self.reject_reasons["nfa_pred_text_missing"] = (
                        self.reject_reasons.get("nfa_pred_text_missing", 0) + 1
                    )
                    continue
                ref_ph = (p.phonemes or "").strip()
                if not ref_ph:
                    write_reject_row(
                        self.rejects_path,
                        {
                            "source_path": p.row.path,
                            "client_id": p.row.client_id,
                            "reason": "nfa_ref_phonemes_missing",
                            "sentence_excerpt": p.excerpt,
                        },
                        self.reject_fields,
                    )
                    self.reject_reasons["nfa_ref_phonemes_missing"] = (
                        self.reject_reasons.get("nfa_ref_phonemes_missing", 0) + 1
                    )
                    continue
                hyp_norm = normalize_for_tts(hyp)
                try:
                    pred_ph = g2p_phonemes(hyp_norm, kana=cfg.text.g2p_kana)
                except Exception:
                    write_reject_row(
                        self.rejects_path,
                        {
                            "source_path": p.row.path,
                            "client_id": p.row.client_id,
                            "reason": "nfa_pred_phonemize_failed",
                            "sentence_excerpt": p.excerpt,
                        },
                        self.reject_fields,
                    )
                    self.reject_reasons["nfa_pred_phonemize_failed"] = (
                        self.reject_reasons.get("nfa_pred_phonemize_failed", 0) + 1
                    )
                    continue
                if not phoneme_sequences_accept(
                    ref_ph,
                    pred_ph,
                    max_token_error_rate=ng.max_pred_phoneme_error_rate_vs_norm,
                ):
                    write_reject_row(
                        self.rejects_path,
                        {
                            "source_path": p.row.path,
                            "client_id": p.row.client_id,
                            "reason": "nfa_pred_phoneme_mismatch",
                            "sentence_excerpt": p.excerpt,
                        },
                        self.reject_fields,
                    )
                    self.reject_reasons["nfa_pred_phoneme_mismatch"] = (
                        self.reject_reasons.get("nfa_pred_phoneme_mismatch", 0) + 1
                    )
                    continue
            elif ng.compare_tokens_to_g2p:
                g2p_s = (p.phonemes or "").strip()
                nfa_raw = (res.token_string or "").strip()
                nfa_s = map_mfa_space_separated_to_g2p_tokens(nfa_raw, self.nfa_g2p_token_map)
                if not phoneme_sequences_accept(
                    g2p_s,
                    nfa_s,
                    max_token_error_rate=ng.max_token_error_rate_vs_g2p,
                ):
                    write_reject_row(
                        self.rejects_path,
                        {
                            "source_path": p.row.path,
                            "client_id": p.row.client_id,
                            "reason": "nfa_token_mismatch",
                            "sentence_excerpt": p.excerpt,
                        },
                        self.reject_fields,
                    )
                    self.reject_reasons["nfa_token_mismatch"] = (
                        self.reject_reasons.get("nfa_token_mismatch", 0) + 1
                    )
                    continue
            survivors.append(p)
        self.flush_finalize_survivors(survivors)
        self.nfa_batch.clear()
        self.nfa_batches_flushed += 1

    def flush_asr(self) -> None:
        if not self.asr_batch:
            return
        chunk = list(self.asr_batch)
        self.asr_batch.clear()
        self.flush_finalize_survivors(chunk)
