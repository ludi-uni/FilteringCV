from __future__ import annotations

import json
import random
import sys
import traceback

from tqdm import tqdm

from cv_preprocess.audio.asr_batch import close_asr_worker
from cv_preprocess.audio.mfa_batch import mfa_cli_available, run_mfa_align_batch
from cv_preprocess.audio.nfa_batch import close_nfa_worker, run_nfa_align_batch
from cv_preprocess.audio.sgmse_dereverb import maybe_warmup_sgmse
from cv_preprocess.audio.wpe_deepfilternet_denoise import maybe_warmup_wpe_deepfilternet
from cv_preprocess.config import PipelineConfig, QualityGateConfig
from cv_preprocess.io.alignment_phoneme_manifest import load_alignment_phoneme_manifest
from cv_preprocess.io.tsv_loader import (
    ClipRow,
    apply_merge_filtered_speakers_as_one,
    filter_by_clip_metadata,
    filter_by_speakers,
    load_validated_tsv,
)
from cv_preprocess.pipeline.export import append_jsonl, write_reject_row
from cv_preprocess.pipeline.ljspeech_tsv import write_ljspeech_validated_tsv
from cv_preprocess.pipeline.preprocess.clip_accept import process_pending_to_acceptance
from cv_preprocess.pipeline.preprocess.helpers import (
    _merged_quality_gate_for_mfa_prefilter,
    _merged_quality_gate_for_nfa_prefilter,
    infer_release,
)
from cv_preprocess.pipeline.preprocess.two_pass import (
    finalize_two_pass_denoise,
    finalize_two_pass_sgmse_microbatch,
)
from cv_preprocess.pipeline.preprocess.types import PendingClip
from cv_preprocess.pipeline.preprocess_efficiency import (
    effective_audio_catalog_for_preprocess,
    exclusive_single_sgmse_denoise_for_two_pass_batch,
    resolve_mfa_parallelism,
    resolve_preprocess_pass1_pipeline,
    two_pass_uses_split_pipelines,
)
from cv_preprocess.pipeline.split import assign_speaker_splits, build_counts
from cv_preprocess.text.mfa_token_map import map_mfa_space_separated_to_g2p_tokens
from cv_preprocess.text.normalize import normalize_for_tts
from cv_preprocess.text.phoneme_compare import phoneme_sequences_accept
from cv_preprocess.text.phonemize import g2p_phonemes

from cv_preprocess.pipeline.preprocess.asr_gate_apply import apply_asr_gate
from cv_preprocess.pipeline.preprocess.session_row import PreprocessRowMixin


class PreprocessSession(PreprocessRowMixin):
    """``run_preprocess`` の状態と MFA/NFA バッチ flush を束ねる。"""

    def __init__(self, cfg: PipelineConfig, *, show_progress: bool) -> None:
        self.cfg = cfg
        self.show_progress = show_progress
        root = cfg.input.corpus_root
        self.root = root
        tsv_path = root / cfg.input.clip_tsv
        rows, load_stats = load_validated_tsv(tsv_path)
        include_ids = cfg.speakers.include_client_ids or None
        rows = filter_by_speakers(rows, include_ids)
        self.rows_after_speaker_filter = len(rows)
        rows = filter_by_clip_metadata(rows, cfg.speakers.clip_metadata_filters)
        self.rows_after_metadata_filter = len(rows)
        rows = sorted(rows, key=lambda r: r.path)
        max_clips = cfg.input.max_clips
        if max_clips is not None and len(rows) > int(max_clips):
            rows = rows[: int(max_clips)]
        self.rows_after_max_clips_cap = len(rows)
        apply_merge_filtered_speakers_as_one(
            rows,
            enabled=cfg.speakers.merge_filtered_speakers_as_one,
            merged_client_id=cfg.speakers.resolved_merged_speaker_client_id(),
        )
        self.rows: list[ClipRow] = rows
        self.load_stats = load_stats
        self.include_ids = include_ids
        self.max_clips = max_clips

        out_root = cfg.output.root
        out_root.mkdir(parents=True, exist_ok=True)
        self.out_root = out_root

        if cfg.mfa_gate.enabled and not mfa_cli_available(cfg.mfa_gate.mfa_executable):
            raise ValueError(
                "mfa_gate.enabled ですが、MFA CLI が見つかりません "
                f"({cfg.mfa_gate.mfa_executable!r})。"
                "本リポジトリの Dev Container には Montreal Forced Aligner は含めていません。"
                "対処: (1) config で mfa_gate.enabled を false にする、"
                "(2) nfa_gate を有効にする（docs/開発環境.md・仕様.md §5.3）、"
                "(3) ホスト等で mfa を入れ mfa_executable にフルパスを書く、のいずれか。"
            )

        self.wav_dir = out_root / cfg.output.wav_subdir
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = out_root / cfg.output.manifest
        self.validated_tsv_path = out_root / cfg.output.validated_tsv
        self.rejects_path = out_root / cfg.output.rejects_name
        self.report_path = out_root / cfg.output.report_name

        if self.manifest_path.exists():
            self.manifest_path.unlink()
        if self.validated_tsv_path.exists():
            self.validated_tsv_path.unlink()
        if self.rejects_path.exists():
            self.rejects_path.unlink()

        self.reject_fields = ["source_path", "client_id", "reason", "sentence_excerpt"]

        pac = cfg.text.phoneme_alignment_check
        self.pac = pac
        alignment_by_path: dict[str, str] | None = None
        if pac.enabled:
            alignment_by_path = load_alignment_phoneme_manifest(pac.manifest_path)
        self.alignment_by_path = alignment_by_path

        self.accepted: list[dict] = []
        self.reject_reasons: dict[str, int] = {}
        self.lang = (cfg.input.locale_expected or "ja").split("-")[0]

        mfa_g2p_token_map: dict[str, str] = {}
        if cfg.mfa_gate.enabled and cfg.mfa_gate.compare_phones_to_g2p:
            from cv_preprocess.text.mfa_token_map import load_mfa_token_map_yaml

            mfa_g2p_token_map = load_mfa_token_map_yaml(cfg.mfa_gate.mfa_to_g2p_token_map_path)
        self.mfa_g2p_token_map = mfa_g2p_token_map

        nfa_g2p_token_map: dict[str, str] = {}
        if cfg.nfa_gate.enabled and cfg.nfa_gate.compare_tokens_to_g2p:
            from cv_preprocess.text.mfa_token_map import load_mfa_token_map_yaml

            nfa_g2p_token_map = load_mfa_token_map_yaml(cfg.nfa_gate.nfa_to_g2p_token_map_path)
        self.nfa_g2p_token_map = nfa_g2p_token_map

        self.mfa_nj_resolved, self.mfa_bs_resolved = resolve_mfa_parallelism(cfg.mfa_gate)

        self.accept_idx = 0
        self.audio_pipeline_error_logged = False
        self.mfa_batch: list[PendingClip] = []
        self.mfa_utt_counter = 0
        self.mfa_batches_flushed = 0
        self.nfa_batch: list[PendingClip] = []
        self.nfa_utt_counter = 0
        self.nfa_batches_flushed = 0
        self.asr_batch: list[PendingClip] = []
        self.asr_batches_flushed = 0
        self.mg_for_mfa = cfg.mfa_gate.model_copy(
            update={"num_jobs": self.mfa_nj_resolved, "batch_size": self.mfa_bs_resolved}
        )
        self.pipeline_for_pass1 = resolve_preprocess_pass1_pipeline(cfg)

        mfa_prefilter_qg: QualityGateConfig | None = None
        if cfg.mfa_gate.enabled and cfg.mfa_gate.prefilter.enabled:
            mfa_prefilter_qg = _merged_quality_gate_for_mfa_prefilter(cfg)
        self.mfa_prefilter_qg = mfa_prefilter_qg

        nfa_prefilter_qg: QualityGateConfig | None = None
        if cfg.nfa_gate.enabled and cfg.nfa_gate.prefilter.enabled:
            nfa_prefilter_qg = _merged_quality_gate_for_nfa_prefilter(cfg)
        self.nfa_prefilter_qg = nfa_prefilter_qg

        align_prefilter_qg: QualityGateConfig | None = (
            mfa_prefilter_qg if mfa_prefilter_qg is not None else nfa_prefilter_qg
        )
        self.align_prefilter_qg = align_prefilter_qg
        prefilter_mora_fail_reason: str | None = None
        if mfa_prefilter_qg is not None:
            prefilter_mora_fail_reason = "mfa_prefilter_mora_estimate_failed"
        elif nfa_prefilter_qg is not None:
            prefilter_mora_fail_reason = "nfa_prefilter_mora_estimate_failed"
        self.prefilter_mora_fail_reason = prefilter_mora_fail_reason

        self.two_pass_denoise_error_logged = False

        _cat = effective_audio_catalog_for_preprocess(cfg)
        self.exclusive_sgmse_step = exclusive_single_sgmse_denoise_for_two_pass_batch(list(_cat.steps))

        self.deferred_enhance_queue: list[PendingClip] = []

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

    def _map_split(self, name: str) -> str:
        if self.cfg.split.emit_split_as_dev and name == "val":
            return "dev"
        return name

    def run(self) -> dict:
        cfg = self.cfg
        rows = self.rows
        root = self.root
        out_root = self.out_root
        pac = self.pac
        alignment_by_path = self.alignment_by_path
        accepted = self.accepted

        if self.show_progress:
            align_banner = ""
            if cfg.mfa_gate.enabled:
                align_banner = (
                    f" mfa_batch_size={self.mfa_bs_resolved}(yaml {cfg.mfa_gate.batch_size})"
                    f" mfa_num_jobs={self.mfa_nj_resolved}(yaml {cfg.mfa_gate.num_jobs})"
                )
            elif cfg.nfa_gate.enabled:
                align_banner = f" nfa_batch_size={cfg.nfa_gate.batch_size}"
            if cfg.asr_gate.enabled:
                align_banner += f" asr_batch_size={cfg.asr_gate.batch_size}"
            if two_pass_uses_split_pipelines(cfg):
                assert cfg.audio_pipeline_align is not None and cfg.audio_pipeline_enhance is not None
                pipe_banner = (
                    f"align={cfg.audio_pipeline_align.audio_pipeline_id!r} "
                    f"enhance={cfg.audio_pipeline_enhance.audio_pipeline_id!r}"
                )
            else:
                pipe_banner = f"pipeline={cfg.audio_pipeline.audio_pipeline_id!r}"
            print(
                "[cv-preprocess] preprocess: "
                f"clips={len(rows)} "
                f"(after_speaker={self.rows_after_speaker_filter} after_metadata={self.rows_after_metadata_filter}"
                f"{f' max_clips={self.max_clips}' if self.max_clips is not None else ''}) "
                f"{pipe_banner} "
                f"out={out_root}"
                f"{align_banner}",
                file=sys.stderr,
                flush=True,
            )

        if not (
            cfg.two_pass_denoise.enabled
            and str(cfg.two_pass_denoise.enhance_phase).strip().lower() == "after_align_complete"
        ):
            ap_warm = effective_audio_catalog_for_preprocess(cfg)
            maybe_warmup_sgmse(ap_warm)
            maybe_warmup_wpe_deepfilternet(ap_warm)

        row_iter = rows
        if self.show_progress and len(rows) > 0:
            row_iter = tqdm(
                rows,
                desc="preprocess",
                unit="clip",
                total=len(rows),
                file=sys.stderr,
                dynamic_ncols=True,
                mininterval=0.25,
            )

        for row in row_iter:
            try:
                self._preprocess_one_row(row)
            finally:
                if isinstance(row_iter, tqdm):
                    pf: dict[str, object] = {
                        "accepted": self.accept_idx,
                        "rej": sum(self.reject_reasons.values()),
                    }
                    if cfg.mfa_gate.enabled:
                        pf["mfa_buf"] = len(self.mfa_batch)
                        pf["mfa_bs"] = self.mfa_bs_resolved
                    elif cfg.nfa_gate.enabled:
                        pf["nfa_buf"] = len(self.nfa_batch)
                        pf["nfa_bs"] = cfg.nfa_gate.batch_size
                    if cfg.asr_gate.enabled:
                        pf["asr_buf"] = len(self.asr_batch)
                        pf["asr_bs"] = cfg.asr_gate.batch_size
                    if cfg.two_pass_denoise.enabled and self._enhance_phase_is_after_align_complete():
                        pf["defer"] = len(self.deferred_enhance_queue)
                    row_iter.set_postfix(**pf, refresh=False)

        self.flush_mfa()
        self.flush_nfa()
        if cfg.nfa_gate.enabled:
            close_nfa_worker()
        self.flush_asr()
        if cfg.asr_gate.enabled:
            close_asr_worker()

        if (
            cfg.two_pass_denoise.enabled
            and str(cfg.two_pass_denoise.enhance_phase).strip().lower() == "after_align_complete"
            and self.deferred_enhance_queue
        ):
            if self.show_progress:
                n_def = len(self.deferred_enhance_queue)
                print(
                    f"[cv-preprocess] 二段 denoise enhance_phase=after_align_complete: "
                    f"足切り済み {n_def} クリップのエンハンスを開始（NFA 子プロセスは既に終了）",
                    file=sys.stderr,
                    flush=True,
                )
            ap_warm2 = effective_audio_catalog_for_preprocess(cfg)
            maybe_warmup_sgmse(ap_warm2)
            maybe_warmup_wpe_deepfilternet(ap_warm2)
            pending_enhance = list(self.deferred_enhance_queue)
            self.deferred_enhance_queue.clear()
            self._enhance_survivors_and_accept(
                pending_enhance, apply_release=False, progress_bar=self.show_progress
            )

        if cfg.split.mode == "random" and accepted:
            rng = random.Random(cfg.split.seed)
            order = list(range(len(accepted)))
            rng.shuffle(order)
            n = len(accepted)
            nt = int(n * cfg.split.train)
            nv = int(n * cfg.split.val)
            if nt + nv > n:
                nv = max(0, n - nt)
            by_pos_split: dict[int, str] = {}
            for i, pos in enumerate(order):
                if i < nt:
                    by_pos_split[pos] = "train"
                elif i < nt + nv:
                    by_pos_split[pos] = "val"
                else:
                    by_pos_split[pos] = "test"
            for i, r in enumerate(accepted):
                r["split"] = self._map_split(by_pos_split[i])
        else:
            pairs = [(r["speaker_id"], r["utt_id"]) for r in accepted]
            counts = build_counts(pairs)
            speakers = list(counts.keys())
            if speakers:
                sp_split = assign_speaker_splits(
                    speakers,
                    counts,
                    train=cfg.split.train,
                    val=cfg.split.val,
                    test=cfg.split.test,
                    seed=cfg.split.seed,
                )
            else:
                sp_split = {}
            for r in accepted:
                sp = sp_split.get(r["speaker_id"], "train")
                r["split"] = self._map_split(sp)

        manifest_path = self.manifest_path
        validated_tsv_path = self.validated_tsv_path
        report_path = self.report_path

        for r in accepted:
            append_jsonl(manifest_path, r)

        write_ljspeech_validated_tsv(validated_tsv_path, accepted)

        report_warnings: list[str] = []
        if self.include_ids and self.rows_after_speaker_filter == 0:
            report_warnings.append(
                "話者フィルタ後の行数が 0 です。preprocess はスキップ相当です。"
                " corpus_root の validated.tsv に speakers.include_client_ids が存在するか確認するか、"
                " include_client_ids: [] で全話者を指定してください。"
            )
        if cfg.speakers.clip_metadata_filters.is_active() and self.rows_after_metadata_filter == 0:
            report_warnings.append(
                "speakers.clip_metadata_filters 適用後の行数が 0 です。"
                " gender / age 等の許容値とコーパスの列値（空欄の有無）を確認してください。"
                " 空セル行を残したい軸では許容リストに \"\" を含めます。"
            )

        report = {
            "report_schema_version": 1,
            "corpus_root": str(root.resolve()),
            "source_release": infer_release(root),
            "output_manifest": str(manifest_path.resolve()),
            "output_validated_tsv": str(validated_tsv_path.resolve()),
            "audio_pipeline_id": effective_audio_catalog_for_preprocess(cfg).audio_pipeline_id,
            "two_pass_split_pipelines": two_pass_uses_split_pipelines(cfg),
            "audio_pipeline_align_id": cfg.audio_pipeline_align.audio_pipeline_id
            if cfg.audio_pipeline_align
            else None,
            "audio_pipeline_enhance_id": cfg.audio_pipeline_enhance.audio_pipeline_id
            if cfg.audio_pipeline_enhance
            else None,
            "quality_gate_profile": cfg.quality_gate_profile,
            "quality_gate_profiles_keys": sorted(cfg.quality_gate_profiles.keys()),
            "load_stats": self.load_stats,
            "rows_after_speaker_filter": self.rows_after_speaker_filter,
            "rows_after_clip_metadata_filter": self.rows_after_metadata_filter,
            "input_max_clips": self.max_clips,
            "rows_after_max_clips_cap": self.rows_after_max_clips_cap,
            "clip_metadata_filters": cfg.speakers.clip_metadata_filters.model_dump(),
            "speaker_filter_list_size": len(self.include_ids) if self.include_ids else 0,
            "warnings": report_warnings,
            "accepted": len(accepted),
            "rejected_by_reason": self.reject_reasons,
            "split": {
                "mode": cfg.split.mode,
                "seed": cfg.split.seed,
                "train": cfg.split.train,
                "val": cfg.split.val,
                "test": cfg.split.test,
                "emit_split_as_dev": cfg.split.emit_split_as_dev,
            },
            "phoneme_alignment_check": {
                "enabled": pac.enabled,
                "manifest_path": str(pac.manifest_path.resolve()) if pac.manifest_path else None,
                "manifest_entries": len(alignment_by_path) if alignment_by_path else 0,
                "missing_manifest_entry": pac.missing_manifest_entry,
                "max_token_error_rate": pac.max_token_error_rate,
            },
            "early_audio_gate": cfg.early_audio_gate.model_dump(),
            "two_pass_denoise": cfg.two_pass_denoise.model_dump(),
            "mfa_gate": {
                "enabled": cfg.mfa_gate.enabled,
                "batches_flushed": self.mfa_batches_flushed,
                "dictionary": cfg.mfa_gate.dictionary,
                "acoustic_model": cfg.mfa_gate.acoustic_model,
                "batch_size": cfg.mfa_gate.batch_size,
                "batch_size_resolved": self.mfa_bs_resolved,
                "num_jobs": cfg.mfa_gate.num_jobs,
                "num_jobs_resolved": self.mfa_nj_resolved,
                "auto_num_jobs": cfg.mfa_gate.auto_num_jobs,
                "auto_scale_batch_size": cfg.mfa_gate.auto_scale_batch_size,
                "prefilter": {
                    "enabled": cfg.mfa_gate.prefilter.enabled,
                    "quality_gate_override_keys": sorted(
                        cfg.mfa_gate.prefilter.quality_gate_overrides.keys()
                    ),
                },
                "compare_phones_to_g2p": cfg.mfa_gate.compare_phones_to_g2p,
                "max_token_error_rate_vs_g2p": cfg.mfa_gate.max_token_error_rate_vs_g2p,
                "mfa_to_g2p_token_map_path": str(cfg.mfa_gate.mfa_to_g2p_token_map_path.resolve())
                if cfg.mfa_gate.mfa_to_g2p_token_map_path
                else None,
                "mfa_to_g2p_token_map_size": len(self.mfa_g2p_token_map),
            },
            "nfa_gate": {
                "enabled": cfg.nfa_gate.enabled,
                "batches_flushed": self.nfa_batches_flushed,
                "pretrained_name": cfg.nfa_gate.pretrained_name,
                "model_path": str(cfg.nfa_gate.model_path.resolve())
                if cfg.nfa_gate.model_path
                else None,
                "model_sample_rate_hz": cfg.nfa_gate.model_sample_rate_hz,
                "batch_size": cfg.nfa_gate.batch_size,
                "persistent_worker": cfg.nfa_gate.persistent_worker,
                "use_local_attention": cfg.nfa_gate.use_local_attention,
                "prefilter": {
                    "enabled": cfg.nfa_gate.prefilter.enabled,
                    "quality_gate_override_keys": sorted(
                        cfg.nfa_gate.prefilter.quality_gate_overrides.keys()
                    ),
                },
                "align_using_pred_text": cfg.nfa_gate.align_using_pred_text,
                "compare_pred_text_to_norm": cfg.nfa_gate.compare_pred_text_to_norm,
                "max_pred_phoneme_error_rate_vs_norm": cfg.nfa_gate.max_pred_phoneme_error_rate_vs_norm,
                "compare_tokens_to_g2p": cfg.nfa_gate.compare_tokens_to_g2p,
                "max_token_error_rate_vs_g2p": cfg.nfa_gate.max_token_error_rate_vs_g2p,
                "nfa_to_g2p_token_map_path": str(cfg.nfa_gate.nfa_to_g2p_token_map_path.resolve())
                if cfg.nfa_gate.nfa_to_g2p_token_map_path
                else None,
                "nfa_to_g2p_token_map_size": len(self.nfa_g2p_token_map),
            },
            "asr_gate": {
                "enabled": cfg.asr_gate.enabled,
                "batches_flushed": self.asr_batches_flushed,
                "backend": cfg.asr_gate.backend,
                "pretrained_name": cfg.asr_gate.pretrained_name,
                "model_path": str(cfg.asr_gate.model_path.resolve())
                if cfg.asr_gate.model_path
                else None,
                "sample_rate_hz": cfg.asr_gate.sample_rate_hz,
                "batch_size": cfg.asr_gate.batch_size,
                "persistent_worker": cfg.asr_gate.persistent_worker,
                "mock_mode": cfg.asr_gate.mock_mode,
                "compare_text": cfg.asr_gate.compare_text,
                "compare_phonemes": cfg.asr_gate.compare_phonemes,
                "max_char_error_rate": cfg.asr_gate.max_char_error_rate,
                "max_phoneme_error_rate": cfg.asr_gate.max_phoneme_error_rate,
                "min_asr_confidence": cfg.asr_gate.min_asr_confidence,
            },
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        return report
