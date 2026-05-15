from __future__ import annotations

import random
import sys

from tqdm import tqdm

from cv_preprocess.audio.asr_batch import close_asr_worker
from cv_preprocess.audio.mfa_batch import mfa_cli_available
from cv_preprocess.audio.nfa_batch import close_nfa_worker
from cv_preprocess.audio.sgmse_dereverb import maybe_warmup_sgmse
from cv_preprocess.audio.wpe_deepfilternet_denoise import maybe_warmup_wpe_deepfilternet
from cv_preprocess.config import PipelineConfig, QualityGateConfig
from cv_preprocess.io.alignment_phoneme_manifest import load_alignment_phoneme_manifest
from cv_preprocess.io.tsv_loader import ClipRow, load_clip_rows_for_pipeline
from cv_preprocess.pipeline.export import append_jsonl, write_json_report
from cv_preprocess.pipeline.ljspeech_tsv import write_ljspeech_validated_tsv
from cv_preprocess.pipeline.preprocess.helpers import (
    _merged_quality_gate_for_mfa_prefilter,
    _merged_quality_gate_for_nfa_prefilter,
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

from cv_preprocess.pipeline.preprocess.session_flush_mixin import PreprocessFlushMixin
from cv_preprocess.pipeline.preprocess.session_report_mixin import PreprocessReportMixin
from cv_preprocess.pipeline.preprocess.session_row import PreprocessRowMixin


class PreprocessSession(PreprocessRowMixin, PreprocessFlushMixin, PreprocessReportMixin):
    """``run_preprocess`` の状態と MFA/NFA バッチ flush を束ねる。"""

    def __init__(self, cfg: PipelineConfig, *, show_progress: bool) -> None:
        self.cfg = cfg
        self.show_progress = show_progress
        root = cfg.input.corpus_root
        self.root = root
        loaded = load_clip_rows_for_pipeline(cfg, apply_input_max_clips=True)
        rows = loaded.rows
        self.rows_after_speaker_filter = loaded.rows_after_speaker_filter
        self.rows_after_metadata_filter = loaded.rows_after_metadata_filter
        self.rows_after_max_clips_cap = loaded.rows_after_max_clips_cap
        self.rows: list[ClipRow] = rows
        self.load_stats = loaded.load_stats
        self.include_ids = loaded.include_client_ids
        self.max_clips = cfg.input.max_clips

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

    def _map_split(self, name: str) -> str:
        if self.cfg.split.emit_split_as_dev and name == "val":
            return "dev"
        return name

    def run(self) -> dict:
        cfg = self.cfg
        rows = self.rows
        root = self.root
        out_root = self.out_root
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

        for r in accepted:
            append_jsonl(self.manifest_path, r)

        write_ljspeech_validated_tsv(self.validated_tsv_path, accepted)

        report = self._build_preprocess_report_dict()
        write_json_report(self.report_path, report)

        return report
