"""preprocess 完了レポート JSON の組み立て（``PreprocessSession`` 用ミックスイン）。"""

from __future__ import annotations

from typing import Any

from cv_preprocess.pipeline.preprocess.helpers import effective_final_quality_gate, infer_release
from cv_preprocess.pipeline.preprocess_efficiency import (
    effective_audio_catalog_for_preprocess,
    two_pass_uses_split_pipelines,
)


class PreprocessReportMixin:
    def _build_preprocess_report_dict(self) -> dict[str, Any]:
        cfg = self.cfg
        pac = self.pac
        alignment_by_path = self.alignment_by_path
        accepted = self.accepted
        root = self.root

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

        return {
            "report_schema_version": 1,
            "corpus_root": str(root.resolve()),
            "source_release": infer_release(root),
            "output_manifest": str(self.manifest_path.resolve()),
            "output_validated_tsv": str(self.validated_tsv_path.resolve()),
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
                "quality_gate_profile": cfg.nfa_gate.quality_gate_profile,
                "quality_gate_override_keys": sorted(cfg.nfa_gate.quality_gate_overrides.keys()),
                "final_quality_gate_min_estimated_snr_db": effective_final_quality_gate(cfg).min_estimated_snr_db
                if cfg.nfa_gate.enabled
                else None,
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
