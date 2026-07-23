from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from cv_preprocess.application.common import AnalyzeResult, CancellationToken, ProgressEvent, ProgressSink
from cv_preprocess.audio.decode import load_audio
from cv_preprocess.audio.pipeline import run_steps_on_array
from cv_preprocess.audio.quality_gate import run_early_audio_gate, run_quality_gate
from cv_preprocess.audio.resample import resample_audio
from cv_preprocess.catalog.cache import cached_wav_path, pipeline_cache_key, write_wav_atomic
from cv_preprocess.catalog.ids import stable_clip_id
from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.catalog.aggregates import build_speaker_stats
from cv_preprocess.compute.loader import resolve_compute_backend
from cv_preprocess.catalog.linguistic_enrich import enrich_row_with_linguistic_features
from cv_preprocess.catalog.writer import write_catalog_bundle
from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import ClipRow, iter_clip_audio_paths, load_clip_rows_for_pipeline
from cv_preprocess.pipeline.g2p_map_suggest_core import validate_clip_text_norm
from cv_preprocess.pipeline.preprocess.helpers import (
    _compute_clip_mora_count_once,
    _mora_gates_needed,
    effective_final_quality_gate,
    infer_release,
)
from cv_preprocess.pipeline.preprocess.two_pass import finalize_two_pass_denoise
from cv_preprocess.pipeline.preprocess.types import PendingClip
from cv_preprocess.pipeline.preprocess_efficiency import resolve_preprocess_pass1_pipeline
from cv_preprocess.text.phonemize import g2p_phonemes_for_dataset


@dataclass(frozen=True)
class ClipAnalyzeOutcome:
    row: dict[str, object]
    disposition: ClipDisposition
    processed_y: np.ndarray | None = None
    processed_sr: int | None = None
    audio_sha256: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_relative_path(path: str) -> str:
    return path.replace("\\", "/")


def _sorted_clip_rows(rows: list[ClipRow]) -> list[tuple[int, ClipRow]]:
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda item: (_normalize_relative_path(item[1].path), item[0]))
    return indexed


def _linguistic_module_available() -> bool:
    return importlib.util.find_spec("cv_preprocess.linguistic") is not None


def _phonemes_for_row(text_norm: str, cfg: PipelineConfig) -> tuple[str | None, str]:
    if not cfg.text.phonemize:
        return None, "none"
    try:
        phonemes = g2p_phonemes_for_dataset(
            text_norm,
            kana=cfg.text.g2p_kana,
            word_separator=cfg.text.phoneme_word_separator,
        )
    except Exception:
        return None, "none"
    return phonemes, "text_g2p"


def _base_catalog_row(
    *,
    clip_id: str,
    config: PipelineConfig,
    row: ClipRow,
    source_row_index: int,
    audio_sha256: str,
    text_raw: str,
    text_norm: str | None,
    pipeline_hash: str,
    source_release: str,
) -> dict[str, object]:
    return {
        "clip_id": clip_id,
        "source_release": source_release,
        "normalized_relative_source_path": _normalize_relative_path(row.path),
        "source_row_index": int(source_row_index),
        "audio_sha256": audio_sha256,
        "text_raw": text_raw,
        "text_norm": text_norm,
        "speaker_id": row.client_id,
        "sentence_id": row.sentence_id,
        "pipeline_hash": pipeline_hash,
        "split": None,
        "duplicate_group_ids": None,
        "selection_rank": None,
        "selection_utility": None,
        "override_flags": None,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "phonemes": None,
        "feature_source": None,
        "biphones": None,
        "triphones": None,
        "moras": None,
        "fullcontext_labels": None,
        "analysis_warnings": None,
        "audio_cache_rel_path": None,
        "duration_sec": None,
        "quality_score": None,
        "estimated_snr_db": None,
        "silence_ratio": None,
        "reject_reason": None,
        "disposition": None,
    }


def _hard_rejected_row(
    base: dict[str, object],
    *,
    reason: str,
) -> ClipAnalyzeOutcome:
    out = dict(base)
    out["disposition"] = ClipDisposition.HARD_REJECTED.value
    out["reject_reason"] = reason
    return ClipAnalyzeOutcome(row=out, disposition=ClipDisposition.HARD_REJECTED)


def analyze_clip_with_gates(
    row: ClipRow,
    *,
    config: PipelineConfig,
    source_row_index: int,
    pipeline_hash: str,
    source_release: str,
    root: Path,
    lang: str,
) -> ClipAnalyzeOutcome:
    """Decode and run the builder analyze path without speaker caps or MFA/NFA/ASR."""
    text_raw = row.sentence
    audio_sha256 = ""
    clip_path = iter_clip_audio_paths(root, config.input.audio_subdir, row)
    if clip_path.is_file():
        try:
            audio_sha256 = _sha256_file(clip_path)
        except OSError:
            audio_sha256 = ""

    clip_id = stable_clip_id(
        source_release=source_release,
        normalized_relative_source_path=_normalize_relative_path(row.path),
        source_row_index=source_row_index,
        audio_sha256=audio_sha256,
        text_raw=text_raw,
    )
    base = _base_catalog_row(
        clip_id=clip_id,
        config=config,
        row=row,
        source_row_index=source_row_index,
        audio_sha256=audio_sha256,
        text_raw=text_raw,
        text_norm=None,
        pipeline_hash=pipeline_hash,
        source_release=source_release,
    )

    text_norm, text_rej = validate_clip_text_norm(row, config)
    if text_rej is not None:
        return _hard_rejected_row(base, reason=text_rej)
    base["text_norm"] = text_norm

    phonemes, feature_source = _phonemes_for_row(text_norm, config)
    if config.text.phonemize and phonemes is None:
        return _hard_rejected_row(base, reason="phonemize_failed")
    base["phonemes"] = phonemes
    base["feature_source"] = feature_source if phonemes else None
    enrich_row_with_linguistic_features(
        base,
        text_norm=text_norm,
        phonemes=phonemes,
        duration_sec=None,
        config=config,
    )

    mora_early, mora_pref, mora_fin = _mora_gates_needed(lang, config, align_prefilter_qg=None)
    clip_mora_count, mora_reject_reason = _compute_clip_mora_count_once(
        text_norm,
        need_early=mora_early,
        need_pref=False,
        need_final=mora_fin,
    )
    if mora_reject_reason is not None:
        return _hard_rejected_row(base, reason=mora_reject_reason)

    if not clip_path.is_file():
        return _hard_rejected_row(base, reason="missing_audio")

    try:
        y, sr = load_audio(clip_path)
    except Exception:
        return _hard_rejected_row(base, reason="decode_failed")

    if not np.isfinite(y).all():
        return _hard_rejected_row(base, reason="nan_inf_audio")

    pipeline_for_pass1 = resolve_preprocess_pass1_pipeline(config)
    if config.early_audio_gate.enabled:
        tsr = int(pipeline_for_pass1.target_sample_rate)
        y_chk = (
            resample_audio(np.asarray(y, dtype=np.float32), sr, tsr)
            if sr != tsr
            else np.asarray(y, dtype=np.float32)
        )
        eg = run_early_audio_gate(
            y_chk,
            tsr,
            text_len=len(text_norm),
            mora_count=clip_mora_count if mora_early else None,
            main_gate=config.quality_gate,
            snr_cfg=config.snr,
            early=config.early_audio_gate,
        )
        if not eg.ok:
            return _hard_rejected_row(base, reason=eg.reason or "early_gate")

    try:
        y, sr, ameta = run_steps_on_array(y, sr, pipeline_for_pass1)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
        if len(err) > 220:
            err = err[:217] + "..."
        return _hard_rejected_row(base, reason=f"audio_pipeline_failed: {err}")

    if ameta.get("trim_exceeds_max_keep_sec"):
        return _hard_rejected_row(base, reason="trim_exceeds_max_keep_sec")

    pending = PendingClip(
        row=row,
        y=y,
        sr=sr,
        text_raw=text_raw,
        text_norm=text_norm,
        phonemes=phonemes,
        excerpt=text_raw[:80] if text_raw else "",
        ameta=ameta,
        mora_count=clip_mora_count if mora_fin else None,
    )
    if config.two_pass_denoise.enabled:
        pending = finalize_two_pass_denoise(pending, config)
        y, sr = pending.y, pending.sr

    final_qg = effective_final_quality_gate(config)
    gate = run_quality_gate(
        y,
        sr,
        text_len=len(text_norm),
        gate=final_qg,
        snr_cfg=config.snr,
        mora_count=clip_mora_count if mora_fin else None,
    )
    if not gate.ok:
        return _hard_rejected_row(base, reason=gate.reason or "gate")

    work_dir = config.dataset_builder.work_dir
    cache_abs = cached_wav_path(work_dir, pipeline_hash, audio_sha256)
    write_wav_atomic(cache_abs, y, sr)
    cache_rel = cache_abs.relative_to(work_dir).as_posix()

    out = dict(base)
    out["disposition"] = ClipDisposition.ELIGIBLE.value
    out["reject_reason"] = None
    out["duration_sec"] = gate.duration_sec
    out["quality_score"] = gate.quality_score
    out["estimated_snr_db"] = gate.estimated_snr_db
    out["silence_ratio"] = gate.silence_ratio
    out["audio_cache_rel_path"] = cache_rel
    enrich_row_with_linguistic_features(
        out,
        text_norm=text_norm,
        phonemes=phonemes,
        duration_sec=gate.duration_sec,
        config=config,
    )
    return ClipAnalyzeOutcome(
        row=out,
        disposition=ClipDisposition.ELIGIBLE,
        processed_y=y,
        processed_sr=sr,
        audio_sha256=audio_sha256,
    )


def analyze_project(
    config: PipelineConfig,
    *,
    progress: ProgressSink | None = None,
    cancellation: CancellationToken | None = None,
) -> AnalyzeResult:
    if not config.dataset_builder.enabled:
        raise ValueError("dataset_builder.enabled must be true to run analyze_project")

    warnings: list[str] = []
    if config.mfa_gate.enabled or config.nfa_gate.enabled or config.asr_gate.enabled:
        warnings.append("alignment_status=skipped (MFA/NFA/ASR gates not run in analyze v1)")
    if not _linguistic_module_available():
        warnings.append("linguistic features unavailable; phonemes from G2P only")

    enhance = config.audio_pipeline_enhance
    two_pass = config.two_pass_denoise.enabled
    enhance_types = (
        {str(getattr(step, "type", "")) for step in enhance.steps} if enhance is not None else set()
    )
    has_tts_restore = bool(enhance_types & {"sidon_restore", "denoise", "bandwidth_extension"})
    if not two_pass or enhance is None or not has_tts_restore:
        warnings.append(
            "TTS enhance missing: set two_pass_denoise.enabled=true and "
            "audio_pipeline_enhance with sidon_restore (see config/example.yaml). "
            "Without this, audio_cache/wavs are near-raw decode only."
        )
    if "sidon_restore" in enhance_types:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "audio_pipeline_enhance uses sidon_restore but torch is not installed. "
                "Run: uv sync --extra sidon"
            ) from None

    root = config.input.corpus_root
    loaded = load_clip_rows_for_pipeline(
        config,
        apply_input_max_clips=False,
        apply_speaker_merge=True,
        sort_by_path=False,
    )
    indexed_rows = _sorted_clip_rows(loaded.rows)
    total = len(indexed_rows)
    pipeline_hash = pipeline_cache_key(config)
    source_release = infer_release(root)
    lang = config.input.locale_expected or "ja"

    catalog_rows: list[dict[str, object]] = []
    eligible_count = 0
    hard_rejected_count = 0

    for current, (source_row_index, row) in enumerate(indexed_rows, start=1):
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if progress is not None:
            progress(
                ProgressEvent(
                    stage="analyze",
                    message=row.path,
                    current=current,
                    total=total,
                    fraction=current / total if total else 1.0,
                )
            )
        outcome = analyze_clip_with_gates(
            row,
            config=config,
            source_row_index=source_row_index,
            pipeline_hash=pipeline_hash,
            source_release=source_release,
            root=root,
            lang=lang,
        )
        catalog_rows.append(outcome.row)
        if outcome.disposition == ClipDisposition.ELIGIBLE:
            eligible_count += 1
        else:
            hard_rejected_count += 1

    manifest = {
        "schema_version": config.schema_version,
        "pipeline_hash": pipeline_hash,
        "alignment_status": "skipped",
        "warnings": warnings,
        "eligible_count": eligible_count,
        "hard_rejected_count": hard_rejected_count,
        "total_clips": total,
        "linguistic_module_available": _linguistic_module_available(),
    }
    clips_df = pl.DataFrame(catalog_rows)
    compute = resolve_compute_backend(config.compute.backend)
    feature_counts_df = compute.count_features(clips_df)
    speaker_stats_df = build_speaker_stats(clips_df)
    duplicate_groups_df = compute.build_duplicate_groups(clips_df)
    catalog = write_catalog_bundle(
        config.dataset_builder.work_dir,
        clips_df,
        feature_counts_df=feature_counts_df,
        speaker_stats_df=speaker_stats_df,
        duplicate_groups_df=duplicate_groups_df,
        manifest=manifest,
    )
    return AnalyzeResult(
        catalog=catalog,
        eligible_count=eligible_count,
        hard_rejected_count=hard_rejected_count,
        warnings=warnings,
    )
