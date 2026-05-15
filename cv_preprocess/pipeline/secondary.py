from __future__ import annotations

import json
import sys
import traceback
from typing import Any

import numpy as np
from tqdm import tqdm

from cv_preprocess.audio.decode import load_audio
from cv_preprocess.audio.pipeline import run_steps_on_array
from cv_preprocess.audio.quality_gate import run_quality_gate
from cv_preprocess.config import PipelineConfig, QualityGateConfig
from cv_preprocess.pipeline.export import append_jsonl, write_json_report, write_reject_row, write_wav_16bit
from cv_preprocess.pipeline.ljspeech_tsv import write_ljspeech_validated_tsv
from cv_preprocess.text.mora_estimate import mora_count_for_text


def _resolve_secondary_quality_gate(cfg: PipelineConfig) -> QualityGateConfig:
    sec = cfg.secondary
    assert sec is not None
    merged: dict[str, Any] = dict(cfg.quality_gate.model_dump())
    merged.update(sec.quality_gate_overrides)
    prof = sec.quality_gate_profile
    if prof:
        if prof not in cfg.quality_gate_profiles:
            raise ValueError(
                f"secondary.quality_gate_profile {prof!r} not in quality_gate_profiles "
                f"(keys: {list(cfg.quality_gate_profiles)})"
            )
        p = cfg.quality_gate_profiles[prof]
        if not isinstance(p, dict):
            raise ValueError(f"quality_gate_profiles[{prof!r}] must be a mapping")
        merged.update(p)
    return QualityGateConfig.model_validate(merged)


def _primary_quality_snapshot(rec: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "duration_sec",
        "silence_ratio",
        "estimated_snr_db",
        "quality_tier",
        "quality_score",
        "trailing_silence_sec",
        "mora_count",
        "min_required_duration_sec",
    )
    return {k: rec.get(k) for k in keys}


def _secondary_quality_dict(gate: Any) -> dict[str, Any]:
    return {
        "duration_sec": gate.duration_sec,
        "silence_ratio": gate.silence_ratio,
        "estimated_snr_db": gate.estimated_snr_db,
        "quality_tier": gate.quality_tier,
        "quality_score": gate.quality_score,
        "trailing_silence_sec": gate.trailing_silence_sec,
        "mora_count": gate.mora_count,
        "min_required_duration_sec": gate.min_required_duration_sec,
        "gate_ok": gate.ok,
        "gate_reason": gate.reason,
    }


def run_secondary(cfg: PipelineConfig, *, show_progress: bool = True) -> dict[str, Any]:
    """一次 preprocess の出力を読み、二次音声チェーンと再品質ゲートを適用する。"""
    sec = cfg.secondary
    if sec is None:
        raise ValueError("config.secondary が未設定です。YAML に secondary: ブロックを追加してください。")

    in_root = sec.input_root or cfg.output.root
    manifest_path = sec.input_manifest or (in_root / cfg.output.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"secondary 入力 manifest が見つかりません: {manifest_path}")

    out_root = sec.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    wav_dir = out_root / sec.wav_subdir
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = out_root / cfg.output.manifest
    out_validated = out_root / cfg.output.validated_tsv
    rejects_path = out_root / sec.rejects_name
    report_path = out_root / sec.report_name

    for p in (out_manifest, out_validated, rejects_path):
        if p.exists():
            p.unlink()

    for stale in wav_dir.glob("*.wav"):
        stale.unlink(missing_ok=True)

    qg = _resolve_secondary_quality_gate(cfg)
    snr_cfg = sec.snr or cfg.snr
    lang = (cfg.input.locale_expected or "ja").split("-")[0]

    reject_fields = ["source_path", "client_id", "reason", "sentence_excerpt"]
    reject_reasons: dict[str, int] = {}
    accepted: list[dict[str, Any]] = []

    rows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if show_progress:
        print(
            f"[cv-preprocess] secondary: clips={len(rows)} in={in_root} out={out_root}",
            file=sys.stderr,
            flush=True,
        )

    audio_err_logged = False
    row_iter: list[dict[str, Any]] | tqdm = rows
    if show_progress and rows:
        row_iter = tqdm(rows, desc="secondary", unit="clip", file=sys.stderr, dynamic_ncols=True)

    for rec in row_iter:
        excerpt = str(rec.get("text_norm", ""))[:80]
        client_id = str(rec.get("speaker_id", ""))
        source_path = str(rec.get("source_path", rec.get("audio_path", "")))
        utt_id = str(rec.get("utt_id", ""))
        rel_audio = str(rec.get("audio_path", ""))
        if not rel_audio or not utt_id:
            write_reject_row(
                rejects_path,
                {
                    "source_path": source_path,
                    "client_id": client_id,
                    "reason": "secondary_missing_audio_path_or_utt",
                    "sentence_excerpt": excerpt,
                },
                reject_fields,
            )
            reject_reasons["secondary_missing_audio_path_or_utt"] = (
                reject_reasons.get("secondary_missing_audio_path_or_utt", 0) + 1
            )
            continue

        wav_in = in_root / rel_audio
        if not wav_in.is_file():
            write_reject_row(
                rejects_path,
                {
                    "source_path": source_path,
                    "client_id": client_id,
                    "reason": "secondary_missing_wav",
                    "sentence_excerpt": excerpt,
                },
                reject_fields,
            )
            reject_reasons["secondary_missing_wav"] = reject_reasons.get("secondary_missing_wav", 0) + 1
            continue

        try:
            y, sr = load_audio(wav_in)
        except Exception:
            write_reject_row(
                rejects_path,
                {
                    "source_path": source_path,
                    "client_id": client_id,
                    "reason": "secondary_decode_failed",
                    "sentence_excerpt": excerpt,
                },
                reject_fields,
            )
            reject_reasons["secondary_decode_failed"] = reject_reasons.get("secondary_decode_failed", 0) + 1
            continue

        if not np.isfinite(y).all():
            write_reject_row(
                rejects_path,
                {
                    "source_path": source_path,
                    "client_id": client_id,
                    "reason": "secondary_nan_inf_audio",
                    "sentence_excerpt": excerpt,
                },
                reject_fields,
            )
            reject_reasons["secondary_nan_inf_audio"] = reject_reasons.get("secondary_nan_inf_audio", 0) + 1
            continue

        pq = _primary_quality_snapshot(rec)

        try:
            y, sr, ameta = run_steps_on_array(y, sr, sec.audio_pipeline)
        except Exception as e:
            err_one_line = f"{type(e).__name__}: {e}".replace("\n", " ").strip()
            if len(err_one_line) > 220:
                err_one_line = err_one_line[:217] + "..."
            if not audio_err_logged:
                audio_err_logged = True
                print(
                    "[cv-preprocess] secondary 音声チェーンで例外（以降は要約のみ）:",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exception(type(e), e, e.__traceback__, limit=12, file=sys.stderr)
            write_reject_row(
                rejects_path,
                {
                    "source_path": source_path,
                    "client_id": client_id,
                    "reason": f"secondary_audio_pipeline_failed: {err_one_line}",
                    "sentence_excerpt": excerpt,
                },
                reject_fields,
            )
            reject_reasons["secondary_audio_pipeline_failed"] = (
                reject_reasons.get("secondary_audio_pipeline_failed", 0) + 1
            )
            continue

        if not np.isfinite(y).all():
            write_reject_row(
                rejects_path,
                {
                    "source_path": source_path,
                    "client_id": client_id,
                    "reason": "secondary_nan_inf_audio_after_pipeline",
                    "sentence_excerpt": excerpt,
                },
                reject_fields,
            )
            reject_reasons["secondary_nan_inf_audio_after_pipeline"] = (
                reject_reasons.get("secondary_nan_inf_audio_after_pipeline", 0) + 1
            )
            continue

        text_norm = str(rec.get("text_norm", ""))
        mora_n: int | None = None
        if qg.min_sec_per_mora is not None and lang == "ja":
            try:
                mora_n = mora_count_for_text(text_norm)
            except Exception:
                write_reject_row(
                    rejects_path,
                    {
                        "source_path": source_path,
                        "client_id": client_id,
                        "reason": "secondary_mora_estimate_failed",
                        "sentence_excerpt": excerpt,
                    },
                    reject_fields,
                )
                reject_reasons["secondary_mora_estimate_failed"] = (
                    reject_reasons.get("secondary_mora_estimate_failed", 0) + 1
                )
                continue

        gate = run_quality_gate(
            y,
            sr,
            text_len=len(text_norm),
            gate=qg,
            snr_cfg=snr_cfg,
            mora_count=mora_n,
        )
        if not gate.ok:
            write_reject_row(
                rejects_path,
                {
                    "source_path": source_path,
                    "client_id": client_id,
                    "reason": f"secondary_{gate.reason or 'gate'}",
                    "sentence_excerpt": excerpt,
                },
                reject_fields,
            )
            key = f"secondary_{gate.reason or 'gate'}"
            reject_reasons[key] = reject_reasons.get(key, 0) + 1
            continue

        rel_wav = f"{sec.wav_subdir}/{utt_id}.wav"
        wav_path = out_root / rel_wav
        bit_depth = 16
        for st in sec.audio_pipeline.steps:
            if st.type == "save_wav":
                bit_depth = st.bit_depth
        if bit_depth == 16:
            write_wav_16bit(wav_path, y, sr)
        else:
            write_wav_16bit(wav_path, y, sr)

        out_rec = dict(rec)
        out_rec["primary_quality"] = pq
        out_rec["secondary_corrections"] = ameta.get("steps_trace", [])
        out_rec["secondary_quality"] = _secondary_quality_dict(gate)
        out_rec["audio_path"] = rel_wav.replace("\\", "/")
        out_rec["duration_sec"] = gate.duration_sec
        out_rec["silence_ratio"] = gate.silence_ratio
        out_rec["estimated_snr_db"] = gate.estimated_snr_db
        out_rec["quality_score"] = gate.quality_score
        out_rec["quality_tier"] = gate.quality_tier
        out_rec["trailing_silence_sec"] = gate.trailing_silence_sec
        out_rec["mora_count"] = gate.mora_count
        out_rec["min_required_duration_sec"] = gate.min_required_duration_sec
        out_rec["secondary_audio_pipeline_id"] = sec.audio_pipeline.audio_pipeline_id
        if "edge_removed_leading_ms" in ameta:
            out_rec["edge_removed_leading_ms"] = ameta.get("edge_removed_leading_ms", 0.0)
            out_rec["edge_removed_trailing_ms"] = ameta.get("edge_removed_trailing_ms", 0.0)
            out_rec["edge_click_confidence"] = ameta.get("edge_click_confidence")
        accepted.append(out_rec)

    for r in accepted:
        append_jsonl(out_manifest, r)
    write_ljspeech_validated_tsv(out_validated, accepted)

    report: dict[str, Any] = {
        "report_schema_version": 1,
        "stage": "secondary",
        "input_manifest": str(manifest_path.resolve()),
        "input_root": str(in_root.resolve()),
        "output_root": str(out_root.resolve()),
        "output_manifest": str(out_manifest.resolve()),
        "secondary_audio_pipeline_id": sec.audio_pipeline.audio_pipeline_id,
        "accepted": len(accepted),
        "rejected_by_reason": reject_reasons,
        "quality_gate_profile": sec.quality_gate_profile,
        "secondary_quality_gate": qg.model_dump(),
    }
    write_json_report(report_path, report)
    return report
