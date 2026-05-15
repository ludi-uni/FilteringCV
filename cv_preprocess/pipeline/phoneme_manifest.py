"""OpenJTalk G2P 互換の音素照合マニフェスト（JSONL）を生成する。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import ClipRow, load_clip_rows_for_pipeline
from cv_preprocess.pipeline.export import write_json_report
from cv_preprocess.pipeline.g2p_map_suggest_core import (
    try_load_mfa_textgrid_phone_tokens,
    try_normalize_and_g2p_tokens,
    try_normalize_clip_text,
)
from cv_preprocess.text.mfa_token_map import load_mfa_token_map_yaml, map_mfa_space_separated_to_g2p_tokens


def new_phoneme_manifest_counts() -> dict[str, int]:
    return {
        "rows_written": 0,
        "skipped_text_locale": 0,
        "skipped_text_not_japanese": 0,
        "skipped_text_length": 0,
        "skipped_phonemize_failed": 0,
        "skipped_missing_textgrid": 0,
        "skipped_empty_phonemes": 0,
    }


def _resolve_phoneme_manifest_settings(
    cfg: PipelineConfig,
    *,
    output_path: Path | None,
    source: str | None,
    mfa_textgrid_root: Path | None,
    mfa_token_map_path: Path | None,
) -> tuple[Path, str, Path | None, Path | None]:
    pm = cfg.phoneme_manifest
    out = output_path
    if out is None and pm is not None:
        out = pm.output_path
    if out is None:
        raise ValueError(
            "音素マニフェストの出力先が未指定です。"
            " ``--output`` か config の ``phoneme_manifest.output_path`` を設定してください。"
        )
    src = (source or (pm.source if pm else "g2p_text")).strip().lower()
    if src not in ("g2p_text", "mfa_textgrid"):
        raise ValueError(f"source must be g2p_text or mfa_textgrid, got {src!r}")
    root = mfa_textgrid_root
    if root is None and pm is not None:
        root = pm.mfa_textgrid_root
    mpath = mfa_token_map_path
    if mpath is None and pm is not None:
        mpath = pm.mfa_token_map_path
    return out, src, root, mpath


def run_phoneme_manifest(
    cfg: PipelineConfig,
    *,
    output_path: Path | None = None,
    source: str | None = None,
    mfa_textgrid_root: Path | None = None,
    mfa_token_map_path: Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    ``validated.tsv``（設定の ``clip_tsv``）を走査し、``phoneme_alignment_check`` 用 JSONL を書く。

    * ``g2p_text``: preprocess と同じ ``normalize_for_tts`` + ``g2p_phonemes``（OpenJTalk）。
    * ``mfa_textgrid``: ``{mfa_textgrid_root}/{Path(path).stem}.TextGrid`` から phones を読み、
      ``mfa_token_map_path`` の YAML（MFA トークン → OJ 側の空白区切り列）で変換。マップに無い MFA トークンはそのまま出力。
    """
    out_path, src, mfa_root, map_path = _resolve_phoneme_manifest_settings(
        cfg,
        output_path=output_path,
        source=source,
        mfa_textgrid_root=mfa_textgrid_root,
        mfa_token_map_path=mfa_token_map_path,
    )

    if src == "mfa_textgrid" and mfa_root is None:
        raise ValueError("source=mfa_textgrid のときは mfa_textgrid_root（または config.phoneme_manifest）が必要です。")

    token_map = load_mfa_token_map_yaml(map_path)
    warnings: list[str] = []
    if src == "mfa_textgrid" and not token_map:
        warnings.append(
            "mfa_token_map_path が未指定です。MFA の phones ラベルをそのまま出力します。"
            "記号体系が OpenJTalk G2P と異なる場合は preprocess の照合で不一致になりやすいです。"
        )

    loaded = load_clip_rows_for_pipeline(cfg)
    rows = loaded.rows
    load_stats = loaded.load_stats
    root = cfg.input.corpus_root
    tsv_path = root / cfg.input.clip_tsv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    counts = new_phoneme_manifest_counts()

    if show_progress:
        print(
            f"[cv-preprocess] phoneme-manifest: source={src} clips={len(rows)} out={out_path}",
            file=sys.stderr,
            flush=True,
        )

    row_iter: list[ClipRow] | tqdm = rows
    if show_progress and rows:
        row_iter = tqdm(rows, desc="phoneme-manifest", unit="clip", file=sys.stderr, dynamic_ncols=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for row in row_iter:
            if src == "g2p_text":
                if not cfg.text.phonemize:
                    raise ValueError("phoneme-manifest g2p_text には text.phonemize=true が必要です。")
                text_g2p = try_normalize_and_g2p_tokens(row, cfg, counts)
                if text_g2p is None:
                    continue
                _text_norm, g2p_toks = text_g2p
                phonemes = " ".join(g2p_toks)
            else:
                assert mfa_root is not None
                if try_normalize_clip_text(row, cfg, counts) is None:
                    continue
                mfa_toks = try_load_mfa_textgrid_phone_tokens(
                    row.path,
                    mfa_root,
                    counts,
                    empty_count_key="skipped_empty_phonemes",
                )
                if mfa_toks is None:
                    continue
                phonemes = map_mfa_space_separated_to_g2p_tokens(" ".join(mfa_toks), token_map)
                if not phonemes.strip():
                    counts["skipped_empty_phonemes"] += 1
                    continue

            rec = {"source_path": row.path, "phonemes": phonemes.strip()}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["rows_written"] += 1

    report_path = out_path.parent / f"{out_path.stem}_report.json"
    report: dict[str, Any] = {
        "report_schema_version": 1,
        "stage": "phoneme_manifest",
        "source": src,
        "output_path": str(out_path.resolve()),
        "corpus_root": str(root.resolve()),
        "clip_tsv": str(tsv_path.resolve()),
        "load_stats": load_stats,
        "rows_after_filters": len(rows),
        "counts": counts,
        "warnings": warnings,
        "mfa_textgrid_root": str(mfa_root.resolve()) if mfa_root else None,
        "mfa_token_map_path": str(map_path.resolve()) if map_path else None,
        "mfa_token_map_size": len(token_map),
        "text_g2p_kana": cfg.text.g2p_kana,
    }
    write_json_report(report_path, report)
    report["report_path"] = str(report_path.resolve())
    return report
