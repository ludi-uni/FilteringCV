"""MFA phones と OpenJTalk G2P から YAML マップの草案を集計する（人手レビュー前提）。"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml
from tqdm import tqdm

from cv_preprocess.audio.textgrid_phones import extract_phone_tokens_from_textgrid
from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import (
    apply_merge_filtered_speakers_as_one,
    filter_by_clip_metadata,
    filter_by_speakers,
    load_validated_tsv,
)
from cv_preprocess.text.language_detect import passes_japanese_policy, passes_locale_expected
from cv_preprocess.text.mfa_token_map import load_mfa_token_map_yaml
from cv_preprocess.text.normalize import normalize_for_tts
from cv_preprocess.pipeline.token_map_suggest import (
    finalize_g2p_token_suggestions,
    merge_suggestion_dict_into_base,
)
from cv_preprocess.text.phonemize import g2p_phonemes

Strategy = Literal["adaptive", "zip_only", "proportional_only"]


def _g2p_tokens(text_norm: str, *, kana: bool) -> list[str]:
    s = g2p_phonemes(text_norm, kana=kana).replace("\t", " ").strip()
    return [x for x in s.split() if x]


def _pairs_proportional(mfa: list[str], g2p: list[str]) -> list[tuple[str, str]]:
    if not mfa or not g2p:
        return []
    n, m = len(mfa), len(g2p)
    out: list[tuple[str, str]] = []
    denom = 2 * n
    for i, mt in enumerate(mfa):
        j = min(m - 1, max(0, ((i * 2 + 1) * m) // denom))
        out.append((mt, g2p[j]))
    return out


def _pairs_zip(mfa: list[str], g2p: list[str]) -> list[tuple[str, str]] | None:
    if len(mfa) != len(g2p):
        return None
    return list(zip(mfa, g2p, strict=True))


def _collect_votes_for_row(
    mfa: list[str],
    g2p: list[str],
    strategy: Strategy,
) -> tuple[list[tuple[str, str]], str]:
    """Returns (pairs, method_used: zip|proportional|none)."""
    if strategy == "zip_only":
        z = _pairs_zip(mfa, g2p)
        return (z, "zip") if z is not None else ([], "none")
    if strategy == "proportional_only":
        return _pairs_proportional(mfa, g2p), "proportional"
    z = _pairs_zip(mfa, g2p)
    if z is not None:
        return z, "zip"
    return _pairs_proportional(mfa, g2p), "proportional"


def run_mfa_g2p_map_suggest(
    cfg: PipelineConfig,
    *,
    mfa_textgrid_root: Path,
    output_yaml: Path,
    strategy: Strategy = "adaptive",
    min_votes: int = 2,
    min_ratio: float = 0.55,
    existing_map_path: Path | None = None,
    fill_missing_keys_only: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    ``validated.tsv`` を ``phoneme-manifest`` と同様に走査し、各行で MFA phones と G2P を並べ、
    (MFA トークン → G2P トークン) の投票を集計して YAML 草案を書く。

    * **adaptive**: 長さが一致する行は zip、それ以外は比例配置でペアを作る。
    * **zip_only** / **proportional_only**: 名前どおり。

    比例配置は記号粒度が異なるときの近似であり、**誤提案が混ざる**。必ずレポートと突き合わせて人手で直すこと。
    """
    if not cfg.text.phonemize:
        raise ValueError("suggest-mfa-g2p-map には text.phonemize=true が必要です。")
    if not mfa_textgrid_root.is_dir():
        raise ValueError(f"mfa_textgrid_root must be a directory: {mfa_textgrid_root}")

    root = cfg.input.corpus_root
    tsv_path = root / cfg.input.clip_tsv
    rows, load_stats = load_validated_tsv(tsv_path)
    include_ids = cfg.speakers.include_client_ids or None
    rows = filter_by_speakers(rows, include_ids)
    rows = filter_by_clip_metadata(rows, cfg.speakers.clip_metadata_filters)
    rows = sorted(rows, key=lambda r: r.path)
    apply_merge_filtered_speakers_as_one(
        rows,
        enabled=cfg.speakers.merge_filtered_speakers_as_one,
        merged_client_id=cfg.speakers.resolved_merged_speaker_client_id(),
    )

    base_map: dict[str, str] = {}
    if existing_map_path is not None:
        base_map = dict(load_mfa_token_map_yaml(existing_map_path))

    per_mfa: defaultdict[str, Counter[str]] = defaultdict(Counter)
    method_counts = {"zip": 0, "proportional": 0, "none": 0}
    counts: dict[str, int] = {
        "rows_considered": 0,
        "skipped_text_locale": 0,
        "skipped_text_not_japanese": 0,
        "skipped_text_length": 0,
        "skipped_phonemize_failed": 0,
        "skipped_missing_textgrid": 0,
        "skipped_empty_mfa": 0,
        "skipped_empty_g2p": 0,
        "pairs_collected": 0,
    }

    if show_progress:
        print(
            f"[cv-preprocess] suggest-mfa-g2p-map: clips={len(rows)} "
            f"tg_root={mfa_textgrid_root} strategy={strategy}",
            file=sys.stderr,
            flush=True,
        )

    row_iter = tqdm(rows, desc="suggest-mfa-g2p-map", unit="clip", file=sys.stderr, dynamic_ncols=True) if (
        show_progress and rows
    ) else rows

    for row in row_iter:
        if not passes_locale_expected(row.locale, cfg.input.locale_expected):
            counts["skipped_text_locale"] += 1
            continue
        text_norm = normalize_for_tts(row.sentence)
        if not passes_japanese_policy(text_norm, cfg.text.require_japanese):
            counts["skipped_text_not_japanese"] += 1
            continue
        tl = len(text_norm)
        if tl < cfg.text.min_text_len or tl > cfg.text.max_text_len:
            counts["skipped_text_length"] += 1
            continue

        stem = Path(row.path).stem
        tg_path = mfa_textgrid_root / f"{stem}.TextGrid"
        if not tg_path.is_file():
            counts["skipped_missing_textgrid"] += 1
            continue
        try:
            mfa_toks = extract_phone_tokens_from_textgrid(tg_path)
        except OSError:
            counts["skipped_missing_textgrid"] += 1
            continue
        if not mfa_toks:
            counts["skipped_empty_mfa"] += 1
            continue

        try:
            g2p_toks = _g2p_tokens(text_norm, kana=cfg.text.g2p_kana)
        except Exception:
            counts["skipped_phonemize_failed"] += 1
            continue
        if not g2p_toks:
            counts["skipped_empty_g2p"] += 1
            continue

        counts["rows_considered"] += 1
        pairs, method = _collect_votes_for_row(mfa_toks, g2p_toks, strategy)
        method_counts[method] += 1
        for mt, gt in pairs:
            per_mfa[mt][gt] += 1
            counts["pairs_collected"] += 1

    suggested, ambiguous, skipped_weak = finalize_g2p_token_suggestions(
        per_mfa,
        min_votes=min_votes,
        min_ratio=min_ratio,
        report_source_key="mfa",
    )

    out_map, added, skipped_existing, overwritten = merge_suggestion_dict_into_base(
        base_map,
        suggested,
        fill_missing_keys_only=fill_missing_keys_only,
    )

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# AUTO-GENERATED DRAFT — MFA phones → OpenJTalk G2P single-token hints.\n"
        "# Proportional alignment is approximate; review against docs/音素照合マニフェスト.md.\n"
        "# Empty value \"\" omits a token in map_mfa_space_separated_to_g2p_tokens.\n"
    )
    body = yaml.safe_dump(out_map, allow_unicode=True, default_flow_style=False, sort_keys=True)
    output_yaml.write_text(header + body, encoding="utf-8")

    report_path = output_yaml.parent / f"{output_yaml.stem}_report.json"
    report: dict[str, Any] = {
        "report_schema_version": 1,
        "stage": "mfa_g2p_map_suggest",
        "strategy": strategy,
        "mfa_textgrid_root": str(mfa_textgrid_root.resolve()),
        "output_yaml": str(output_yaml.resolve()),
        "clip_tsv": str(tsv_path.resolve()),
        "load_stats": load_stats,
        "rows_after_filters": len(rows),
        "counts": counts,
        "method_counts": method_counts,
        "min_votes": min_votes,
        "min_ratio": min_ratio,
        "existing_map_path": str(existing_map_path.resolve()) if existing_map_path else None,
        "fill_missing_keys_only": fill_missing_keys_only,
        "suggested_entries": len(suggested),
        "yaml_keys_written": len(out_map),
        "new_keys_merged": added,
        "suggestions_overwritten_existing": overwritten,
        "skipped_existing_key": skipped_existing,
        "ambiguous_mfa": ambiguous,
        "skipped_low_confidence": skipped_weak,
        "unique_mfa_in_votes": len(per_mfa),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    return report
