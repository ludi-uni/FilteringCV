"""MFA/NFA → G2P トークンマップ草案で共有する Strategy・ペアリング・テキスト前処理・成果物出力。"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml

from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import ClipRow
from cv_preprocess.pipeline.export import write_json_report
from cv_preprocess.pipeline.token_map_suggest import (
    finalize_g2p_token_suggestions,
    merge_suggestion_dict_into_base,
)
from cv_preprocess.text.language_detect import passes_japanese_policy, passes_locale_expected
from cv_preprocess.text.normalize import normalize_for_tts
from cv_preprocess.text.phonemize import g2p_phonemes

Strategy = Literal["adaptive", "zip_only", "proportional_only"]
ReportSourceKey = Literal["mfa", "nfa"]

def _g2p_tokens(text_norm: str, *, kana: bool) -> list[str]:
    s = g2p_phonemes(text_norm, kana=kana).replace("\t", " ").strip()
    return [x for x in s.split() if x]


def _pairs_proportional(align: list[str], g2p: list[str]) -> list[tuple[str, str]]:
    if not align or not g2p:
        return []
    n, m = len(align), len(g2p)
    out: list[tuple[str, str]] = []
    denom = 2 * n
    for i, at in enumerate(align):
        j = min(m - 1, max(0, ((i * 2 + 1) * m) // denom))
        out.append((at, g2p[j]))
    return out


def _pairs_zip(align: list[str], g2p: list[str]) -> list[tuple[str, str]] | None:
    if len(align) != len(g2p):
        return None
    return list(zip(align, g2p, strict=True))


def _collect_votes_for_row(
    align_tokens: list[str],
    g2p: list[str],
    strategy: Strategy,
) -> tuple[list[tuple[str, str]], str]:
    """Returns (pairs, method_used: zip|proportional|none)."""
    if strategy == "zip_only":
        z = _pairs_zip(align_tokens, g2p)
        return (z, "zip") if z is not None else ([], "none")
    if strategy == "proportional_only":
        return _pairs_proportional(align_tokens, g2p), "proportional"
    z = _pairs_zip(align_tokens, g2p)
    if z is not None:
        return z, "zip"
    return _pairs_proportional(align_tokens, g2p), "proportional"


def try_normalize_clip_text(
    row: ClipRow,
    cfg: PipelineConfig,
    counts: dict[str, int],
) -> str | None:
    """ロケール・日本語・長さチェック後の ``text_norm``。スキップ時は ``counts`` を増やして ``None``。"""
    if not passes_locale_expected(row.locale, cfg.input.locale_expected):
        counts["skipped_text_locale"] += 1
        return None
    text_norm = normalize_for_tts(row.sentence)
    if not passes_japanese_policy(text_norm, cfg.text.require_japanese):
        counts["skipped_text_not_japanese"] += 1
        return None
    tl = len(text_norm)
    if tl < cfg.text.min_text_len or tl > cfg.text.max_text_len:
        counts["skipped_text_length"] += 1
        return None
    return text_norm


def try_normalize_and_g2p_tokens(
    row: ClipRow,
    cfg: PipelineConfig,
    counts: dict[str, int],
) -> tuple[str, list[str]] | None:
    """ロケール・日本語・長さ・G2P を通過した ``(text_norm, g2p_tokens)`` を返す。スキップ時は ``counts`` を増やして ``None``。"""
    text_norm = try_normalize_clip_text(row, cfg, counts)
    if text_norm is None:
        return None
    try:
        g2p_toks = _g2p_tokens(text_norm, kana=cfg.text.g2p_kana)
    except Exception:
        counts["skipped_phonemize_failed"] += 1
        return None
    if not g2p_toks:
        counts["skipped_empty_g2p"] += 1
        return None
    return text_norm, g2p_toks


def try_load_mfa_textgrid_phone_tokens(
    clip_path: str,
    mfa_textgrid_root: Path,
    counts: dict[str, int],
    *,
    missing_count_key: str = "skipped_missing_textgrid",
    empty_count_key: str = "skipped_empty_mfa",
) -> list[str] | None:
    """``{mfa_textgrid_root}/{stem}.TextGrid`` から phones トークン列を読む。"""
    tg_path = mfa_textgrid_root / f"{Path(clip_path).stem}.TextGrid"
    if not tg_path.is_file():
        counts[missing_count_key] += 1
        return None
    from cv_preprocess.audio.textgrid_phones import extract_phone_tokens_from_textgrid

    try:
        tokens = extract_phone_tokens_from_textgrid(tg_path)
    except OSError:
        counts[missing_count_key] += 1
        return None
    if not tokens:
        counts[empty_count_key] += 1
        return None
    return tokens


def accumulate_pairs_vote(
    align_tokens: list[str],
    g2p_tokens: list[str],
    strategy: Strategy,
    per_source: defaultdict[str, Counter[str]],
    method_counts: dict[str, int],
    counts: dict[str, int],
) -> None:
    """1 クリップ分の (アライナトークン, G2P) ペアを投票表に加える（``rows_considered`` / ``pairs_collected`` を更新）。"""
    counts["rows_considered"] += 1
    pairs, method = _collect_votes_for_row(align_tokens, g2p_tokens, strategy)
    method_counts[method] += 1
    for a, g in pairs:
        per_source[a][g] += 1
        counts["pairs_collected"] += 1


def new_mfa_g2p_map_suggest_counts() -> dict[str, int]:
    return {
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


def new_method_counts() -> dict[str, int]:
    return {"zip": 0, "proportional": 0, "none": 0}


def finalize_and_write_g2p_map_draft(
    *,
    per_source: defaultdict[str, Counter[str]],
    base_map: dict[str, str],
    output_yaml: Path,
    yaml_header: str,
    report_source_key: ReportSourceKey,
    stage: str,
    strategy: Strategy,
    min_votes: int,
    min_ratio: float,
    fill_missing_keys_only: bool,
    existing_map_path: Path | None,
    counts: dict[str, int],
    method_counts: dict[str, int],
    load_stats: dict[str, int],
    rows_after_filters: int,
    clip_tsv: Path,
    extra_report_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """投票表から YAML 草案と ``*_report.json`` を書き、レポート dict を返す。"""
    suggested, ambiguous, skipped_weak = finalize_g2p_token_suggestions(
        per_source,
        min_votes=min_votes,
        min_ratio=min_ratio,
        report_source_key=report_source_key,
    )
    out_map, added, skipped_existing, overwritten = merge_suggestion_dict_into_base(
        base_map,
        suggested,
        fill_missing_keys_only=fill_missing_keys_only,
    )

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(out_map, allow_unicode=True, default_flow_style=False, sort_keys=True)
    output_yaml.write_text(yaml_header + body, encoding="utf-8")

    report_path = output_yaml.parent / f"{output_yaml.stem}_report.json"
    report: dict[str, Any] = {
        "report_schema_version": 1,
        "stage": stage,
        "strategy": strategy,
        "output_yaml": str(output_yaml.resolve()),
        "clip_tsv": str(clip_tsv.resolve()),
        "load_stats": load_stats,
        "rows_after_filters": rows_after_filters,
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
        f"ambiguous_{report_source_key}": ambiguous,
        "skipped_low_confidence": skipped_weak,
        f"unique_{report_source_key}_in_votes": len(per_source),
    }
    if extra_report_fields:
        report.update(extra_report_fields)
    write_json_report(report_path, report)
    report["report_path"] = str(report_path.resolve())
    return report
