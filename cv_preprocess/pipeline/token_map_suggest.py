"""MFA/NFA → G2P トークンマップ草案の共有ロジック（集計後の確定・YAML マージ）。"""

from __future__ import annotations

from collections import Counter
from typing import Any


def finalize_g2p_token_suggestions(
    counts_by_source_token: dict[str, Counter[str]],
    *,
    min_votes: int,
    min_ratio: float,
    report_source_key: str,
) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    """各ソース音素トークンの G2P 投票から、採用案・あいまい・低信頼スキップを決める。

    ``counts_by_source_token`` はソース音素トークン → G2P トークンごとの票数。
    ``report_source_key`` はレポート JSON 用（例: ``\"mfa\"`` / ``\"nfa\"``）。
    """
    suggested: dict[str, str] = {}
    ambiguous: list[dict[str, Any]] = []
    skipped_weak: list[dict[str, Any]] = []

    for tok, ctr in sorted(counts_by_source_token.items(), key=lambda x: x[0]):
        total = sum(ctr.values())
        top_two = ctr.most_common(2)
        best_tok, best_n = top_two[0]
        ratio = best_n / total if total else 0.0
        if best_n < min_votes or ratio < min_ratio:
            skipped_weak.append(
                {
                    report_source_key: tok,
                    "best_g2p": best_tok,
                    "votes": best_n,
                    "total": total,
                    "ratio": round(ratio, 4),
                }
            )
            continue
        second_n = top_two[1][1] if len(top_two) > 1 else 0
        if len(ctr) > 1 and second_n * 2 >= best_n:
            ambiguous.append(
                {
                    report_source_key: tok,
                    "top": ctr.most_common(5),
                    "total": total,
                }
            )
            continue
        suggested[tok] = best_tok

    return suggested, ambiguous, skipped_weak


def merge_suggestion_dict_into_base(
    base_map: dict[str, str],
    suggested: dict[str, str],
    *,
    fill_missing_keys_only: bool,
) -> tuple[dict[str, str], int, int, int]:
    """``suggested`` を ``base_map`` にマージする。戻り値: ``(out_map, added, skipped_existing, overwritten)``。"""
    out_map: dict[str, str] = dict(base_map)
    added = 0
    skipped_existing = 0
    overwritten = 0
    if fill_missing_keys_only:
        for k, v in suggested.items():
            if k in out_map:
                skipped_existing += 1
                continue
            out_map[k] = v
            added += 1
    else:
        for k, v in suggested.items():
            if k in out_map and out_map[k] != v:
                overwritten += 1
            elif k not in out_map:
                added += 1
            out_map[k] = v
    return out_map, added, skipped_existing, overwritten
