"""MFA phones トークン列を OpenJTalk G2P 互換の空白区切り列へ変換する（YAML 辞書）。"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_mfa_token_map_yaml(path: Path | None) -> dict[str, str]:
    """YAML 1 オブジェクト（辞書）を読み、キー・値を文字列に正規化する。値が空ならその MFA トークンはマッピング結果から省略される。"""
    if path is None:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"MFA token map YAML must be a mapping, got {type(raw)}")
    out: dict[str, str] = {}
    for k, v in raw.items():
        out[str(k).strip()] = "" if v is None else str(v).strip()
    return out


def map_mfa_space_separated_to_g2p_tokens(mfa_phone_string: str, token_map: dict[str, str]) -> str:
    """
    MFA の phones 層から得た空白区切り（または単一トークン）を、OpenJTalk G2P 側の空白区切りに変換する。

    * ``token_map`` にキーがある場合: 値を空白で分割して連結（値が空なら当該 MFA トークンは出力しない）。
    * キーが無い場合: MFA トークンをそのまま 1 トークンとして出力（従来どおりの素通し）。
    """
    parts: list[str] = []
    for t in mfa_phone_string.replace("\t", " ").split():
        key = t.strip()
        if not key:
            continue
        if key in token_map:
            mapped = token_map[key].strip()
            if mapped:
                parts.extend(mapped.split())
            continue
        parts.append(key)
    return " ".join(parts)
