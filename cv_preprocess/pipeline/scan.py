from __future__ import annotations

from collections import Counter
from pathlib import Path

from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import filter_by_clip_metadata, filter_by_speakers, load_validated_tsv


def _first_n_unique_client_ids(rows: list, n: int = 15) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        cid = r.client_id
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
            if len(out) >= n:
                break
    return out


def _physical_line_count(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def scan_corpus(cfg: PipelineConfig) -> dict:
    root = cfg.input.corpus_root
    tsv = root / cfg.input.clip_tsv
    rows, stats = load_validated_tsv(tsv)
    include = cfg.speakers.include_client_ids or None
    filtered = filter_by_speakers(rows, include)
    filtered = filter_by_clip_metadata(filtered, cfg.speakers.clip_metadata_filters)
    unique_client_ids_after_filters = len({r.client_id.strip() for r in filtered})
    unique_client_ids_effective = (
        1
        if cfg.speakers.merge_filtered_speakers_as_one and filtered
        else unique_client_ids_after_filters
    )
    clients = Counter(r.client_id for r in rows)
    missing_audio: list[str] = []
    clips = root / cfg.input.audio_subdir
    sample_rows = filtered
    for r in sample_rows[: min(5000, len(sample_rows))]:
        p = clips / r.path
        if not p.is_file():
            missing_audio.append(r.path)

    sample_ids = _first_n_unique_client_ids(rows, 15)

    warnings: list[str] = []
    if include and len(filter_by_speakers(rows, include)) == 0:
        warnings.append(
            "speakers.include_client_ids に一致する行が 0 件です。"
            " 別リリースでは client_id が変わります。"
            " また、validated.tsv に改行を含むセルがあると、grep の行と CSV パーサの論理行がずれ、"
            " ファイル上で先頭列に見えるハッシュでも client_id としては存在しないことがあります。"
            " 下の sample_client_ids_from_parsed_tsv を設定に使うか、include_client_ids: [] で全話者にしてください。"
        )
    if cfg.speakers.clip_metadata_filters.is_active() and len(filtered) == 0 and len(rows) > 0:
        warnings.append(
            "speakers.clip_metadata_filters 適用後に 0 件です。"
            " 各軸の許容リストに明示的に \"\" を含めない限り、TSV でその列が空欄の行は除外されます。"
        )

    if stats["rows_ok"] > 0 and tsv.is_file():
        physical = _physical_line_count(tsv)
        # ヘッダ1行を除いた物理行が論理行より多い → クォート内改行等で1レコードが複数物理行
        if physical - 1 > stats["rows_ok"]:
            warnings.append(
                "validated.tsv の物理行数が論理行（CSV として解釈した行）より大きいです。"
                " 文中のダブルクォート（\"）に伴うクォートフィールド内改行で、1 レコードが複数行にまたがっている可能性があります。"
                " Excel の行番号や、行単位の grep / 検索で先頭列に見える文字列でも、パーサ上は client_id 列ではないことがあります。"
                " 話者 ID の有無は本ツールの load_validated_tsv（csv モジュール）の結果を基準にしてください。"
            )

    return {
        "tsv_path": str(tsv),
        "stats": stats,
        "rows_after_speaker_filter": len(filter_by_speakers(rows, include)),
        "rows_after_clip_metadata_filter": len(filtered),
        "merge_filtered_speakers_as_one": cfg.speakers.merge_filtered_speakers_as_one,
        "merged_speaker_client_id_effective": (
            cfg.speakers.resolved_merged_speaker_client_id()
            if cfg.speakers.merge_filtered_speakers_as_one
            else None
        ),
        "unique_client_ids_after_filters": unique_client_ids_after_filters,
        "unique_client_ids_effective": unique_client_ids_effective,
        "clip_metadata_filters": cfg.speakers.clip_metadata_filters.model_dump(),
        "speaker_filter_list_size": len(include) if include else 0,
        "unique_client_ids": len(clients),
        "sample_client_ids_from_parsed_tsv": sample_ids,
        "warnings": warnings,
        "sample_missing_audio_first10": missing_audio[:10],
        "total_missing_audio_sampled": len(missing_audio),
    }
