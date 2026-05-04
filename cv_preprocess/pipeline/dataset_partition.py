from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

from cv_preprocess.pipeline.ljspeech_tsv import write_ljspeech_validated_tsv

GroupBy = Literal["quality_tier", "split", "split_quality_tier"]


def _bucket_name(rec: dict[str, Any], group_by: GroupBy) -> str:
    if group_by == "quality_tier":
        t = rec.get("quality_tier")
        if t is None or str(t).strip() == "":
            return "unknown"
        return str(t).strip().upper()
    if group_by == "split":
        s = rec.get("split")
        if s is None or str(s).strip() == "":
            return "unsplit"
        return str(s).strip().lower()
    # split_quality_tier
    sp = rec.get("split")
    sp_s = str(sp).strip().lower() if sp is not None and str(sp).strip() else "unsplit"
    t = rec.get("quality_tier")
    t_s = str(t).strip().upper() if t is not None and str(t).strip() else "unknown"
    return f"{sp_s}__{t_s}"


def _passes_filters(
    rec: dict[str, Any],
    *,
    min_quality_score: float | None,
    max_quality_score: float | None,
    only_tiers: frozenset[str] | None,
) -> bool:
    sc = rec.get("quality_score")
    if min_quality_score is not None:
        if sc is None or float(sc) < float(min_quality_score):
            return False
    if max_quality_score is not None:
        if sc is None or float(sc) > float(max_quality_score):
            return False
    if only_tiers is not None:
        t = rec.get("quality_tier")
        t_s = str(t).strip().upper() if t is not None and str(t).strip() else "UNKNOWN"
        if t_s not in only_tiers:
            return False
    return True


def run_dataset_partition(
    *,
    metadata_path: Path,
    audio_root: Path,
    output_root: Path,
    group_by: GroupBy,
    min_quality_score: float | None = None,
    max_quality_score: float | None = None,
    only_tiers: list[str] | None = None,
    use_symlink: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """metadata.jsonl の各行について ``audio_path`` の WAV をバケット別ディレクトリへ集約する。

    各バケット ``{output_root}/{bucket}/`` に ``wavs/``・``metadata.jsonl``・``validated.tsv`` を出力する。
    ``metadata`` 内の ``audio_path`` はバケット内でも ``wavs/...`` のまま（相対パス不変）。
    """
    metadata_path = metadata_path.resolve()
    audio_root = audio_root.resolve()
    output_root = output_root.resolve()

    tier_filter: frozenset[str] | None = None
    if only_tiers:
        tier_filter = frozenset(x.strip().upper() for x in only_tiers if x.strip())

    by_bucket: dict[str, list[dict[str, Any]]] = {}
    skipped_filter = 0
    missing_wav = 0
    errors: list[str] = []

    with metadata_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not _passes_filters(
                rec,
                min_quality_score=min_quality_score,
                max_quality_score=max_quality_score,
                only_tiers=tier_filter,
            ):
                skipped_filter += 1
                continue
            ap = rec.get("audio_path")
            if not ap or not isinstance(ap, str):
                errors.append("missing_or_invalid_audio_path")
                continue
            rel = ap.replace("\\", "/").lstrip("/")
            src_wav = audio_root / rel
            if not src_wav.is_file():
                missing_wav += 1
                continue
            bucket = _bucket_name(rec, group_by)
            by_bucket.setdefault(bucket, []).append(rec)

    for bucket, rows in by_bucket.items():
        if dry_run:
            continue
        dest_dir = output_root / bucket
        wavs_dir = dest_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)
        for rec in rows:
            ap = str(rec.get("audio_path", "")).replace("\\", "/").lstrip("/")
            src = audio_root / ap
            name = Path(ap).name
            dst = wavs_dir / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if use_symlink:
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
        meta_out = dest_dir / "metadata.jsonl"
        tsv_out = dest_dir / "validated.tsv"
        with meta_out.open("w", encoding="utf-8") as mf:
            for rec in rows:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_ljspeech_validated_tsv(tsv_out, rows)

    counts = {k: len(v) for k, v in sorted(by_bucket.items())}
    return {
        "output_root": str(output_root),
        "group_by": group_by,
        "buckets": counts,
        "total_partitioned": sum(counts.values()),
        "skipped_by_filter": skipped_filter,
        "skipped_missing_wav": missing_wav,
        "use_symlink": use_symlink,
        "dry_run": dry_run,
        "errors_sample": errors[:5],
    }


def validate_group_by(value: str) -> GroupBy:
    v = value.strip().lower().replace("-", "_")
    if v == "quality_tier":
        return "quality_tier"
    if v == "split":
        return "split"
    if v == "split_quality_tier":
        return "split_quality_tier"
    raise ValueError(
        "group_by must be one of quality_tier, split, split_quality_tier "
        f"(got {value!r})"
    )
