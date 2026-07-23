from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import polars as pl

from cv_preprocess.application.common import MaterializeResult, ProgressEvent, ProgressSink, SelectionPlan
from cv_preprocess.catalog import CatalogRef
from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.catalog.reader import read_clips
from cv_preprocess.config import PipelineConfig


def resolve_materialize_output_root(config: PipelineConfig) -> Path:
    db = config.dataset_builder
    if db.materialize.output_root is not None:
        return Path(db.materialize.output_root)
    if db.output_dir is not None:
        return Path(db.output_dir)
    return Path(config.output.root)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial.replace(path)


def _place_audio(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "hardlink":
            os.link(src, dst)
        elif mode == "symlink":
            os.symlink(src.resolve(), dst)
        else:
            raise ValueError(f"unsupported materialize mode: {mode!r}")
    except OSError:
        shutil.copy2(src, dst)


def _metadata_record(
    *,
    clip_id: str,
    audio_path: str,
    row: dict[str, Any],
    split: str | None,
    selection_rank: int | None,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "utt_id": clip_id,
        "audio_path": audio_path,
        "text_raw": row.get("text_raw"),
        "text_norm": row.get("text_norm"),
        "phonemes": row.get("phonemes"),
        "speaker_id": row.get("speaker_id"),
        "sentence_id": row.get("sentence_id"),
        "duration_sec": row.get("duration_sec"),
        "quality_score": row.get("quality_score"),
        "estimated_snr_db": row.get("estimated_snr_db"),
        "silence_ratio": row.get("silence_ratio"),
        "split": split,
        "selection_rank": selection_rank,
        "source_release": row.get("source_release"),
        "normalized_relative_source_path": row.get("normalized_relative_source_path"),
    }


def _publish_staging_dir(staging_root: Path, output_root: Path) -> None:
    output_root = Path(output_root)
    staging_root = Path(staging_root)
    if not staging_root.exists():
        raise FileNotFoundError(f"staging directory missing: {staging_root}")
    backup_root = output_root.with_name(output_root.name + ".old")
    if backup_root.exists():
        shutil.rmtree(backup_root)
    if output_root.exists():
        output_root.rename(backup_root)
    try:
        staging_root.rename(output_root)
        if backup_root.exists():
            shutil.rmtree(backup_root)
    except Exception:
        if output_root.exists():
            shutil.rmtree(output_root)
        if backup_root.exists():
            backup_root.rename(output_root)
        raise


def _copy_or_link_tree(src_dir: Path, dst_dir: Path, mode: str) -> None:
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if item.is_dir():
            _copy_or_link_tree(item, target, mode)
            continue
        if mode == "copy":
            shutil.copy2(item, target)
        else:
            try:
                if mode == "hardlink":
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    os.link(item, target)
                else:
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    os.symlink(item.resolve(), target)
            except OSError:
                shutil.copy2(item, target)


def _copy_or_link_file(src: Path, dst: Path, mode: str) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if mode == "hardlink":
            os.link(src, dst)
        else:
            os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_dataset(
    config: PipelineConfig,
    catalog: CatalogRef,
    selection_plan: SelectionPlan,
    *,
    progress: ProgressSink | None = None,
) -> MaterializeResult:
    output_root = resolve_materialize_output_root(config)
    work_dir = Path(catalog.work_dir)
    mode = config.dataset_builder.materialize.mode
    wav_subdir = config.output.wav_subdir

    plan_path = selection_plan.plan_path or (work_dir / "plans" / "selection_plan.parquet")
    if not plan_path.is_file():
        raise FileNotFoundError(f"selection plan not found: {plan_path}")

    plan_df = pl.read_parquet(plan_path)
    selected_df = plan_df.filter(pl.col("disposition") == ClipDisposition.SELECTED.value)
    if selected_df.is_empty():
        raise ValueError("selection plan contains no SELECTED clips")

    clips_df = read_clips(catalog.resolved_clips_path())
    selected_ids = selected_df["clip_id"].to_list()
    selected_clips = clips_df.filter(pl.col("clip_id").is_in(selected_ids))
    clip_by_id = {str(row["clip_id"]): row for row in selected_clips.iter_rows(named=True)}
    rank_by_id = {
        str(row["clip_id"]): int(row["selection_rank"])
        for row in selected_df.iter_rows(named=True)
        if row.get("selection_rank") is not None
    }
    split_by_id = {str(row["clip_id"]): row.get("split") for row in selected_df.iter_rows(named=True)}

    staging_root = output_root.with_name(output_root.name + ".staging")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    wavs_dir = staging_root / wav_subdir
    wavs_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    split_name_map = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}

    total = len(selected_ids)
    for index, clip_id in enumerate(selected_ids, start=1):
        if progress is not None:
            progress(
                ProgressEvent(
                    stage="materialize",
                    message=clip_id,
                    current=index,
                    total=total,
                    fraction=index / total if total else 1.0,
                )
            )
        row = clip_by_id.get(clip_id)
        if row is None:
            raise KeyError(f"selected clip {clip_id!r} missing from catalog")

        cache_rel = row.get("audio_cache_rel_path")
        if not cache_rel:
            raise ValueError(f"selected clip {clip_id!r} has no audio_cache_rel_path")
        src_wav = work_dir / str(cache_rel)
        if not src_wav.is_file():
            raise FileNotFoundError(f"cached audio missing for {clip_id}: {src_wav}")

        rel_audio = f"{wav_subdir}/{clip_id}.wav"
        dst_wav = staging_root / rel_audio
        _place_audio(src_wav, dst_wav, mode)

        split_raw = split_by_id.get(clip_id) or row.get("split") or "train"
        split_key = split_name_map.get(str(split_raw), "train")
        record = _metadata_record(
            clip_id=clip_id,
            audio_path=rel_audio.replace("\\", "/"),
            row=row,
            split=split_raw,
            selection_rank=rank_by_id.get(clip_id),
        )
        metadata_rows.append(record)
        split_rows[split_key].append(record)

    metadata_path = staging_root / config.output.manifest
    _write_jsonl_atomic(metadata_path, metadata_rows)
    manifest_paths.append(str(metadata_path))

    for split_name, rows in split_rows.items():
        split_path = staging_root / f"{split_name}.jsonl"
        _write_jsonl_atomic(split_path, rows)
        manifest_paths.append(str(split_path))

    plans_dir = staging_root / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    _copy_or_link_file(plan_path, plans_dir / "selection_plan.parquet", mode)
    manifest_paths.append(str(plans_dir / "selection_plan.parquet"))

    split_plan_src = work_dir / "plans" / "split_plan.json"
    if split_plan_src.is_file():
        _copy_or_link_file(split_plan_src, plans_dir / "split_plan.json", mode)
        manifest_paths.append(str(plans_dir / "split_plan.json"))

    reports_src = work_dir / "reports"
    if reports_src.is_dir():
        _copy_or_link_tree(reports_src, staging_root / "reports", mode)

    run_manifest_src = work_dir / "run_manifest.json"
    if run_manifest_src.is_file():
        _copy_or_link_file(run_manifest_src, staging_root / "run_manifest.json", mode)
        manifest_paths.append(str(staging_root / "run_manifest.json"))

    if config.dataset_builder.materialize.atomic_rename:
        _publish_staging_dir(staging_root, output_root)
    else:
        if output_root.exists():
            shutil.rmtree(output_root)
        staging_root.rename(output_root)

    return MaterializeResult(
        output_root=str(output_root),
        selected_count=len(selected_ids),
        manifest_paths=manifest_paths,
    )
