"""Lightweight clip index builder (JSONL + meta)."""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cv_preprocess.application.common import CancellationToken, ProgressEvent, ProgressSink
from cv_preprocess.catalog.ids import stable_clip_id
from cv_preprocess.config.pipeline import PipelineConfig
from cv_preprocess.coverage.feature_extractor import extract_coverage_features
from cv_preprocess.coverage.models import CheapQuality, ClipIndexMeta, ClipIndexRecord
from cv_preprocess.io.tsv_loader import ClipRow, iter_clip_audio_paths, load_clip_rows_for_pipeline
from cv_preprocess.pipeline.g2p_map_suggest_core import validate_clip_text_norm
from cv_preprocess.pipeline.preprocess.helpers import infer_release
from cv_preprocess.reports.serializer import write_json_atomic
from cv_preprocess.text.phonemize import g2p_phonemes_for_dataset

logger = logging.getLogger(__name__)

INDEX_VERSION = 1
NORMALIZER_VERSION = "normalize_for_tts:v1"
G2P_VERSION = "pyopenjtalk-plus:dataset_g2p:v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_relative_path(path: str) -> str:
    return path.replace("\\", "/")


def config_hash_for_coverage(config: PipelineConfig) -> str:
    payload = {
        "text": config.text.model_dump(mode="json"),
        "coverage": config.coverage.model_dump(mode="json") if hasattr(config, "coverage") else {},
        "input_tsv": str(config.input.clip_tsv),
        "corpus_root": str(config.input.corpus_root),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def source_fingerprint(tsv_path: Path, config: PipelineConfig) -> str:
    parts = [
        str(tsv_path.resolve()),
        str(tsv_path.stat().st_mtime_ns) if tsv_path.is_file() else "missing",
        str(tsv_path.stat().st_size) if tsv_path.is_file() else "0",
        config_hash_for_coverage(config),
        NORMALIZER_VERSION,
        G2P_VERSION,
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _cheap_audio_stats(y: np.ndarray, sr: int) -> tuple[CheapQuality, float]:
    if y.size == 0:
        return CheapQuality(decode_ok=False), 0.0
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(np.square(y))))
    clipping_ratio = float(np.mean(np.abs(y) >= 0.99))
    frame = max(1, int(sr * 0.02))
    if y.size < frame:
        silence_ratio = 1.0 if rms < 0.01 else 0.0
    else:
        frames = y[: len(y) - (len(y) % frame)].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        silence_ratio = float(np.mean(frame_rms < 0.01))
    duration = float(y.size / max(sr, 1))
    return (
        CheapQuality(
            decode_ok=True,
            peak=peak,
            rms=rms,
            clipping_ratio=clipping_ratio,
            silence_ratio=silence_ratio,
        ),
        duration,
    )


def _row_int(row: ClipRow, key: str) -> int | None:
    raw = row.raw.get(key)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def build_index_record(
    row: ClipRow,
    *,
    config: PipelineConfig,
    source_row_index: int,
    root: Path,
    source_release: str,
    decode_audio: bool = True,
) -> ClipIndexRecord | None:
    text_raw = row.sentence
    text_norm, text_rej = validate_clip_text_norm(row, config)
    if text_rej is not None or text_norm is None:
        return None

    try:
        phoneme_str = g2p_phonemes_for_dataset(
            text_norm,
            kana=config.text.g2p_kana,
            word_separator=config.text.phoneme_word_separator,
        )
    except Exception:
        return None

    features = extract_coverage_features(
        normalized_text=text_norm,
        phoneme_str=phoneme_str,
        word_separator=config.text.phoneme_word_separator or "|",
    )

    clip_path = iter_clip_audio_paths(root, config.input.audio_subdir, row)
    audio_sha256 = ""
    cheap = CheapQuality(decode_ok=False)
    duration: float | None = None
    sample_rate: int | None = None
    if clip_path.is_file():
        try:
            audio_sha256 = _sha256_file(clip_path)
        except OSError:
            audio_sha256 = ""
        if decode_audio:
            try:
                from cv_preprocess.audio.decode import load_audio

                y, sr = load_audio(clip_path)
                sample_rate = int(sr)
                cheap, duration = _cheap_audio_stats(np.asarray(y, dtype=np.float32), int(sr))
            except Exception:
                cheap = CheapQuality(decode_ok=False)

    clip_id = stable_clip_id(
        source_release=source_release,
        normalized_relative_source_path=_normalize_relative_path(row.path),
        source_row_index=source_row_index,
        audio_sha256=audio_sha256,
        text_raw=text_raw,
    )
    return ClipIndexRecord(
        clip_id=clip_id,
        source_path=row.path,
        client_id=row.client_id,
        sentence=text_raw,
        normalized_text=text_norm,
        duration_sec=duration,
        sample_rate=sample_rate,
        phonemes=features["phonemes"],
        unique_phonemes=features["unique_phonemes"],
        moras=features["moras"],
        biphones=features["biphones"],
        positioned_phonemes=features["positioned_phonemes"],
        up_votes=_row_int(row, "up_votes"),
        down_votes=_row_int(row, "down_votes"),
        cheap_quality=cheap,
        index_version=INDEX_VERSION,
        source_row_index=source_row_index,
        audio_sha256=audio_sha256,
        feature_keys=features["feature_keys"],
    )


def load_index_jsonl(path: Path) -> list[ClipIndexRecord]:
    records: list[ClipIndexRecord] = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(ClipIndexRecord.model_validate(json.loads(line)))
    return records


def write_index_jsonl(path: Path, records: list[ClipIndexRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    partial.replace(path)


def meta_path_for_index(index_path: Path) -> Path:
    return index_path.with_name(index_path.stem + ".meta.json")


@dataclass
class IndexBuildResult:
    index_path: Path
    meta_path: Path
    clip_count: int
    meta: ClipIndexMeta


def _reuse_existing_record(
    row: ClipRow,
    *,
    source_row_index: int,
    root: Path,
    audio_subdir: str,
    existing_by_row: dict[tuple[str, int], ClipIndexRecord],
) -> ClipIndexRecord | None:
    """Reuse a prior index row when text + audio hash are unchanged (skip G2P/decode)."""
    key = (_normalize_relative_path(row.path), source_row_index)
    old = existing_by_row.get(key)
    if old is None or old.sentence != row.sentence:
        return None
    clip_path = iter_clip_audio_paths(root, audio_subdir, row)
    if not clip_path.is_file():
        if not old.audio_sha256:
            return old
        return None
    try:
        audio_sha256 = _sha256_file(clip_path)
    except OSError:
        return None
    if audio_sha256 != old.audio_sha256:
        return None
    return old


def _emit_index_progress(
    progress: ProgressSink | None,
    *,
    current: int,
    total: int,
    message: str,
    reused: int = 0,
    built_new: int = 0,
) -> None:
    if progress is None:
        return
    fraction = (current / total) if total else 1.0
    progress(
        ProgressEvent(
            stage="coverage-index",
            message=message,
            current=current,
            total=total,
            fraction=fraction,
            metadata={
                "phase": "index",
                "reused": reused,
                "built_new": built_new,
            },
        )
    )


def build_clip_index(
    config: PipelineConfig,
    *,
    output: Path,
    input_tsv: Path | None = None,
    force: bool = False,
    incremental: bool = False,
    workers: int = 1,
    limit: int | None = None,
    decode_audio: bool = True,
    progress: ProgressSink | None = None,
    cancellation: CancellationToken | None = None,
) -> IndexBuildResult:
    root = config.input.corpus_root
    audio_subdir = config.input.audio_subdir
    tsv_path = input_tsv or (root / config.input.clip_tsv)
    fingerprint = source_fingerprint(tsv_path, config)
    cfg_hash = config_hash_for_coverage(config)
    meta_out = meta_path_for_index(output)

    existing_by_row: dict[tuple[str, int], ClipIndexRecord] = {}
    if incremental and output.is_file() and not force:
        for record in load_index_jsonl(output):
            existing_by_row[(_normalize_relative_path(record.source_path), record.source_row_index)] = (
                record
            )
        if meta_out.is_file():
            old_meta = ClipIndexMeta.model_validate_json(meta_out.read_text(encoding="utf-8"))
            if old_meta.source_fingerprint != fingerprint:
                logger.warning(
                    "coverage index fingerprint changed; incremental rebuild will refresh changed clips"
                )

    _emit_index_progress(progress, current=0, total=1, message="loading clip rows")
    if cancellation is not None:
        cancellation.raise_if_cancelled()

    loaded = load_clip_rows_for_pipeline(
        config,
        apply_input_max_clips=False,
        apply_speaker_merge=True,
        sort_by_path=False,
    )
    indexed = list(enumerate(loaded.rows))
    indexed.sort(key=lambda item: (_normalize_relative_path(item[1].path), item[0]))
    if limit is not None:
        indexed = indexed[: max(0, limit)]

    source_release = infer_release(root)
    built: list[ClipIndexRecord] = []
    reused = 0
    built_new = 0
    total = len(indexed)
    can_reuse = bool(incremental and not force and existing_by_row)

    def _process(item: tuple[int, ClipRow]) -> tuple[ClipIndexRecord | None, bool]:
        source_row_index, row = item
        if can_reuse:
            reused_record = _reuse_existing_record(
                row,
                source_row_index=source_row_index,
                root=root,
                audio_subdir=audio_subdir,
                existing_by_row=existing_by_row,
            )
            if reused_record is not None:
                return reused_record, True
        record = build_index_record(
            row,
            config=config,
            source_row_index=source_row_index,
            root=root,
            source_release=source_release,
            decode_audio=decode_audio,
        )
        return record, False

    _emit_index_progress(
        progress,
        current=0,
        total=max(total, 1),
        message="indexing clips",
    )

    workers = max(1, int(workers))
    if workers == 1:
        for i, item in enumerate(indexed, start=1):
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            record, was_reused = _process(item)
            if record is None:
                if i == 1 or i == total or i % 25 == 0:
                    _emit_index_progress(
                        progress,
                        current=i,
                        total=total,
                        message="indexing clips",
                        reused=reused,
                        built_new=built_new,
                    )
                continue
            if was_reused:
                reused += 1
            else:
                built_new += 1
            built.append(record)
            if i == 1 or i == total or i % 25 == 0:
                _emit_index_progress(
                    progress,
                    current=i,
                    total=total,
                    message="indexing clips",
                    reused=reused,
                    built_new=built_new,
                )
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, item): item for item in indexed}
            for future in as_completed(futures):
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                record, was_reused = future.result()
                completed += 1
                if record is None:
                    if completed == 1 or completed == total or completed % 25 == 0:
                        _emit_index_progress(
                            progress,
                            current=completed,
                            total=total,
                            message="indexing clips",
                            reused=reused,
                            built_new=built_new,
                        )
                    continue
                if was_reused:
                    reused += 1
                else:
                    built_new += 1
                built.append(record)
                if completed == 1 or completed == total or completed % 25 == 0:
                    _emit_index_progress(
                        progress,
                        current=completed,
                        total=total,
                        message="indexing clips",
                        reused=reused,
                        built_new=built_new,
                    )

    # Preserve stable order by source_row_index
    built.sort(key=lambda r: (r.source_row_index, r.clip_id))
    _emit_index_progress(
        progress,
        current=total,
        total=max(total, 1),
        message="writing index",
        reused=reused,
        built_new=built_new,
    )
    write_index_jsonl(output, built)
    meta = ClipIndexMeta(
        schema_version=1,
        created_at=_utc_now(),
        source_fingerprint=fingerprint,
        normalizer_version=NORMALIZER_VERSION,
        g2p_version=G2P_VERSION,
        config_hash=cfg_hash,
        clip_count=len(built),
        incremental=bool(incremental and not force),
    )
    write_json_atomic(meta_out, meta)
    _emit_index_progress(
        progress,
        current=total,
        total=max(total, 1),
        message="index complete",
        reused=reused,
        built_new=built_new,
    )
    return IndexBuildResult(index_path=output, meta_path=meta_out, clip_count=len(built), meta=meta)
