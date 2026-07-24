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

from cv_preprocess.catalog.ids import stable_clip_id
from cv_preprocess.config.coverage import CoverageAutomationConfig
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
) -> IndexBuildResult:
    root = config.input.corpus_root
    tsv_path = input_tsv or (root / config.input.clip_tsv)
    fingerprint = source_fingerprint(tsv_path, config)
    cfg_hash = config_hash_for_coverage(config)
    meta_out = meta_path_for_index(output)

    existing: dict[str, ClipIndexRecord] = {}
    if incremental and output.is_file() and not force:
        for record in load_index_jsonl(output):
            existing[record.clip_id] = record
        if meta_out.is_file():
            old_meta = ClipIndexMeta.model_validate_json(meta_out.read_text(encoding="utf-8"))
            if old_meta.source_fingerprint != fingerprint:
                logger.warning(
                    "coverage index fingerprint changed; incremental rebuild will refresh changed clips"
                )

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
    current_ids: set[str] = set()
    built: list[ClipIndexRecord] = []

    def _process(item: tuple[int, ClipRow]) -> ClipIndexRecord | None:
        source_row_index, row = item
        # provisional id inputs for reuse check need audio hash; always rebuild cheaply when forced
        return build_index_record(
            row,
            config=config,
            source_row_index=source_row_index,
            root=root,
            source_release=source_release,
            decode_audio=decode_audio,
        )

    workers = max(1, int(workers))
    if workers == 1:
        for item in indexed:
            record = _process(item)
            if record is None:
                continue
            current_ids.add(record.clip_id)
            if incremental and not force and record.clip_id in existing:
                old = existing[record.clip_id]
                if (
                    old.sentence == record.sentence
                    and old.normalized_text == record.normalized_text
                    and old.audio_sha256 == record.audio_sha256
                    and old.source_path == record.source_path
                ):
                    built.append(old)
                    continue
            built.append(record)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, item): item for item in indexed}
            for future in as_completed(futures):
                record = future.result()
                if record is None:
                    continue
                current_ids.add(record.clip_id)
                if incremental and not force and record.clip_id in existing:
                    old = existing[record.clip_id]
                    if (
                        old.sentence == record.sentence
                        and old.normalized_text == record.normalized_text
                        and old.audio_sha256 == record.audio_sha256
                        and old.source_path == record.source_path
                    ):
                        built.append(old)
                        continue
                built.append(record)

    # Preserve stable order by source_row_index
    built.sort(key=lambda r: (r.source_row_index, r.clip_id))
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
    return IndexBuildResult(index_path=output, meta_path=meta_out, clip_count=len(built), meta=meta)
