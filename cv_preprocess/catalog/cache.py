from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cv_preprocess import __version__
from cv_preprocess.config import PipelineConfig
from cv_preprocess.config.audio_steps import SaveWavStep
from cv_preprocess.pipeline.export import write_wav_16bit
from cv_preprocess.pipeline.preprocess_efficiency import (
    effective_audio_catalog_for_preprocess,
    resolve_preprocess_pass1_pipeline,
    two_pass_uses_split_pipelines,
)


def _collect_related_model_ids(config: PipelineConfig) -> list[str]:
    """Collect denoise / restore model identifiers that affect cached audio."""
    ids: list[str] = []
    pipelines = [config.audio_pipeline]
    if config.audio_pipeline_align is not None:
        pipelines.append(config.audio_pipeline_align)
    if config.audio_pipeline_enhance is not None:
        pipelines.append(config.audio_pipeline_enhance)
    for pipe in pipelines:
        ids.append(pipe.audio_pipeline_id)
        for step in pipe.steps:
            dumped = step.model_dump()
            for key in (
                "method",
                "model_path",
                "checkpoint_path",
                "hf_repo_id",
                "ssl_model_id",
                "feature_extractor_filename_cuda",
                "decoder_filename_cuda",
                "feature_extractor_filename_cpu",
                "decoder_filename_cpu",
            ):
                value = dumped.get(key)
                if value is not None and str(value).strip():
                    ids.append(f"{step.type}:{key}={value}")
    return sorted(set(ids))


def _output_format_descriptor(config: PipelineConfig) -> str:
    catalog = effective_audio_catalog_for_preprocess(config)
    for step in catalog.steps:
        if isinstance(step, SaveWavStep):
            return f"wav_pcm_{step.bit_depth}"
    return "wav_pcm_16"


def _pipeline_settings_payload(config: PipelineConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "audio_pipeline": config.audio_pipeline.model_dump(mode="json"),
        "two_pass_denoise": config.two_pass_denoise.model_dump(mode="json"),
    }
    if config.audio_pipeline_align is not None:
        payload["audio_pipeline_align"] = config.audio_pipeline_align.model_dump(mode="json")
    if config.audio_pipeline_enhance is not None:
        payload["audio_pipeline_enhance"] = config.audio_pipeline_enhance.model_dump(mode="json")
    if two_pass_uses_split_pipelines(config):
        payload["pass1_pipeline"] = resolve_preprocess_pass1_pipeline(config).model_dump(mode="json")
    return payload


def pipeline_cache_key(config: PipelineConfig) -> str:
    """Hash audio pipeline settings, implementation version, model ids, sample rate, and output format."""
    catalog = effective_audio_catalog_for_preprocess(config)
    canonical = {
        "implementation_version": __version__,
        "target_sample_rate": int(catalog.target_sample_rate),
        "output_format": _output_format_descriptor(config),
        "related_model_ids": _collect_related_model_ids(config),
        "pipeline_settings": _pipeline_settings_payload(config),
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cached_wav_path(work_dir: Path, pipeline_hash: str, audio_sha256: str) -> Path:
    prefix = audio_sha256[:2]
    return work_dir / "audio_cache" / pipeline_hash / prefix / f"{audio_sha256}.wav"


def write_wav_atomic(path: Path, y: np.ndarray, sr: int) -> None:
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    write_wav_16bit(partial, y, sr)
    partial.replace(path)
