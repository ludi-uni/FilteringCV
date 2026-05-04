# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of the alignment loop below are derived from NVIDIA NeMo
# tools/nemo_forced_aligner/align.py (main alignment pass).
#
# This script is spawned by cv-preprocess (NFA_PYTHON) with cwd = NFA_ALIGN_DIR.
# It loads the ASR model once, then reads JSON lines on stdin:
#   {"op":"init", ...}  -> {"ok":true} or {"ok":false,"detail":"..."}
#   {"op":"align","manifest_filepath":"...","output_dir":"...","batch_size":8} -> {"ok":true} / {"ok":false,...}
#   {"op":"shutdown"} -> exit

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

# NeMo NFA tree (utils/, conf/) — cwd must be NFA_ALIGN_DIR before import.
_align_dir = os.environ.get("CV_NFA_ALIGN_DIR")
if not _align_dir or not os.path.isdir(_align_dir):
    print(json.dumps({"ok": False, "detail": "CV_NFA_ALIGN_DIR missing or not a directory"}), flush=True)
    sys.exit(1)
os.chdir(_align_dir)
if _align_dir not in sys.path:
    sys.path.insert(0, _align_dir)

from utils.data_prep import get_batch_starts_ends, get_manifest_lines_batch, is_entry_in_all_lines, is_entry_in_any_lines
from utils.make_ass_files import make_ass_files
from utils.make_ctm_files import make_ctm_files
from utils.make_output_manifest import write_manifest_out_line

from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.utils.aligner_utils import (
    add_t_start_end_to_utt_obj,
    get_batch_variables,
    viterbi_decoding,
)
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
from nemo.utils import logging as nemo_logging


def _reply(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _load_model(init: dict[str, Any]) -> tuple[Any, torch.device, torch.device, dict[str, Any]]:
    transcribe_device = torch.device(
        init["transcribe_device"] if init.get("transcribe_device") else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    viterbi_device = torch.device(
        init["viterbi_device"] if init.get("viterbi_device") else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg_setup = OmegaConf.create(
        {
            "pretrained_name": init.get("pretrained_name"),
            "model_path": init.get("model_path"),
        }
    )
    if cfg_setup.model_path not in (None, "None", "") and str(cfg_setup.model_path).strip():
        pass
    elif cfg_setup.pretrained_name:
        cfg_setup.model_path = None
    else:
        raise ValueError("init requires pretrained_name or model_path")

    model, _ = setup_model(cfg_setup, transcribe_device)
    model.eval()
    if isinstance(model, EncDecHybridRNNTCTCModel):
        model.change_decoding_strategy(decoder_type="ctc")
    if init.get("use_local_attention", True):
        nemo_logging.info("use_local_attention=True => try local attention for Conformer")
        model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[64, 64])
    if not (isinstance(model, EncDecCTCModel) or isinstance(model, EncDecHybridRNNTCTCModel)):
        raise NotImplementedError("NFA supports EncDecCTCModel or EncDecHybridRNNTCTCModel only")
    static = {
        "align_using_pred_text": bool(init.get("align_using_pred_text", False)),
        "use_local_attention": bool(init.get("use_local_attention", True)),
        "use_buffered_chunked_streaming": bool(init.get("use_buffered_chunked_streaming", False)),
        "chunk_len_in_secs": float(init.get("chunk_len_in_secs", 1.6)),
        "total_buffer_in_secs": float(init.get("total_buffer_in_secs", 4.0)),
        "chunk_batch_size": int(init.get("chunk_batch_size", 32)),
        "simulate_cache_aware_streaming": init.get("simulate_cache_aware_streaming", False),
        "additional_segment_grouping_separator": init.get("additional_segment_grouping_separator")
        or [".", "?", "!", "..."],
        "audio_filepath_parts_in_utt_id": int(init.get("audio_filepath_parts_in_utt_id", 1)),
        "save_output_file_formats": init.get("save_output_file_formats") or ["ctm"],
        "ctm_file_config": init.get("ctm_file_config")
        or {"remove_blank_tokens": False, "minimum_timestamp_duration": 0.0},
        "ass_file_config": init.get("ass_file_config"),
    }
    return model, transcribe_device, viterbi_device, static


def _run_align(
    model: Any,
    transcribe_device: torch.device,
    viterbi_device: torch.device,
    static: dict[str, Any],
    manifest_filepath: str,
    output_dir: str,
    batch_size: int,
) -> None:
    cfg = OmegaConf.create(
        {
            "manifest_filepath": manifest_filepath,
            "output_dir": output_dir,
            "batch_size": int(batch_size),
            "align_using_pred_text": static["align_using_pred_text"],
            "additional_segment_grouping_separator": static["additional_segment_grouping_separator"],
            "audio_filepath_parts_in_utt_id": static["audio_filepath_parts_in_utt_id"],
            "save_output_file_formats": static["save_output_file_formats"],
            "ctm_file_config": static["ctm_file_config"],
            "ass_file_config": static.get("ass_file_config")
            or {
                "fontsize": 20,
                "vertical_alignment": "center",
                "resegment_text_to_fill_space": False,
                "max_lines_per_segment": 2,
                "text_already_spoken_rgb": [49, 46, 61],
                "text_being_spoken_rgb": [57, 171, 9],
                "text_not_yet_spoken_rgb": [194, 193, 199],
            },
            "use_buffered_chunked_streaming": static["use_buffered_chunked_streaming"],
            "simulate_cache_aware_streaming": static["simulate_cache_aware_streaming"],
            "chunk_len_in_secs": static["chunk_len_in_secs"],
            "total_buffer_in_secs": static["total_buffer_in_secs"],
            "chunk_batch_size": static["chunk_batch_size"],
        }
    )

    if not is_entry_in_all_lines(cfg.manifest_filepath, "audio_filepath"):
        raise RuntimeError("manifest missing audio_filepath on some lines")
    if cfg.align_using_pred_text:
        if is_entry_in_any_lines(cfg.manifest_filepath, "pred_text"):
            raise RuntimeError("align_using_pred_text incompatible with pred_text in manifest")
    else:
        if not is_entry_in_all_lines(cfg.manifest_filepath, "text"):
            raise RuntimeError("manifest missing text on some lines")

    use_buf = bool(cfg.use_buffered_chunked_streaming)
    buffered_chunk_params: dict[str, Any] = {}
    work_model: Any = model
    if use_buf:
        model_cfg = copy.deepcopy(model._cfg)
        OmegaConf.set_struct(model_cfg.preprocessor, False)
        model_cfg.preprocessor.dither = 0.0
        model_cfg.preprocessor.pad_to = 0
        if model_cfg.preprocessor.normalize != "per_feature":
            raise RuntimeError("buffered NFA requires per_feature normalization in model config")
        OmegaConf.set_struct(model_cfg.preprocessor, True)
        mdf = float(cfg.get("model_downsample_factor", 8))
        feature_stride = model_cfg.preprocessor["window_stride"]
        model_stride_in_secs = feature_stride * mdf
        chunk_len = float(cfg.chunk_len_in_secs)
        total_buffer = float(cfg.total_buffer_in_secs)
        tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
        mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
        work_model = FrameBatchASR(
            asr_model=model,
            frame_len=chunk_len,
            total_buffer=total_buffer,
            batch_size=int(cfg.chunk_batch_size),
        )
        buffered_chunk_params = {
            "delay": mid_delay,
            "model_stride_in_secs": model_stride_in_secs,
            "tokens_per_chunk": tokens_per_chunk,
        }

    starts, ends = get_batch_starts_ends(cfg.manifest_filepath, int(cfg.batch_size))
    output_timestep_duration = None
    os.makedirs(cfg.output_dir, exist_ok=True)
    tgt_manifest_name = str(Path(cfg.manifest_filepath).stem) + "_with_output_file_paths.json"
    tgt_manifest_filepath = str(Path(cfg.output_dir) / tgt_manifest_name)
    with open(tgt_manifest_filepath, "w", encoding="utf-8") as f_manifest_out:
        for start, end in zip(starts, ends):
            manifest_lines_batch = get_manifest_lines_batch(cfg.manifest_filepath, start, end)
            if not cfg.align_using_pred_text:
                gt_text_batch = [line.get("text", "") for line in manifest_lines_batch]
            else:
                gt_text_batch = None
            (
                log_probs_batch,
                y_batch,
                T_batch,
                U_batch,
                utt_obj_batch,
                output_timestep_duration,
            ) = get_batch_variables(
                audio=[line["audio_filepath"] for line in manifest_lines_batch],
                model=work_model,
                segment_separators=cfg.additional_segment_grouping_separator,
                align_using_pred_text=cfg.align_using_pred_text,
                audio_filepath_parts_in_utt_id=cfg.audio_filepath_parts_in_utt_id,
                gt_text_batch=gt_text_batch,
                output_timestep_duration=output_timestep_duration,
                simulate_cache_aware_streaming=cfg.simulate_cache_aware_streaming,
                use_buffered_chunked_streaming=cfg.use_buffered_chunked_streaming,
                buffered_chunk_params=buffered_chunk_params,
            )
            alignments_batch = viterbi_decoding(log_probs_batch, y_batch, T_batch, U_batch, viterbi_device)
            for utt_obj, alignment_utt in zip(utt_obj_batch, alignments_batch):
                utt_obj = add_t_start_end_to_utt_obj(utt_obj, alignment_utt, output_timestep_duration)
                if "ctm" in cfg.save_output_file_formats:
                    utt_obj = make_ctm_files(utt_obj, cfg.output_dir, cfg.ctm_file_config)
                if "ass" in cfg.save_output_file_formats:
                    utt_obj = make_ass_files(utt_obj, cfg.output_dir, cfg.ass_file_config)
                write_manifest_out_line(f_manifest_out, utt_obj)


def main() -> None:
    model = None
    transcribe_device: torch.device | None = None
    viterbi_device: torch.device | None = None
    static: dict[str, Any] | None = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _reply({"ok": False, "detail": f"json:{e}"})
            continue
        op = msg.get("op")
        if op == "shutdown":
            break
        if op == "init":
            try:
                model, transcribe_device, viterbi_device, static = _load_model(msg)
                _reply({"ok": True})
            except Exception as e:
                _reply({"ok": False, "detail": f"{type(e).__name__}: {e}"})
                sys.exit(2)
            continue
        if op == "align":
            if model is None or static is None or transcribe_device is None or viterbi_device is None:
                _reply({"ok": False, "detail": "worker not initialized"})
                continue
            try:
                _run_align(
                    model,
                    transcribe_device,
                    viterbi_device,
                    static,
                    str(msg["manifest_filepath"]),
                    str(msg["output_dir"]),
                    int(msg.get("batch_size", 1)),
                )
                _reply({"ok": True})
            except Exception as e:
                _reply({"ok": False, "detail": f"{type(e).__name__}: {e}"})
            continue
        _reply({"ok": False, "detail": f"unknown op: {op!r}"})


if __name__ == "__main__":
    main()
