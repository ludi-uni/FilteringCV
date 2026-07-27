"""Dispatch trainer exports for materialize and CLI re-export."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from cv_preprocess.config.dataset_builder import TrainerExportsConfig
from cv_preprocess.export.common import load_metadata_jsonl, utterances_from_metadata
from cv_preprocess.export.piper_plus import export_piper_plus
from cv_preprocess.export.protocol import ExportFormatName, ExportResult, PlaceMode
from cv_preprocess.export.style_bert_vits2 import export_style_bert_vits2

logger = logging.getLogger(__name__)


def run_trainer_exports(
    *,
    materialize_root: Path,
    exports_root: Path,
    config: TrainerExportsConfig,
    place_mode: PlaceMode,
    formats: list[ExportFormatName] | None = None,
    resample: bool | None = None,
) -> list[ExportResult]:
    materialize_root = Path(materialize_root)
    exports_root = Path(exports_root)
    metadata_path = materialize_root / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.jsonl not found under {materialize_root}")

    rows = load_metadata_jsonl(metadata_path)
    utterances, warnings, skipped = utterances_from_metadata(
        rows,
        materialize_root=materialize_root,
        text_field=config.text_field,
    )
    if not utterances:
        raise ValueError(
            f"no exportable utterances under {materialize_root} "
            f"(rows={len(rows)}, skipped_empty_text={skipped})"
        )

    selected = list(formats) if formats is not None else list(config.formats)
    do_resample = bool(config.resample if resample is None else resample)
    mode: PlaceMode = config.mode or place_mode

    results: list[ExportResult] = []
    for fmt in selected:
        out_dir = exports_root / fmt
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if fmt == "piper_plus":
            result = export_piper_plus(
                utterances,
                out_dir,
                config=config.piper_plus,
                mode=mode,
                resample=do_resample,
            )
        elif fmt == "style_bert_vits2":
            result = export_style_bert_vits2(
                utterances,
                out_dir,
                config=config.style_bert_vits2,
                mode=mode,
                resample=do_resample,
            )
        else:
            raise ValueError(f"unsupported trainer export format: {fmt!r}")
        result.skipped_empty_text = skipped
        result.warnings = list(warnings)
        results.append(result)
        logger.info(
            "trainer export %s: %s utterances -> %s",
            fmt,
            result.utterance_count,
            out_dir,
        )
    return results


def export_from_materialize_root(
    *,
    materialize_root: Path,
    config: TrainerExportsConfig,
    place_mode: PlaceMode = "copy",
    formats: list[ExportFormatName] | None = None,
    resample: bool | None = None,
    exports_root: Path | None = None,
) -> list[ExportResult]:
    materialize_root = Path(materialize_root)
    target = Path(exports_root) if exports_root is not None else materialize_root / "exports"
    return run_trainer_exports(
        materialize_root=materialize_root,
        exports_root=target,
        config=config,
        place_mode=place_mode,
        formats=formats,
        resample=resample,
    )
