"""Shared types for trainer exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ExportFormatName = Literal["piper_plus", "style_bert_vits2"]
TextFieldName = Literal["text_norm", "text_raw"]
PlaceMode = Literal["copy", "hardlink", "symlink"]


@dataclass(frozen=True)
class UtteranceRow:
    clip_id: str
    speaker_id: str
    text: str
    source_audio: Path
    split: str | None = None


@dataclass
class ExportResult:
    format: str
    output_dir: Path
    utterance_count: int
    skipped_empty_text: int = 0
    warnings: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
