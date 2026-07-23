from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ClipDisposition(str, Enum):
    HARD_REJECTED = "hard_rejected"
    ELIGIBLE = "eligible"
    SELECTED = "selected"
    RESERVE = "reserve"


class CatalogRef(BaseModel):
    """Pointer to a catalog artifact under the dataset builder work tree."""

    work_dir: Path = Field(default=Path("work"))
    clips_path: Path | None = None
    feature_counts_path: Path | None = None
    speaker_stats_path: Path | None = None
    duplicate_groups_path: Path | None = None
    manifest_path: Path | None = None

    def resolved_clips_path(self) -> Path:
        return self.clips_path or (self.work_dir / "catalog" / "clips.parquet")

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path or (self.work_dir / "catalog" / "manifest.json")
