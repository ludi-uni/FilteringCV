from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from cv_preprocess.catalog import CatalogRef


@runtime_checkable
class ProgressSink(Protocol):
    def __call__(self, event: ProgressEvent) -> None: ...


class ProgressEvent(BaseModel):
    stage: str
    message: str = ""
    current: int | None = None
    total: int | None = None
    fraction: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise RuntimeError("operation cancelled")


class ScanResult(BaseModel):
    tsv_path: str
    stats: dict[str, Any]
    rows_after_speaker_filter: int
    rows_after_clip_metadata_filter: int
    merge_filtered_speakers_as_one: bool
    merged_speaker_client_id_effective: str | None = None
    unique_client_ids_after_filters: int
    unique_client_ids_effective: int
    clip_metadata_filters: dict[str, Any]
    speaker_filter_list_size: int
    unique_client_ids: int
    sample_client_ids_from_parsed_tsv: list[str]
    warnings: list[str]
    sample_missing_audio_first10: list[str]
    total_missing_audio_sampled: int


class AnalyzeResult(BaseModel):
    catalog: CatalogRef
    eligible_count: int = 0
    hard_rejected_count: int = 0


class SplitPlan(BaseModel):
    catalog: CatalogRef
    protocol: str
    assignments: dict[str, str] = Field(default_factory=dict)


class SelectionPlan(BaseModel):
    catalog: CatalogRef
    selected_clip_ids: list[str] = Field(default_factory=list)
    reserve_clip_ids: list[str] = Field(default_factory=list)


class MaterializeResult(BaseModel):
    output_root: str
    selected_count: int = 0
    manifest_paths: list[str] = Field(default_factory=list)


class AuditReport(BaseModel):
    catalog: CatalogRef
    passed: bool = True
    issues: list[str] = Field(default_factory=list)
