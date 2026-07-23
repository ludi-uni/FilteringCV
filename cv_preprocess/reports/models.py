from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureCoverageEntry(BaseModel):
    feature_type: str
    feature: str
    pool_count: int
    pool_speaker_count: int
    pool_utterance_count: int


class CoverageReport(BaseModel):
    total_clips: int = 0
    eligible_clips: int = 0
    feature_types: list[str] = Field(default_factory=list)
    unique_features: int = 0
    entries: list[FeatureCoverageEntry] = Field(default_factory=list)
    js_distance_to_uniform: dict[str, float] = Field(default_factory=dict)


class RejectionReasonEntry(BaseModel):
    reject_reason: str
    clip_count: int
    lost_feature_counts: dict[str, int] = Field(default_factory=dict)


class RejectionReport(BaseModel):
    total_clips: int = 0
    eligible_count: int = 0
    hard_rejected_count: int = 0
    by_reason: list[RejectionReasonEntry] = Field(default_factory=list)


class CatalogReport(BaseModel):
    coverage: CoverageReport
    rejection: RejectionReport
