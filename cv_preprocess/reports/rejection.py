from __future__ import annotations

from collections import Counter, defaultdict

import polars as pl

from cv_preprocess.catalog.models import ClipDisposition
from cv_preprocess.reports.models import RejectionReasonEntry, RejectionReport

_FEATURE_LIST_COLUMNS = ("biphones", "triphones", "moras", "fullcontext_labels")


def _lost_features_for_row(row: dict[str, object]) -> Counter[str]:
    lost: Counter[str] = Counter()
    for column in _FEATURE_LIST_COLUMNS:
        values = row.get(column)
        if isinstance(values, list):
            lost[column] += len(values)
    phonemes = row.get("phonemes")
    if phonemes:
        lost["phonemes"] += len(str(phonemes).split())
    return lost


def compute_rejection_summary(clips: pl.DataFrame) -> RejectionReport:
    """Summarize hard rejections and feature loss by reject reason."""
    eligible_value = ClipDisposition.ELIGIBLE.value
    total_clips = clips.height
    eligible_count = int(clips.filter(pl.col("disposition") == eligible_value).height)
    hard_rejected_count = total_clips - eligible_count

    rejected = clips.filter(pl.col("disposition") != eligible_value)
    if rejected.is_empty():
        return RejectionReport(
            total_clips=total_clips,
            eligible_count=eligible_count,
            hard_rejected_count=hard_rejected_count,
        )

    by_reason_counts: Counter[str] = Counter()
    lost_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rejected.iter_rows(named=True):
        reason = str(row.get("reject_reason") or "unknown")
        by_reason_counts[reason] += 1
        lost_by_reason[reason].update(_lost_features_for_row(row))

    by_reason = [
        RejectionReasonEntry(
            reject_reason=reason,
            clip_count=count,
            lost_feature_counts=dict(sorted(lost_by_reason[reason].items())),
        )
        for reason, count in sorted(by_reason_counts.items())
    ]
    return RejectionReport(
        total_clips=total_clips,
        eligible_count=eligible_count,
        hard_rejected_count=hard_rejected_count,
        by_reason=by_reason,
    )
