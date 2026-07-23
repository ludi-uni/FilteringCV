from __future__ import annotations

from cv_preprocess.application.common import (
    AnalyzeResult,
    AuditReport,
    CancellationToken,
    MaterializeResult,
    ProgressSink,
    ScanResult,
    SelectionPlan,
    SplitPlan,
)
from cv_preprocess.config import PipelineConfig


def build_dataset(
    config: PipelineConfig,
    *,
    progress: ProgressSink | None = None,
    cancellation: CancellationToken | None = None,
) -> tuple[ScanResult, AnalyzeResult, SplitPlan, SelectionPlan, MaterializeResult, AuditReport]:
    raise NotImplementedError(
        "build_dataset is not implemented yet; orchestration arrives in a later dataset builder phase"
    )
