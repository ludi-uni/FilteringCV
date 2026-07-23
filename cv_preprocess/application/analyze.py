from __future__ import annotations

from cv_preprocess.application.common import AnalyzeResult, CancellationToken, ProgressSink
from cv_preprocess.config import PipelineConfig


def analyze_project(
    config: PipelineConfig,
    *,
    progress: ProgressSink | None = None,
    cancellation: CancellationToken | None = None,
) -> AnalyzeResult:
    raise NotImplementedError(
        "analyze_project is not implemented yet; enable in a later dataset builder phase"
    )
