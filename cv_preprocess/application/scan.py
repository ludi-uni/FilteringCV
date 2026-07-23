from __future__ import annotations

from cv_preprocess.application.common import ProgressEvent, ProgressSink, ScanResult
from cv_preprocess.config import PipelineConfig
from cv_preprocess.pipeline.scan import scan_corpus


def scan_project(
    config: PipelineConfig,
    *,
    progress: ProgressSink | None = None,
) -> ScanResult:
    if progress is not None:
        progress(ProgressEvent(stage="scan", message="scanning corpus"))
    raw = scan_corpus(config)
    return ScanResult.model_validate(raw)
