from __future__ import annotations

from cv_preprocess.config import PipelineConfig
from cv_preprocess.pipeline.preprocess.session import PreprocessSession


def run_preprocess(cfg: PipelineConfig, *, show_progress: bool = True) -> dict:
    return PreprocessSession(cfg, show_progress=show_progress).run()
