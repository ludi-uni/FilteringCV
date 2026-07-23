from __future__ import annotations

from cv_preprocess.application.common import ProgressSink, SplitPlan
from cv_preprocess.catalog import CatalogRef
from cv_preprocess.config import PipelineConfig


def plan_dataset_split(
    config: PipelineConfig,
    catalog: CatalogRef,
    *,
    progress: ProgressSink | None = None,
) -> SplitPlan:
    return SplitPlan(
        catalog=catalog,
        protocol=config.dataset_builder.split.protocol,
        assignments={},
    )
