from __future__ import annotations

from cv_preprocess.application.common import ProgressSink, SelectionPlan, SplitPlan
from cv_preprocess.catalog import CatalogRef
from cv_preprocess.config import PipelineConfig


def select_dataset(
    config: PipelineConfig,
    catalog: CatalogRef,
    split_plan: SplitPlan,
    *,
    backend: str | None = None,
    progress: ProgressSink | None = None,
) -> SelectionPlan:
    return SelectionPlan(catalog=catalog)
