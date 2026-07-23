from __future__ import annotations

from cv_preprocess.application.common import MaterializeResult, ProgressSink, SelectionPlan
from cv_preprocess.catalog import CatalogRef
from cv_preprocess.config import PipelineConfig


def materialize_dataset(
    config: PipelineConfig,
    catalog: CatalogRef,
    selection_plan: SelectionPlan,
    *,
    progress: ProgressSink | None = None,
) -> MaterializeResult:
    output_root = config.dataset_builder.materialize.output_root or config.output.root
    return MaterializeResult(output_root=str(output_root))
