from __future__ import annotations

from cv_preprocess.application.common import AuditReport, SelectionPlan
from cv_preprocess.catalog import CatalogRef
from cv_preprocess.config import PipelineConfig


def audit_dataset(
    config: PipelineConfig,
    catalog: CatalogRef,
    selection_plan: SelectionPlan,
) -> AuditReport:
    return AuditReport(catalog=catalog)
