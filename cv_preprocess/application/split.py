from __future__ import annotations

import json
from pathlib import Path

from cv_preprocess.application.common import ProgressEvent, ProgressSink, SplitPlan
from cv_preprocess.catalog import CatalogRef
from cv_preprocess.config import PipelineConfig
from cv_preprocess.reports.serializer import write_json_atomic


def write_split_plan(path: Path, plan: SplitPlan) -> None:
    write_json_atomic(
        path,
        {
            "protocol": plan.protocol,
            "assignments": plan.assignments,
        },
    )


def load_split_plan(catalog: CatalogRef, path: Path) -> SplitPlan:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitPlan(
        catalog=catalog,
        protocol=str(data.get("protocol") or "unseen_speaker"),
        assignments={str(k): str(v) for k, v in (data.get("assignments") or {}).items()},
    )


def plan_dataset_split(
    config: PipelineConfig,
    catalog: CatalogRef,
    *,
    progress: ProgressSink | None = None,
) -> SplitPlan:
    if progress is not None:
        progress(ProgressEvent(stage="plan-split", message="creating split plan"))

    plan = SplitPlan(
        catalog=catalog,
        protocol=config.dataset_builder.split.protocol,
        assignments={},
    )
    plans_dir = Path(catalog.work_dir) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    write_split_plan(plans_dir / "split_plan.json", plan)
    return plan
