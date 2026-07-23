from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from cv_preprocess.reports.comparison import compare_runs
from cv_preprocess.web.dependencies import get_app_state, reject_path_traversal, resolve_within_root

router = APIRouter()


class CompareRequest(BaseModel):
    left: str
    right: str


@router.post("")
def compare_runs_api(request: Request, body: CompareRequest) -> dict:
    state = get_app_state(request)
    reject_path_traversal(body.left)
    reject_path_traversal(body.right)
    left = resolve_within_root(state.project_root, body.left)
    right = resolve_within_root(state.project_root, body.right)
    if not left.exists():
        raise HTTPException(status_code=404, detail="left path not found")
    if not right.exists():
        raise HTTPException(status_code=404, detail="right path not found")
    return compare_runs(Path(left), Path(right))
