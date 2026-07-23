from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from cv_preprocess.selection.overrides import ClipOverride, load_overrides, resolved_overrides_path
from cv_preprocess.web.dependencies import get_app_state

router = APIRouter()


class OverrideListResponse(BaseModel):
    path: str
    overrides: list[ClipOverride]


class OverrideUpsertRequest(BaseModel):
    clip_id: str
    action: str
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.get("", response_model=OverrideListResponse)
def list_overrides(request: Request) -> OverrideListResponse:
    state = get_app_state(request)
    path = resolved_overrides_path(state.work_dir)
    overrides_map = load_overrides(path)
    return OverrideListResponse(
        path=str(path),
        overrides=list(overrides_map.values()),
    )


@router.put("", response_model=OverrideListResponse)
def upsert_override(request: Request, body: OverrideUpsertRequest) -> OverrideListResponse:
    state = get_app_state(request)
    path = resolved_overrides_path(state.work_dir)
    overrides_map = load_overrides(path)
    override = ClipOverride.model_validate(body.model_dump())
    overrides_map[override.clip_id] = override
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item.model_dump(mode="json"), ensure_ascii=False) for item in overrides_map.values()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return OverrideListResponse(path=str(path), overrides=list(overrides_map.values()))


@router.delete("/{clip_id}", response_model=OverrideListResponse)
def delete_override(request: Request, clip_id: str) -> OverrideListResponse:
    if not clip_id or clip_id.startswith(("/", "\\")) or ".." in clip_id.split("/"):
        raise HTTPException(status_code=400, detail="invalid clip_id")
    state = get_app_state(request)
    path = resolved_overrides_path(state.work_dir)
    overrides_map = load_overrides(path)
    overrides_map.pop(clip_id, None)
    if overrides_map:
        lines = [
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
            for item in overrides_map.values()
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif path.is_file():
        path.unlink()
    return OverrideListResponse(path=str(path), overrides=list(overrides_map.values()))
