from __future__ import annotations

from typing import Any

import polars as pl
from fastapi import APIRouter, HTTPException, Query, Request

from cv_preprocess.catalog.reader import read_clips
from cv_preprocess.web.dependencies import get_app_state

router = APIRouter()


@router.get("/clips")
def list_clips(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    disposition: str | None = None,
    speaker_id: str | None = None,
    split: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    state = get_app_state(request)
    clips_path = state.catalog_dir / "clips.parquet"
    if not clips_path.is_file():
        raise HTTPException(status_code=404, detail="catalog not found")

    df = read_clips(clips_path)
    if disposition:
        df = df.filter(pl.col("disposition") == disposition)
    if speaker_id:
        df = df.filter(pl.col("speaker_id") == speaker_id)
    if split:
        df = df.filter(pl.col("split") == split)
    if search:
        needle = search.strip()
        if needle:
            df = df.filter(
                pl.col("clip_id").str.contains(needle, literal=True)
                | pl.col("text_norm").str.contains(needle, literal=True)
                | pl.col("speaker_id").str.contains(needle, literal=True)
            )

    total = df.height
    offset = (page - 1) * page_size
    page_df = df.slice(offset, page_size)
    items = page_df.to_dicts()
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items,
    }
