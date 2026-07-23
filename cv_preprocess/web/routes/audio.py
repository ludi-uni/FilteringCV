from __future__ import annotations

import mimetypes

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from cv_preprocess.web.dependencies import get_app_state, reject_path_traversal, resolve_audio_file

router = APIRouter()

CHUNK_SIZE = 64 * 1024  # reserved for future streaming improvements


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int]:
    if not range_header.startswith("bytes="):
        raise HTTPException(status_code=416, detail="invalid range")
    ranges = range_header.removeprefix("bytes=").split(",")
    if len(ranges) != 1:
        raise HTTPException(status_code=416, detail="multiple ranges not supported")
    start_text, _, end_text = ranges[0].partition("-")
    if not start_text and not end_text:
        raise HTTPException(status_code=416, detail="invalid range")
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    else:
        suffix = int(end_text)
        start = max(file_size - suffix, 0)
        end = file_size - 1
    if start < 0 or end < start or start >= file_size:
        raise HTTPException(status_code=416, detail="range not satisfiable")
    end = min(end, file_size - 1)
    return start, end


@router.get("/{relative_path:path}", response_model=None)
def serve_audio(request: Request, relative_path: str):
    reject_path_traversal(relative_path)
    state = get_app_state(request)
    audio_path = resolve_audio_file(state, relative_path)
    file_size = audio_path.stat().st_size
    media_type = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    if range_header is None:
        return FileResponse(
            path=audio_path,
            media_type=media_type,
            filename=audio_path.name,
        )

    start, end = _parse_range_header(range_header, file_size)
    content_length = end - start + 1
    with audio_path.open("rb") as handle:
        handle.seek(start)
        body = handle.read(content_length)
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": media_type,
    }
    return StreamingResponse(
        iter((body,)),
        status_code=206,
        headers=headers,
        media_type=media_type,
    )
