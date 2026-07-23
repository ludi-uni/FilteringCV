from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from cv_preprocess.jobs.models import TERMINAL_JOB_STATUSES
from cv_preprocess.jobs.progress import ProgressHub
from cv_preprocess.jobs.store import JobStore


def create_websocket_router(hub: ProgressHub, store: JobStore) -> APIRouter:
    ws_router = APIRouter()

    @ws_router.websocket("/ws/jobs/{job_id}")
    async def job_progress_socket(websocket: WebSocket, job_id: str) -> None:
        try:
            store.get_job(job_id)
        except KeyError:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        queue = await hub.subscribe(job_id)
        last_id = 0
        try:
            for record in store.list_progress(job_id):
                last_id = max(last_id, record.id or 0)
                await websocket.send_json(record.model_dump(mode="json"))

            while True:
                try:
                    record = await asyncio.wait_for(queue.get(), timeout=1.0)
                    last_id = max(last_id, record.id or 0)
                    await websocket.send_json(record.model_dump(mode="json"))
                except asyncio.TimeoutError:
                    job = store.get_job(job_id)
                    for record in store.list_progress(job_id, after_id=last_id):
                        last_id = max(last_id, record.id or 0)
                        await websocket.send_json(record.model_dump(mode="json"))
                    if job.status in TERMINAL_JOB_STATUSES:
                        await websocket.send_json(
                            {
                                "type": "terminal",
                                "status": job.status.value,
                            }
                        )
                        break
                    await websocket.send_json({"type": "ping"})
        except WebSocketDisconnect:
            pass
        finally:
            await hub.unsubscribe(job_id, queue)

    return ws_router
