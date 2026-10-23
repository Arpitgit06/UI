"""Local video reconstruction API and upload dashboard with bounded uploads and persisted sequential jobs."""
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sys
import os
import asyncio
import warnings
import logging
from typing import Literal
import aiofiles

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from app.config import settings
from app.core.job_queue import JobQueue
from app.core.pipeline import run_pipeline
from app.core.vram_manager import gpu_status
from app.models.schemas import Job, JobStatus
from app.core.review import ReviewRequest
from app.utils.logger import get_logger

logger = get_logger("omniui.main")

job_queue = JobQueue(worker_fn=run_pipeline)
active_reviews=set()

ACCEPTED_VIDEO_TYPES = {"video/mp4", "video/webm", "video/x-matroska"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    await job_queue.start()
    logger.info("OmniUI orchestration layer started.")
    yield
    await job_queue.stop()
    logger.info("OmniUI orchestration layer stopped.")


app = FastAPI(
    title="OmniUI",
    description="Local, offline video-to-UI-code pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local-only tool; tighten if ever exposed beyond localhost
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(settings.project_root / "static")), name="static")


@app.get("/")
async def serve_dashboard() -> FileResponse:
    index_path = settings.project_root / "static" / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Dashboard index.html not found in static/")
    return FileResponse(index_path)


@app.get("/health")
async def health() -> dict:
    llm_info = {
        "engine":"deterministic", "model":"Deterministic React compiler",
        "quantization":"not applicable",
    }

    return {
        "status": "ok",
        "queue_size": job_queue.pending_count(),
        "active_job": job_queue.active_job_id(),
        "active_reviews": len(active_reviews),
        "gpu": gpu_status(settings.cuda_device_index),
        "local_llm": llm_info,
        "model_plan":{'recognition':settings.vision_detector_model_name,
                      'recognition_precision':'NF4','artwork':settings.artwork_generator,
                      'execution':'separate sequential subprocesses','code_generation':'deterministic; no LLM'},
    }


@app.get("/examples/forge-hub/{asset_path:path}")
async def serve_forge_example(asset_path: str = "") -> FileResponse:
    root = (settings.project_root / "examples" / "forge-hub" / "dist").resolve()
    target = (root / (asset_path or "index.html")).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404, "Example not built or asset not found")
    return FileResponse(target)


@app.post("/jobs", response_model=Job)
async def create_job(
    video: UploadFile = File(...),
    enable_3d: bool = Form(False),
    fast_mode: bool = Form(False),
    render_mode: Literal["hybrid", "asset-backed", "editable"] = Form("hybrid"),
) -> Job:
    if video.content_type not in ACCEPTED_VIDEO_TYPES:
        raise HTTPException(415, f"Unsupported content type: {video.content_type}")

    if job_queue.pending_count() >= settings.max_pending_jobs:
        raise HTTPException(429, "Job queue is full; retry after an active job completes")
    job_id = uuid.uuid4().hex[:12]
    extension = {"video/mp4":".mp4", "video/webm":".webm", "video/x-matroska":".mkv"}[video.content_type]
    dest_path = settings.uploads_dir / f"{job_id}{extension}"
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        async with aiofiles.open(dest_path, "wb") as target:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, "Video exceeds configured upload size limit")
                await target.write(chunk)
        if not size:
            raise HTTPException(400, "Video is empty")
    except BaseException:
        dest_path.unlink(missing_ok=True)
        raise
    finally:
        await video.close()

    now = datetime.now(timezone.utc)
    job = Job(
        job_id=job_id,
        created_at=now,
        updated_at=now,
        source_video_path=str(dest_path),
        enable_3d=enable_3d,
        fast_mode=fast_mode,
        render_mode=render_mode,
    )
    if job_queue.pending_count() >= settings.max_pending_jobs:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(429, "Job queue is full")
    job_queue.register(job)
    await job_queue.enqueue(job_id)
    logger.info(f"Job {job_id} queued ({size} bytes, mode={render_mode})")
    return job


@app.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str) -> Job:
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/jobs/{job_id}/download")
async def download_job(job_id: str) -> FileResponse:
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETE or not job.output_zip_path:
        raise HTTPException(409, f"Job not ready (status={job.status.value})")
    return FileResponse(
        job.output_zip_path,
        media_type="application/zip",
        filename=f"omniui_{job_id}.zip",
    )


@app.get("/jobs/{job_id}/preview/{asset_path:path}")
async def preview_job(job_id: str, asset_path: str = "") -> FileResponse:
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(409, "Preview is available after verification completes")
    root = (settings.jobs_dir / job_id / "generated" / ".verified-build").resolve()
    target = (root / (asset_path or "index.html")).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404, "Preview asset not found")
    return FileResponse(target)


def completed_review_parent(job_id):
    job=job_queue.get(job_id)
    if job is None:raise HTTPException(404,'Job not found')
    if job.status!=JobStatus.COMPLETE:raise HTTPException(409,'Review requires a completed job')
    return job


@app.get('/jobs/{job_id}/review')
async def get_review(job_id:str):
    from app.core.review import review_data
    parent=completed_review_parent(job_id)
    try:return await asyncio.to_thread(review_data,parent)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@app.get('/jobs/{job_id}/review/ui')
async def review_ui(job_id:str):
    completed_review_parent(job_id)
    return FileResponse(settings.project_root/'static'/'review.html')


@app.get('/jobs/{job_id}/review/frame/{state_id}')
async def review_frame(job_id:str,state_id:str):
    from app.core.review import load_layouts
    completed_review_parent(job_id)
    layout=next((layout for layout in load_layouts(job_id) if layout.state_name==state_id),None)
    if layout is None:raise HTTPException(404,'State not found')
    source=Path(layout.source_frame.image_path).resolve()
    if not source.is_relative_to(settings.jobs_dir.resolve()) or not source.is_file():raise HTTPException(404,'Source frame unavailable')
    return FileResponse(source)


@app.post('/jobs/{job_id}/review',response_model=Job,status_code=202)
async def submit_review(job_id:str,request:ReviewRequest,background:BackgroundTasks):
    from app.core.review import review_data,load_layouts,prepare_review,make_review_job,build_review
    parent=completed_review_parent(job_id)
    if len(active_reviews)>=2:raise HTTPException(429,'Two reviews are already rebuilding; retry shortly')
    try:
        data=await asyncio.to_thread(review_data,parent)
        if set(request.exclude_motion_regions)-{region['id'] for region in data['motion']['regions']}:raise ValueError('Unknown motion region exclusion')
        layouts,actions=prepare_review(load_layouts(job_id),request,data['actions'])
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    if len(active_reviews)>=2:raise HTTPException(429,'Review slots filled; retry shortly')
    child=make_review_job(parent);child.key_states_detected=len(layouts)
    job_queue.register(child);active_reviews.add(child.job_id)
    async def rebuild():
        try:await build_review(parent,child,layouts,actions,request)
        finally:active_reviews.discard(child.job_id)
    background.add_task(rebuild)
    return child
