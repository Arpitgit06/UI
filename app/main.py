"""
OmniUI Core Orchestration Layer.

Exposes the video-to-UI-code pipeline as a small async FastAPI service:

    POST /jobs                    - upload a video, get a queued Job back
    GET  /jobs/{job_id}           - poll job status
    GET  /jobs/{job_id}/download  - fetch the finished project zip
    GET  /health                  - queue + GPU status

Interactive docs (and, for now, the easiest way to POST a video) live
at /docs. A dedicated upload page is a later iteration.
"""
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sys
import os
import asyncio
import warnings
import logging

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
from app.utils.logger import get_logger

logger = get_logger("omniui.main")

job_queue = JobQueue(worker_fn=run_pipeline)

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
        "engine": "local",
        "model": settings.local_llm_model_name,
        "quantization": "4-bit" if settings.local_llm_load_in_4bit else "full",
    }

    return {
        "status": "ok",
        "queue_size": job_queue.pending_count(),
        "active_job": job_queue.active_job_id(),
        "gpu": gpu_status(settings.cuda_device_index),
        "local_llm": llm_info,
    }


@app.post("/jobs", response_model=Job)
async def create_job(
    video: UploadFile = File(...),
    enable_3d: bool = Form(False),
    fast_mode: bool = Form(False)
) -> Job:
    if video.content_type not in ACCEPTED_VIDEO_TYPES:
        raise HTTPException(415, f"Unsupported content type: {video.content_type}")

    job_id = uuid.uuid4().hex[:12]
    dest_path = settings.uploads_dir / f"{job_id}_{video.filename}"
    with dest_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    now = datetime.now(timezone.utc)
    job = Job(
        job_id=job_id,
        created_at=now,
        updated_at=now,
        source_video_path=str(dest_path),
        enable_3d=enable_3d,
        fast_mode=fast_mode,
    )
    job_queue.register(job)
    await job_queue.enqueue(job_id)
    logger.info(f"Job {job_id} queued ({video.filename}, {dest_path.stat().st_size} bytes, 3d={enable_3d}, fast_mode={fast_mode})")
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
