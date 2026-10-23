"""
A strictly-sequential async job queue.

Only one job runs at a time by design: Module B/D's GPU-bound stages
cannot safely share VRAM with a second job's models loaded
concurrently. Serializing here is what makes the VRAMBudgetExceeded
checks in vram_manager.py meaningful — there is only ever one caller.
"""
import asyncio
import re
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from app.config import settings
from app.models.schemas import Job, JobStatus
from app.utils.logger import get_logger

logger = get_logger("omniui.queue")


def persist_job(job: Job) -> None:
    directory = settings.jobs_dir / job.job_id
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "job.json"
    temporary = directory / "job.json.tmp"
    temporary.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(target)


class JobQueue:
    def __init__(self, worker_fn: Callable[[Job], Awaitable[None]]):
        self._worker_fn = worker_fn
        self._jobs: dict[str, Job] = {}
        self._queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._active_job_id: Optional[str] = None

    def register(self, job: Job) -> None:
        self._jobs[job.job_id] = job
        persist_job(job)

    def get(self, job_id: str) -> Optional[Job]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", job_id):
            return None
        if job_id not in self._jobs:
            target = settings.jobs_dir / job_id / "job.json"
            if target.exists():
                self._jobs[job_id] = Job.model_validate_json(target.read_text(encoding="utf-8"))
        return self._jobs.get(job_id)

    async def enqueue(self, job_id: str) -> None:
        if self.pending_count() >= settings.max_pending_jobs:
            raise RuntimeError("Job queue is full")
        await self._queue.put(job_id)

    def pending_count(self) -> int:
        return self._queue.qsize()

    def active_job_id(self) -> Optional[str]:
        return self._active_job_id

    async def start(self) -> None:
        from app.core.process_runner import resume_workers
        resume_workers()
        # A lifespan may be restarted by tests or an embedding application.
        # asyncio queues must not retain a binding to a closed event loop.
        self._queue = asyncio.Queue()
        self._jobs = {}
        for target in settings.jobs_dir.glob("*/job.json"):
            try:
                job = Job.model_validate_json(target.read_text(encoding="utf-8"))
                if job.status not in {JobStatus.COMPLETE, JobStatus.FAILED}:
                    job.status = JobStatus.FAILED
                    job.error_message = "Server restarted before this job completed; upload again"
                    job.updated_at = datetime.now(timezone.utc)
                    persist_job(job)
                self._jobs[job.job_id] = job
            except (OSError, ValueError):
                logger.exception("Unable to restore %s", target)
        self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        from app.core.process_runner import shutdown_workers
        shutdown_workers()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while True:
            job_id = await self._queue.get()
            self._active_job_id = job_id
            job = self._jobs[job_id]
            try:
                logger.info(f"Starting job {job_id}")
                await self._worker_fn(job)
            except Exception as exc:
                logger.exception(f"Job {job_id} failed")
                job.status = JobStatus.FAILED
                job.error_message = str(exc)
            except asyncio.CancelledError:
                job.status = JobStatus.FAILED
                job.error_message = "Server stopped during processing"
                raise
            finally:
                job.updated_at = datetime.now(timezone.utc)
                persist_job(job)
                self._active_job_id = None
                self._queue.task_done()
