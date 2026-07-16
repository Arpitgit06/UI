"""
A strictly-sequential async job queue.

Only one job runs at a time by design: Module B/D's GPU-bound stages
cannot safely share VRAM with a second job's models loaded
concurrently. Serializing here is what makes the VRAMBudgetExceeded
checks in vram_manager.py meaningful — there is only ever one caller.
"""
import asyncio
from typing import Awaitable, Callable, Optional

from app.models.schemas import Job, JobStatus
from app.utils.logger import get_logger

logger = get_logger("omniui.queue")


class JobQueue:
    def __init__(self, worker_fn: Callable[[Job], Awaitable[None]]):
        self._worker_fn = worker_fn
        self._jobs: dict[str, Job] = {}
        self._queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._active_job_id: Optional[str] = None

    def register(self, job: Job) -> None:
        self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(job_id)

    def pending_count(self) -> int:
        return self._queue.qsize()

    def active_job_id(self) -> Optional[str]:
        return self._active_job_id

    async def start(self) -> None:
        self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
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
            finally:
                self._active_job_id = None
                self._queue.task_done()
