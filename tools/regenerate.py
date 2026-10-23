"""Run a real, bounded local reconstruction and leave all artifacts reviewable."""
import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.core.pipeline import run_pipeline
from app.core.job_queue import persist_job
from app.models.schemas import Job, JobStatus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--max-states",type=int,default=6)
    parser.add_argument("--mode",choices=["hybrid","asset-backed","editable"],default="hybrid")
    args = parser.parse_args()
    settings.ensure_directories()
    settings.max_key_states = args.max_states
    now = datetime.now(timezone.utc)
    job = Job(job_id="repair_"+uuid.uuid4().hex[:8],created_at=now,updated_at=now,
              source_video_path=str(args.video.resolve()),render_mode=args.mode)
    try:
        asyncio.run(run_pipeline(job))
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error_message = str(exc)
        raise
    finally:
        persist_job(job)
        print(job.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
