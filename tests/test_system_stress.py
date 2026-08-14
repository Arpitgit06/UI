import pytest
import io
import time
from threading import Thread
from fastapi.testclient import TestClient

from app.main import app
from tests.utils import generate_test_video


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c

def test_single_heavy_load_job(client):
    """
    Submits a single, high-complexity video that will push the entire pipeline
    (Module A through D) to its limits. Verifies the job reaches a terminal state.
    """
    video_path = generate_test_video(duration_seconds=10, complexity="high")
    
    with open(video_path, "rb") as f:
        resp = client.post(
            "/jobs",
            files={"video": ("heavy_test.mp4", f, "video/mp4")},
            data={"enable_3d": "true", "fast_mode": "false"}
        )
        
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    
    # Poll until complete or failed
    status, error_message = None, None
    for _ in range(300):  # Wait up to 5 minutes for a heavy job
        resp = client.get(f"/jobs/{job_id}")
        body = resp.json()
        status, error_message = body["status"], body.get("error_message")
        if status in {"complete", "failed"}:
            break
        time.sleep(1)
        
    assert status in {"complete", "failed"}, "Heavy job hung and did not reach a terminal state."
    
    # If the system is fully provisioned, it should be 'complete'
    # If unprovisioned, it cleanly fails.
    if status == "failed":
        assert error_message, "A failed job must have an error message."
        assert "Traceback" not in error_message, "System crashed instead of failing gracefully."


def test_concurrent_job_queueing_stability(client):
    """
    Submits multiple jobs concurrently to ensure the queue processes them safely
    without race conditions, deadlocks, or VRAM clashes.
    """
    video_path = generate_test_video(duration_seconds=3, complexity="low")
    video_bytes = video_path.read_bytes()
    
    num_jobs = 5
    job_ids = []
    
    # Submit concurrently
    def submit():
        resp = client.post(
            "/jobs",
            files={"video": ("concurrent.mp4", io.BytesIO(video_bytes), "video/mp4")},
            data={"enable_3d": "false", "fast_mode": "true"}
        )
        if resp.status_code == 200:
            job_ids.append(resp.json()["job_id"])
            
    threads = [Thread(target=submit) for _ in range(num_jobs)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    assert len(job_ids) == num_jobs, "Not all jobs were successfully queued."
    
    # Wait for all to finish
    finished_count = 0
    for _ in range(300): # 5 minutes max
        finished_count = 0
        for j_id in job_ids:
            resp = client.get(f"/jobs/{j_id}")
            if resp.json()["status"] in {"complete", "failed"}:
                finished_count += 1
        
        if finished_count == num_jobs:
            break
            
        time.sleep(1)
        
    assert finished_count == num_jobs, f"Only {finished_count}/{num_jobs} jobs reached terminal state."
