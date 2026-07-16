"""
End-to-end smoke test for the orchestration layer, using a REAL small
video so Module A's real SSIM/OpenCV logic has something valid to
process. This does NOT assume ultralytics/paddleocr/transformers/torch
are installed -- those are heavy, optional
dependencies most dev/CI environments won't have, and Modules B/D
already handle their absence with clear errors (see each module's
docstring). So instead of asserting the job reaches "complete" (which
requires every one of those to be present and working), this asserts
the job reaches a TERMINAL state without hanging, and that a failure
(the expected outcome in a partially-provisioned environment) comes with
a clear, specific error message rather than a raw crash. In a fully
provisioned environment (all four modules' dependencies installed), the same test
should reach "complete" -- see the assertion at the bottom for that case.
"""
import io
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _make_test_video_bytes() -> bytes:
    """A tiny but genuinely valid MP4 that OpenCV can decode, so Module A's
    real video-reading step has real (if trivial) work to do rather than
    failing immediately on garbage bytes."""
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "smoke_test.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
    frame = np.zeros((64, 64, 3), np.uint8)
    for _ in range(10):
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_job_lifecycle_reaches_a_terminal_state(client):
    video_bytes = _make_test_video_bytes()
    resp = client.post("/jobs", files={"video": ("test.mp4", io.BytesIO(video_bytes), "video/mp4")})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert resp.json()["status"] == "queued"

    status, error_message = None, None
    for _ in range(100):
        resp = client.get(f"/jobs/{job_id}")
        body = resp.json()
        status, error_message = body["status"], body.get("error_message")
        if status in {"complete", "failed"}:
            break
        time.sleep(0.1)

    assert status in {"complete", "failed"}, f"job got stuck at {status!r} instead of reaching a terminal state"

    if status == "failed":
        # Expected outcome without ultralytics/paddleocr/transformers/torch
        # -- confirm it failed CLEANLY, with
        # a specific message, not a raw unhandled crash.
        assert error_message, "a failed job must explain why"
        assert "Traceback" not in error_message, "job_queue should report str(exc), not a raw traceback"
    else:
        # status == "complete": every dependency was present and working.
        download = client.get(f"/jobs/{job_id}/download")
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"
