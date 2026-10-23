"""Bounded worker execution shared by inference subprocesses."""
import os
import subprocess
import threading
import uuid
from app.config import settings

_active = set()
_lock = threading.Lock()
_stopping = threading.Event()


def resume_workers():
    _stopping.clear()


def shutdown_workers():
    _stopping.set()
    with _lock:
        for process in list(_active):
            if process.poll() is None:
                process.kill()


def run_worker(command, *, input=None, timeout=600):
    log_dir = settings.jobs_dir / "_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{uuid.uuid4().hex}.log"
    log = log_path.open("w", encoding="utf-8")
    with _lock:
        if _stopping.is_set():
            log.close()
            raise RuntimeError("Inference is shutting down")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING":"utf-8"})
        _active.add(process)
    try:
        stdout, _ = process.communicate(input, timeout=timeout)
        log.close()
        stderr = log_path.read_text(encoding="utf-8", errors="replace")
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError(f"Inference worker exceeded {timeout}s deadline")
    finally:
        log.close()
        with _lock:
            _active.discard(process)
