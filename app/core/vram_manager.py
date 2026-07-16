"""
VRAM lifecycle manager.

Ensures that at most one heavy in-process model (YOLOv10, PaddleOCR-GPU,
Depth-Anything-V2, ...) occupies GPU memory at any moment. This is the
core safety mechanism preventing OOM crashes on consumer GPUs (8-12GB)
when Module B runs several detection models back-to-back.

Multi-framework note: YOLOv10 and Depth-Anything-V2 are PyTorch models,
but PaddleOCR runs on PaddlePaddle -- a separate framework with its own
CUDA context and memory pool that torch.cuda.empty_cache() cannot touch.
See clear_paddle_gpu_cache() below; Module B's PaddleOCR stage calls it
via its `unloader`, the same hook PyTorch models use for their own
cleanup, so the gate is genuinely one-model-at-a-time regardless of
which framework that model happens to be built on.

The local LLM (Module D) runs in an isolated subprocess that loads and
unloads its own model weights, so it is NOT managed here. VRAM cleanup
for the LLM happens at the OS level when the subprocess exits.
"""

import gc
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

logger = logging.getLogger("omniui.vram")

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception as e:
    _TORCH_AVAILABLE = False
    logger.warning(f"torch not installed or unavailable ({e}) - VRAM manager running in no-op mode.")


def clear_paddle_gpu_cache() -> None:
    """
    PaddleOCR (Module B) runs on PaddlePaddle, a completely separate
    framework from PyTorch with its own CUDA context and memory pool.
    torch.cuda.empty_cache() has no effect on Paddle's allocations --
    this is the Paddle-side equivalent, and it's what actually needs to
    run after a PaddleOCR stage for the VRAM gate to do anything there.

    Known caveat: some users report paddle.device.cuda.empty_cache() not
    fully reclaiming memory after repeated inference, likely due to
    internal state the predictor keeps regardless of the cache call.
    Treat this as best-effort, not a guarantee -- if VRAM creeps up over
    many PaddleOCR stages in production, running it in a short-lived
    subprocess (so OS-level teardown forces a full reclaim) is the more
    reliable fallback.
    """
    try:
        import paddle

        paddle.device.cuda.empty_cache()
    except Exception:
        logger.debug("paddle.device.cuda.empty_cache() unavailable or failed; skipping.", exc_info=True)


class VRAMBudgetExceeded(RuntimeError):
    """Raised when free VRAM looks too low for a model to load safely.

    This is a tripwire, not a precise memory predictor: it mainly catches
    "a previous model wasn't actually unloaded" bugs, not exact
    per-model footprint planning. Tune `min_free_mb` per model once real
    weights are wired in (e.g. Depth-Anything-V2-Large needs more
    headroom than YOLOv10n).
    """


class VRAMSnapshot:
    """Point-in-time GPU memory snapshot, used for logging and budget checks."""

    __slots__ = ("allocated_mb", "reserved_mb", "free_mb", "total_mb")

    def __init__(self, allocated_mb: float, reserved_mb: float, free_mb: float, total_mb: float):
        self.allocated_mb = allocated_mb
        self.reserved_mb = reserved_mb
        self.free_mb = free_mb
        self.total_mb = total_mb

    def __str__(self) -> str:
        return (
            f"allocated={self.allocated_mb:.0f}MB "
            f"reserved={self.reserved_mb:.0f}MB "
            f"free={self.free_mb:.0f}/{self.total_mb:.0f}MB"
        )


def _snapshot(device_index: int = 0) -> Optional[VRAMSnapshot]:
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    free_b, total_b = torch.cuda.mem_get_info(device_index)
    mb = 1024**2
    return VRAMSnapshot(
        allocated_mb=torch.cuda.memory_allocated(device_index) / mb,
        reserved_mb=torch.cuda.memory_reserved(device_index) / mb,
        free_mb=free_b / mb,
        total_mb=total_b / mb,
    )


def gpu_status(device_index: int = 0) -> dict:
    """Small summary consumed by the /health endpoint."""
    if not _TORCH_AVAILABLE:
        return {"torch_installed": False, "cuda_available": False}
    if not torch.cuda.is_available():
        return {"torch_installed": True, "cuda_available": False}
    snap = _snapshot(device_index)
    return {
        "torch_installed": True,
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(device_index),
        "memory": str(snap) if snap else None,
    }


class ManagedModel:
    """Wraps one heavy model's lifecycle: budget check, lazy load, guaranteed unload."""

    def __init__(
        self,
        name: str,
        loader: Callable[[], Any],
        unloader: Optional[Callable[[Any], None]] = None,
        min_free_mb: float = 512.0,
        device_index: int = 0,
    ):
        self.name = name
        self._loader = loader
        self._unloader = unloader
        self.min_free_mb = min_free_mb
        self.device_index = device_index
        self._model: Optional[Any] = None

    def _check_budget(self) -> None:
        snap = _snapshot(self.device_index)
        if snap is None:
            return  # CPU-only environment: nothing to enforce
        if snap.free_mb < self.min_free_mb:
            raise VRAMBudgetExceeded(
                f"[{self.name}] only {snap.free_mb:.0f}MB free, need >= "
                f"{self.min_free_mb:.0f}MB. Is a previous model still resident? ({snap})"
            )

    def load(self) -> Any:
        logger.info(f"[{self.name}] loading... ({_snapshot(self.device_index)})")
        self._check_budget()
        t0 = time.perf_counter()
        self._model = self._loader()
        dt = time.perf_counter() - t0
        logger.info(f"[{self.name}] loaded in {dt:.2f}s ({_snapshot(self.device_index)})")
        return self._model

    def unload(self) -> None:
        if self._model is None:
            return
        logger.info(f"[{self.name}] unloading...")
        try:
            if self._unloader is not None:
                self._unloader(self._model)
        finally:
            self._model = None
            gc.collect()
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info(f"[{self.name}] unloaded, cache cleared ({_snapshot(self.device_index)})")


@contextmanager
def vram_scope(
    name: str,
    loader: Callable[[], Any],
    unloader: Optional[Callable[[Any], None]] = None,
    min_free_mb: float = 512.0,
    device_index: int = 0,
) -> Generator[Any, None, None]:
    """
    Strict single-model VRAM scope.

        with vram_scope("yolov10", loader=lambda: YOLO(weights).to("cuda")) as model:
            results = model.predict(frame)
        # model is guaranteed unloaded and VRAM cache cleared here,
        # even if predict() raised.
    """
    managed = ManagedModel(name, loader, unloader, min_free_mb, device_index)
    model = managed.load()
    try:
        yield model
    finally:
        managed.unload()


class GPUPipelineGuard:
    """
    Sequences multiple vram_scope stages and defensively refuses to let
    two be active at once. Module B uses one instance to run
    YOLOv10 -> PaddleOCR -> Depth-Anything-V2 strictly one at a time.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._active_stage: Optional[str] = None

    def stage(
        self,
        name: str,
        loader: Callable[[], Any],
        unloader: Optional[Callable[[Any], None]] = None,
        min_free_mb: float = 512.0,
    ):
        if self._active_stage is not None:
            raise RuntimeError(
                f"Cannot start stage '{name}' while '{self._active_stage}' is still "
                f"active. Stages must run strictly sequentially."
            )
        return self._guarded_scope(name, loader, unloader, min_free_mb)

    @contextmanager
    def _guarded_scope(self, name, loader, unloader, min_free_mb):
        self._active_stage = name
        try:
            with vram_scope(name, loader, unloader, min_free_mb, self.device_index) as model:
                yield model
        finally:
            self._active_stage = None
