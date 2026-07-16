"""
Module B: Spatial Vision & Detection Engine (VRAM heavy)

Runs four independent analyses per key-state image, one heavy model at
a time via GPUPipelineGuard so at most one occupies VRAM at once:

  1. YOLOv10 (ultralytics)       -> UI element bounding boxes + label + confidence
  2. PaddleOCR                   -> text region bounding boxes + recognized strings
  3. Depth-Anything-V2 (HF)      -> per-pixel relative depth -> z_index per element
  4. Colorgram.py (CPU, no GPU)  -> dominant hex colors per element, from its crop

Architectural note -- no cross-referencing between YOLO and OCR here:
this module emits ONE FLAT LIST of DetectedElement, one per YOLO box and
one per OCR text region, with no attempt to decide that a given text
region "belongs to" a given UI element. That containment decision (is
this text box strictly inside that button's box, making it a child
node?) is Module C's job by design -- see the architecture doc's
description of the DOM Synthesizer. Doing it here too would duplicate
logic Module C already owns.

Verification status (read before trusting the numbers): none of
ultralytics / paddleocr / colorgram / transformers / torch are
installed in the sandbox this was written in, and it has no network
access to install them, so the actual model-loading and inference calls
below could not be executed against real weights. Every API shape used
(ultralytics' Results.boxes.xyxy/conf/cls, PaddleOCR 3.x's .predict()
returning objects with a .json dict, transformers' depth-estimation
pipeline, colorgram.extract()) was checked against current
documentation, not assumed from memory. What WAS verified for real in
this sandbox: every pure/glue function below (bbox conversion, depth
sampling and z-index normalization, hex formatting, PIL cropping, and
the PaddleOCR result parser against a hand-built stand-in matching the
documented shape) -- see tests/test_module_b.py. Two things worth
double-checking on first real run:
  - _parse_paddleocr_result()'s exact key path (res.json["res"][...]);
    if it raises, the error message tells you to print(res.json) and
    fix the key path for your installed version.
  - _normalize_z_indices()'s assumption that Depth-Anything-V2's larger
    values mean "closer to camera" -- flip the sign there if a real
    depth map shows the opposite for your setup.

Also worth flagging up front: the default YOLO checkpoint
(settings.yolo_weights_path) is stock, COCO-pretrained -- it detects
people/cars/dogs, not buttons or navbars. Real UI-element detection
needs a checkpoint fine-tuned on a UI dataset (e.g. Rico). This module
wires up the real inference and VRAM-gate plumbing regardless, so
swapping in that checkpoint later is a one-line config change, not a
rewrite.
"""
import asyncio
from typing import Optional

import numpy as np
from PIL import Image

from app.config import settings
from app.core.vram_manager import GPUPipelineGuard, clear_paddle_gpu_cache
from app.models.schemas import BoundingBox, DetectedElement, KeyStateFrame
from app.utils.logger import get_logger

logger = get_logger("omniui.module_b")

try:
    import colorgram

    _COLORGRAM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised whenever colorgram.py isn't installed
    _COLORGRAM_AVAILABLE = False
    logger.warning("colorgram not installed - color extraction will return empty lists.")


class PaddleOCRResultError(RuntimeError):
    """Raised when a PaddleOCR result doesn't match the expected 3.x shape."""


# ---------------------------------------------------------------------------
# Pure parsing / math helpers -- no ML libraries required, unit-tested with
# hand-built stand-ins that match the documented shapes (see tests/test_module_b.py)
# ---------------------------------------------------------------------------


def _xyxy_to_bbox_tuple(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def _parse_yolo_boxes(boxes, names: dict) -> list[dict]:
    """
    `boxes` is duck-typed as an ultralytics Boxes-like object: iterable,
    each item exposing .xyxy[0] (4 numbers), .cls[0], .conf[0]. `names`
    maps a class index to its label string (ultralytics' Results.names).
    """
    parsed = []
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])
        parsed.append(
            {
                "bbox": _xyxy_to_bbox_tuple(x1, y1, x2, y2),
                "label": names[cls_idx],
                "confidence": conf,
            }
        )
    return parsed


def _parse_paddleocr_result(res) -> list[dict]:
    """
    `res` can be:
      1. A PaddleOCR 3.x result object exposing `.json` (either `res.json["res"]` or `res.json` directly).
      2. A classic PaddleOCR list-of-lines format: `[[[box], [text, score]], ...]` where box is `[x1,y1,x2,y2]` or `[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]`.
      3. `None` if no text detected.
    """
    if res is None:
        return []

    # Case 1: Classic list-of-lines `[[box, [text, score]], ...]`
    if isinstance(res, (list, tuple)):
        parsed = []
        for item in res:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            box, (text, score) = item[0], item[1]
            if isinstance(box, (list, tuple)) and len(box) == 4:
                if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in box):
                    # 4 corner points [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    x1 = min(float(p[0]) for p in box)
                    y1 = min(float(p[1]) for p in box)
                    x2 = max(float(p[0]) for p in box)
                    y2 = max(float(p[1]) for p in box)
                elif all(isinstance(v, (int, float)) for v in box):
                    # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = [float(v) for v in box]
                else:
                    continue
                parsed.append(
                    {
                        "bbox": _xyxy_to_bbox_tuple(x1, y1, x2, y2),
                        "text": str(text),
                        "confidence": float(score),
                    }
                )
        return parsed

    # Case 2: PaddleOCR 3.x result object with `.json`
    try:
        payload = res.json.get("res", res.json) if isinstance(res.json, dict) else res.json
        texts = payload["rec_texts"]
        boxes = payload["rec_boxes"]  # [[x1, y1, x2, y2], ...] or 4-point polygons
        scores = payload["rec_scores"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise PaddleOCRResultError(
            "Unexpected PaddleOCR result shape -- run print(res.json) on one "
            "result from your installed version and update the key path in "
            "_parse_paddleocr_result() to match."
        ) from exc

    parsed = []
    for box, text, score in zip(boxes, texts, scores):
        if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in box):
            x1 = min(float(p[0]) for p in box)
            y1 = min(float(p[1]) for p in box)
            x2 = max(float(p[0]) for p in box)
            y2 = max(float(p[1]) for p in box)
        else:
            x1, y1, x2, y2 = [float(v) for v in box]
        parsed.append(
            {
                "bbox": _xyxy_to_bbox_tuple(x1, y1, x2, y2),
                "text": text,
                "confidence": float(score),
            }
        )
    return parsed


def _sample_depth_region(depth_array: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    """Mean depth value within a bbox region, clamped to the array's bounds."""
    x, y, w, h = bbox
    h_arr, w_arr = depth_array.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w_arr, int(x + w))
    y2 = min(h_arr, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return float(depth_array.mean())  # degenerate bbox: fall back to the image-wide average
    return float(depth_array[y1:y2, x1:x2].mean())


def _normalize_z_indices(raw_depths: list[float]) -> list[float]:
    """
    Min-max normalize raw depth samples to a 0-100 scale within one state
    image. Higher = closer to camera = higher CSS-style z-index, per
    Depth-Anything-V2's usual convention that larger predicted values
    mean nearer objects. This direction is worth confirming visually
    against a real model output; if it looks inverted for your setup,
    swap `(d - lo)` for `(hi - d)` below.
    """
    if not raw_depths:
        return []
    lo, hi = min(raw_depths), max(raw_depths)
    if hi - lo < 1e-9:
        return [50.0 for _ in raw_depths]  # everything at the same depth; park it mid-scale
    return [100.0 * (d - lo) / (hi - lo) for d in raw_depths]


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _extract_hex_colors(crop: Image.Image, num_colors: int) -> list[str]:
    """Dominant hex colors in a crop via colorgram.py. CPU-only, no GPU/VRAM involved."""
    if not _COLORGRAM_AVAILABLE or crop.width < 1 or crop.height < 1:
        return []
    request = max(1, min(num_colors, crop.width * crop.height))
    try:
        colors = colorgram.extract(crop, request)
    except Exception:
        logger.warning("colorgram.extract failed on a crop; skipping colors for this element.", exc_info=True)
        return []
    return [_rgb_to_hex(c.rgb.r, c.rgb.g, c.rgb.b) for c in colors]


def _cuda_is_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Model loading -- impure, needs the real libraries installed. Kept as small,
# separate functions so each one is a single clear place to swap checkpoints
# or add options later.
# ---------------------------------------------------------------------------


def _load_yolo(weights_path: str, device: str):
    from pathlib import Path
    from ultralytics import YOLO, settings as ul_settings

    project_root = Path(__file__).resolve().parent.parent.parent
    models_cache_dir = project_root / "models_cache"
    models_cache_dir.mkdir(parents=True, exist_ok=True)

    # Force ultralytics to use models_cache as its weights directory
    try:
        ul_settings.update({"weights_dir": str(models_cache_dir)})
    except Exception:
        pass

    # Ensure weights_path resolves strictly to the cached file in models_cache if present
    target_path = Path(weights_path)
    if not target_path.is_absolute() or not target_path.exists():
        cached_pt = models_cache_dir / target_path.name
        if cached_pt.exists():
            weights_path = str(cached_pt)

    return YOLO(weights_path).to(device)


def _load_paddleocr(device: str, lang: str):
    from paddleocr import PaddleOCR  # lazy import

    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        device=device,
    )


def _load_depth_pipeline(model_name: str, device: str):
    from transformers import pipeline  # lazy import

    if device == "cpu":
        device_index = -1
    elif ":" in device:
        device_index = int(device.split(":")[-1])
    else:
        device_index = 0

    try:
        # First attempt loading from local cache (offline mode) to avoid any network requests or timeouts
        return pipeline(
            task="depth-estimation",
            model=model_name,
            device=device_index,
            model_kwargs={"local_files_only": True},
        )
    except Exception:
        # Fall back to standard load if model has not been downloaded to cache yet
        return pipeline(task="depth-estimation", model=model_name, device=device_index)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run_paddleocr_subprocess(image_path: str, device: str, lang: str, confidence_threshold: float) -> list[dict]:
    """
    Runs PaddleOCR in a short-lived standalone subprocess on Windows to avoid WinError 127
    DLL symbol collisions when both PyTorch (YOLOv10) and PaddlePaddle (PaddleOCR)
    run in the same process address space, and to guarantee 100% VRAM cleanup
    upon OS-level process termination.
    """
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.modules.ocr_subprocess_worker",
            image_path,
            device,
            lang,
            str(confidence_threshold),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"PaddleOCR subprocess failed. stdout: {result.stdout}, stderr: {result.stderr}")
        return []

    stdout = result.stdout
    if "__OCR_JSON_START__" in stdout and "__OCR_JSON_END__" in stdout:
        json_str = stdout.split("__OCR_JSON_START__")[1].split("__OCR_JSON_END__")[0].strip()
        try:
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse OCR subprocess JSON: {e}")
            return []
    return []


def _analyze_state_sync(
    state: KeyStateFrame,
    yolo_weights_path: Optional[str] = None,
    yolo_confidence_threshold: Optional[float] = None,
    ocr_lang: Optional[str] = None,
    ocr_confidence_threshold: Optional[float] = None,
    depth_model_name: Optional[str] = None,
    colors_per_element: Optional[int] = None,
    cuda_device_index: Optional[int] = None,
) -> list[DetectedElement]:
    yolo_weights_path = yolo_weights_path or settings.yolo_weights_path
    yolo_confidence_threshold = (
        settings.yolo_confidence_threshold if yolo_confidence_threshold is None else yolo_confidence_threshold
    )
    ocr_lang = ocr_lang or settings.paddleocr_lang
    ocr_confidence_threshold = (
        settings.ocr_confidence_threshold if ocr_confidence_threshold is None else ocr_confidence_threshold
    )
    depth_model_name = depth_model_name or settings.depth_model_name
    colors_per_element = colors_per_element or settings.colorgram_colors_per_element
    cuda_device_index = settings.cuda_device_index if cuda_device_index is None else cuda_device_index

    torch_device = f"cuda:{cuda_device_index}" if _cuda_is_available() else "cpu"
    # PaddlePaddle uses its own device string convention ("gpu:N"), separate
    # from PyTorch's ("cuda:N") -- see clear_paddle_gpu_cache()'s docstring
    # for why these two frameworks need separate handling throughout.
    paddle_device = "cpu" if torch_device == "cpu" else f"gpu:{cuda_device_index}"

    image = Image.open(state.image_path).convert("RGB")
    guard = GPUPipelineGuard(device_index=cuda_device_index)

    with guard.stage(
        "yolov10",
        loader=lambda: _load_yolo(yolo_weights_path, torch_device),
        unloader=lambda m: m.to("cpu"),
        min_free_mb=1024,
    ) as model:
        results = model.predict(state.image_path, conf=yolo_confidence_threshold, verbose=False)
        yolo_detections = _parse_yolo_boxes(results[0].boxes, results[0].names)

    with guard.stage(
        "paddleocr",
        loader=lambda: "subprocess_ready",
        unloader=lambda _x: None,
        min_free_mb=1024,
    ):
        ocr_detections = _run_paddleocr_subprocess(state.image_path, paddle_device, ocr_lang, ocr_confidence_threshold)

    with guard.stage(
        "depth-anything-v2",
        loader=lambda: _load_depth_pipeline(depth_model_name, torch_device),
        unloader=lambda p: p.model.to("cpu"),
        min_free_mb=1536,
    ) as depth_pipe:
        depth_result = depth_pipe(image)
        depth_array = np.array(depth_result["depth"])

    # Combine: one DetectedElement per YOLO box, one per OCR text region.
    # No cross-referencing here -- see the module docstring.
    raw_items = [{"kind": "yolo", **d} for d in yolo_detections] + [
        {"kind": "ocr", **d} for d in ocr_detections
    ]
    raw_depths = [_sample_depth_region(depth_array, item["bbox"]) for item in raw_items]
    z_indices = _normalize_z_indices(raw_depths)

    elements: list[DetectedElement] = []
    for idx, (item, z) in enumerate(zip(raw_items, z_indices)):
        x, y, w, h = item["bbox"]
        crop = image.crop((int(x), int(y), int(x + w), int(y + h)))
        hex_colors = _extract_hex_colors(crop, colors_per_element)

        is_text = item["kind"] == "ocr"
        elements.append(
            DetectedElement(
                element_id=f"{item['kind']}-{idx:03d}",
                element_type="text" if is_text else item["label"],
                bbox=BoundingBox(x=x, y=y, width=w, height=h, z_index=round(z, 2)),
                text_content=item["text"] if is_text else None,
                hex_colors=hex_colors,
                confidence=round(item["confidence"], 4),
            )
        )

    logger.info(
        f"Module B: {state.image_path} -> {len(yolo_detections)} element(s), "
        f"{len(ocr_detections)} text region(s)"
    )
    return elements


async def analyze_state(state: KeyStateFrame, **kwargs) -> list[DetectedElement]:
    return await asyncio.to_thread(_analyze_state_sync, state, **kwargs)
