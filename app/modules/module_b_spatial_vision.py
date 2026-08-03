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
import json
import os
import sys

# Inject PyTorch's bundled CUDA/cuDNN DLLs into the Windows DLL search path and PATH.
# This allows onnxruntime-gpu to find `cudart64_12.dll` and `cudnn64_*.dll` without
# forcing the user to install the massive 3GB system-wide NVIDIA CUDA Toolkit.
_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_torch_lib)
    except (OSError, AttributeError):
        pass

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


def _load_and_export_yolo(weights_path: str, device: str, export_onnx: bool):
    from pathlib import Path
    from ultralytics import YOLO, settings as ul_settings

    project_root = Path(__file__).resolve().parent.parent.parent
    models_cache_dir = project_root / "models_cache"
    models_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        ul_settings.update({"weights_dir": str(models_cache_dir)})
    except Exception:
        pass

    target_path = Path(weights_path)
    if not target_path.is_absolute() or not target_path.exists():
        cached_pt = models_cache_dir / target_path.name
        if cached_pt.exists():
            weights_path = str(cached_pt)
            
    # Auto-export to ONNX for 3x-5x speedup if requested
    if export_onnx:
        onnx_path = Path(weights_path).with_suffix(".onnx")
        if not onnx_path.exists() and Path(weights_path).exists():
            logger.info(f"Exporting {weights_path} to ONNX format...")
            try:
                model = YOLO(weights_path)
                model.export(format="onnx")
            except Exception as e:
                logger.warning(f"Failed to export ONNX: {e}. Falling back to PyTorch.")
        if onnx_path.exists():
            logger.info(f"Loading ONNX engine: {onnx_path}")
            return YOLO(str(onnx_path), task="detect")

    logger.info(f"Loading PyTorch engine: {weights_path}")
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


import subprocess
import sys
import json
import threading

class PersistentOCRWorker:
    def __init__(self, max_jobs=4):
        self.process = None
        self.lock = threading.Lock()
        self.jobs_processed = 0
        self.max_jobs = max_jobs
        
    def start(self):
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "app.modules.ocr_subprocess_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            self.jobs_processed = 0
            
    def analyze(self, image_paths: list[str], lang: str, confidence_threshold: float) -> list[list[dict]]:
        with self.lock:
            self.jobs_processed += 1
            if self.jobs_processed > self.max_jobs:
                self.shutdown()
            self.start()
            
            payload = {
                "image_paths": image_paths,
                "lang": lang,
                "confidence_threshold": confidence_threshold
            }
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            
            output = ""
            while True:
                line = self.process.stdout.readline()
                if not line:
                    logger.error("OCR Subprocess died unexpectedly.")
                    self.process = None
                    return [[] for _ in image_paths]
                
                if "__OCR_PROGRESS__" in line:
                    prog = line.split("__OCR_PROGRESS__")[1].split("__")[0].strip()
                    logger.info(f"PaddleOCR progress: {prog} frames...")
                    continue

                output += line
                if "__OCR_JSON_END__" in output:
                    break
                    
            try:
                json_str = output.split("__OCR_JSON_START__")[1].split("__OCR_JSON_END__")[0].strip()
                result = json.loads(json_str)
                if result.get("status") == "error":
                    logger.error(f"OCR Subprocess Error: {result.get('message')}")
                    return [[] for _ in image_paths]
                return result.get("detections", [])
            except Exception as e:
                logger.error(f"Failed to parse OCR subprocess output: {e}")
                return [[] for _ in image_paths]
                
    def shutdown(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None

_global_ocr_worker = PersistentOCRWorker()


def _unload_yolo(m):
    try:
        if hasattr(m, "to"):
            m.to("cpu")
    except Exception:
        pass

def _analyze_states_sync(
    states: list[KeyStateFrame],
    yolo_weights_path: Optional[str] = None,
    yolo_confidence_threshold: Optional[float] = None,
    ocr_lang: Optional[str] = None,
    ocr_confidence_threshold: Optional[float] = None,
    depth_model_name: Optional[str] = None,
    colors_per_element: Optional[int] = None,
    cuda_device_index: Optional[int] = None,
    enable_3d: bool = False,
) -> list[list[DetectedElement]]:
    if not states:
        return []

    yolo_macro_weights_path = settings.yolo_macro_weights_path
    yolo_micro_weights_path = settings.yolo_micro_weights_path
    yolo_export_onnx = settings.yolo_export_onnx
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

    images = [Image.open(state.image_path).convert("RGB") for state in states]
    guard = GPUPipelineGuard(device_index=cuda_device_index)

    logger.info("Starting Hybrid YOLO (Macro + Micro) stage...")
    image_paths = [state.image_path for state in states]
    
    yolo_detections_list = [[] for _ in states]
    
    # Pass 1: YOLOv8 Macro-detector (Containers)
    with guard.stage(
        "yolov8_macro",
        loader=lambda: _load_and_export_yolo(yolo_macro_weights_path, torch_device, yolo_export_onnx),
        unloader=_unload_yolo,
        min_free_mb=1024,
    ) as macro_model:
        logger.info(f"YOLO Macro loaded. Running predict on {len(states)} images...")
        macro_results = [macro_model.predict(path, conf=yolo_confidence_threshold, verbose=False)[0] for path in image_paths]
        for i, res in enumerate(macro_results):
            boxes = _parse_yolo_boxes(res.boxes, res.names)
            # Add a tag to distinguish macro vs micro detections
            for box in boxes:
                box["level"] = "macro"
            yolo_detections_list[i].extend(boxes)
            
    # Pass 2: YOLOv10 Micro-detector (Dense Elements) on Cropped Containers
    with guard.stage(
        "yolov10_micro",
        loader=lambda: _load_and_export_yolo(yolo_micro_weights_path, torch_device, yolo_export_onnx),
        unloader=_unload_yolo,
        min_free_mb=1024,
    ) as micro_model:
        logger.info("YOLO Micro loaded. Processing nested crops...")
        for i, (state, image) in enumerate(zip(states, images)):
            macro_boxes = [d for d in yolo_detections_list[i] if d["level"] == "macro"]
            
            # If no macro containers found, process the full image anyway as a fallback
            if not macro_boxes:
                logger.info(f"No macro containers in {state.image_path}. Running micro on full image.")
                micro_res = micro_model.predict(state.image_path, conf=yolo_confidence_threshold, verbose=False)[0]
                micro_boxes = _parse_yolo_boxes(micro_res.boxes, micro_res.names)
                for mb in micro_boxes:
                    mb["level"] = "micro"
                yolo_detections_list[i].extend(micro_boxes)
                continue
                
            # Crop each macro container and run micro detection
            crops = []
            offsets = []
            for m_box in macro_boxes:
                x, y, w, h = m_box["bbox"]
                crop = image.crop((int(x), int(y), int(x + w), int(y + h)))
                crops.append(crop)
                offsets.append((x, y))
                
            if crops:
                logger.info(f"Running micro detector on {len(crops)} crops for {state.image_path}...")
                micro_results = [micro_model.predict(crop, conf=yolo_confidence_threshold, verbose=False)[0] for crop in crops]
                for crop_idx, m_res in enumerate(micro_results):
                    offset_x, offset_y = offsets[crop_idx]
                    local_boxes = _parse_yolo_boxes(m_res.boxes, m_res.names)
                    
                    # Global Re-projection
                    for local_box in local_boxes:
                        lx, ly, lw, lh = local_box["bbox"]
                        local_box["bbox"] = (lx + offset_x, ly + offset_y, lw, lh)
                        local_box["level"] = "micro"
                        yolo_detections_list[i].append(local_box)
                        
    logger.info("Hybrid YOLO stage complete.")

    logger.info("Starting PaddleOCR stage...")
    with guard.stage(
        "paddleocr",
        loader=lambda: "subprocess_ready",
        unloader=lambda _x: None,
        min_free_mb=1024,
    ):
        logger.info("Running PaddleOCR subprocess...")
        ocr_detections_list = _global_ocr_worker.analyze([s.image_path for s in states], ocr_lang, ocr_confidence_threshold)
        logger.info("PaddleOCR stage complete.")

    depth_array_list = []
    if enable_3d:
        logger.info("Starting Depth-Anything-V2 stage...")
        with guard.stage(
            "depth-anything-v2",
            loader=lambda: _load_depth_pipeline(depth_model_name, torch_device),
            unloader=lambda p: p.model.to("cpu"),
            min_free_mb=1536,
        ) as depth_pipe:
            logger.info(f"Depth-Anything-V2 loaded. Running predict in batch on {len(images)} images...")
            depth_results = depth_pipe(images, batch_size=4)
            if isinstance(depth_results, dict):
                depth_results = [depth_results] # Handle single image edge case
            depth_array_list = [np.array(res["depth"]) for res in depth_results]
            logger.info("Depth-Anything-V2 stage complete.")
    else:
        logger.info("Skipping Depth-Anything-V2 stage (enable_3d=False).")
        depth_array_list = [None for _ in images]

    all_elements = []
    for state, image, yolo_detections, ocr_detections, depth_array in zip(states, images, yolo_detections_list, ocr_detections_list, depth_array_list):
        raw_items = [{"kind": "yolo", **d} for d in yolo_detections] + [
            {"kind": "ocr", **d} for d in ocr_detections
        ]
        
        # Sort by confidence descending and cap to top 40 elements to prevent massive LLM context bloat
        # which causes VRAM exhaustion and PCIe thrashing during the 7B model prefill phase.
        raw_items.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        raw_items = raw_items[:40]
        
        if depth_array is not None:
            raw_depths = [_sample_depth_region(depth_array, item["bbox"]) for item in raw_items]
            z_indices = _normalize_z_indices(raw_depths)
        else:
            z_indices = [None for _ in raw_items]

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
                    bbox=BoundingBox(x=x, y=y, width=w, height=h, z_index=round(z, 2) if z is not None else None),
                    text_content=item["text"] if is_text else None,
                    hex_colors=hex_colors,
                    confidence=round(item["confidence"], 4),
                )
            )

        logger.info(
            f"Module B: {state.image_path} -> {len(yolo_detections)} YOLO, {len(ocr_detections)} OCR. Capped to {len(elements)} elements for LLM context."
        )
        all_elements.append(elements)

    return all_elements


async def analyze_states(states: list[KeyStateFrame], **kwargs) -> list[list[DetectedElement]]:
    return await asyncio.to_thread(_analyze_states_sync, states, **kwargs)

async def analyze_state(state: KeyStateFrame, **kwargs) -> list[DetectedElement]:
    res = await analyze_states([state], **kwargs)
    return res[0] if res else []
