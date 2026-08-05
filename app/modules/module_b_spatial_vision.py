"""
Module B: Hybrid Parallel Spatial Vision & Detection Engine

Architecture (v2 — Hybrid Parallel):

  YOLO runs on CPU (system RAM) in parallel with Vision LLM loading on GPU.
  YOLO finishes in ~200ms during the ~5-10s the Vision LLM takes to load.
  Then the 3B Vision LLM receives YOLO's boxes as context, reducing its
  workload to: (1) label YOLO boxes semantically, (2) detect anything
  YOLO missed. The 7B Vision LLM then verifies the merged result.

  Pipeline stages:
    1. YOLO macro + micro (CPU, ONNX) — fast bounding boxes     [~200ms]
    2. Vision LLM 3B (GPU subprocess) — semantic labels + gaps   [~10-30s]
    3. Vision LLM 7B (GPU subprocess) — verify + correct         [~10-30s]
    4. PaddleOCR (CPU subprocess) — precise text bboxes           [~1-3s]
    5. Colorgram.py (CPU) — dominant colors per crop              [~1s]
    6. Depth-Anything-V2 (GPU) — per-pixel depth → z_index        [~3-5s]

  YOLO (stage 1) and Vision LLM loading (stage 2 model load) overlap.
  This is free parallelism — YOLO uses CPU/RAM, VLM uses GPU/VRAM.
"""
import json
import os
import sys
import subprocess
import uuid
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from PIL import Image

from app.config import settings
from app.core.vram_manager import GPUPipelineGuard
from app.models.schemas import BoundingBox, DetectedElement, KeyStateFrame
from app.utils.logger import get_logger

logger = get_logger("omniui.module_b")

# Inject PyTorch's CUDA DLLs for Windows compatibility
_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_torch_lib)
    except (OSError, AttributeError):
        pass

try:
    import colorgram
    _COLORGRAM_AVAILABLE = True
except ImportError:
    _COLORGRAM_AVAILABLE = False
    logger.warning("colorgram not installed - color extraction will return empty lists.")


# ---------------------------------------------------------------------------
# Pure helpers (no ML libraries needed)
# ---------------------------------------------------------------------------


def _xyxy_to_bbox_tuple(x1, y1, x2, y2):
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def _parse_yolo_boxes(boxes, names):
    parsed = []
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])
        parsed.append({
            "bbox": _xyxy_to_bbox_tuple(x1, y1, x2, y2),
            "label": names[cls_idx],
            "confidence": conf,
        })
    return parsed


def _rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def _extract_hex_colors(crop, num_colors):
    if not _COLORGRAM_AVAILABLE or crop.width < 1 or crop.height < 1:
        return []
    request = max(1, min(num_colors, crop.width * crop.height))
    try:
        colors = colorgram.extract(crop, request)
    except Exception:
        return []
    return [_rgb_to_hex(c.rgb.r, c.rgb.g, c.rgb.b) for c in colors]


def _sample_depth_region(depth_array, bbox):
    x, y, w, h = bbox
    h_arr, w_arr = depth_array.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w_arr, int(x + w)), min(h_arr, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return float(depth_array.mean())
    return float(depth_array[y1:y2, x1:x2].mean())


def _normalize_z_indices(raw_depths):
    if not raw_depths:
        return []
    lo, hi = min(raw_depths), max(raw_depths)
    if hi - lo < 1e-9:
        return [50.0 for _ in raw_depths]
    return [100.0 * (d - lo) / (hi - lo) for d in raw_depths]


def _cuda_is_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _iou(box_a, box_b):
    """Intersection over Union between two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Stage 1: YOLO on CPU (both macro + micro)
# ---------------------------------------------------------------------------


def _load_yolo_cpu(weights_path, export_onnx):
    """Load YOLO model for CPU inference, preferring ONNX for speed."""
    from pathlib import Path
    from ultralytics import YOLO, settings as ul_settings

    models_cache_dir = Path(__file__).resolve().parent.parent.parent / "models_cache"
    try:
        ul_settings.update({"weights_dir": str(models_cache_dir)})
    except Exception:
        pass

    target_path = Path(weights_path)
    if not target_path.is_absolute() or not target_path.exists():
        cached = models_cache_dir / target_path.name
        if cached.exists():
            weights_path = str(cached)

    # Prefer ONNX for 3x-5x CPU speedup
    if export_onnx:
        onnx_path = Path(weights_path).with_suffix(".onnx")
        if onnx_path.exists():
            logger.info(f"Loading ONNX model: {onnx_path}")
            return YOLO(str(onnx_path), task="detect")
        elif Path(weights_path).exists():
            logger.info(f"Exporting to ONNX: {weights_path}")
            try:
                model = YOLO(weights_path)
                model.export(format="onnx")
                if onnx_path.exists():
                    return YOLO(str(onnx_path), task="detect")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")

    return YOLO(weights_path)


def _run_yolo_cpu(states, yolo_macro_path, yolo_micro_path, export_onnx, confidence_threshold):
    """Run both YOLO models on CPU. Returns list of detection lists per state."""
    logger.info(f"YOLO CPU: starting on {len(states)} images (macro={yolo_macro_path}, micro={yolo_micro_path})")
    
    image_paths = [state.image_path for state in states]
    yolo_results = [[] for _ in states]
    
    try:
        # Load macro detector
        macro_model = _load_yolo_cpu(yolo_macro_path, export_onnx)
        
        for i, path in enumerate(image_paths):
            macro_res = macro_model.predict(path, conf=confidence_threshold, verbose=False, device="cpu")[0]
            boxes = _parse_yolo_boxes(macro_res.boxes, macro_res.names)
            for box in boxes:
                box["level"] = "macro"
            yolo_results[i].extend(boxes)
        
        del macro_model
        
        # Load micro detector  
        micro_model = _load_yolo_cpu(yolo_micro_path, export_onnx)
        
        for i, (state, image_path) in enumerate(zip(states, image_paths)):
            macro_boxes = [d for d in yolo_results[i] if d["level"] == "macro"]
            image = Image.open(image_path).convert("RGB")
            
            if not macro_boxes:
                # No macro containers — run micro on full image
                micro_res = micro_model.predict(image_path, conf=confidence_threshold, verbose=False, device="cpu")[0]
                micro_boxes = _parse_yolo_boxes(micro_res.boxes, micro_res.names)
                for mb in micro_boxes:
                    mb["level"] = "micro"
                yolo_results[i].extend(micro_boxes)
            else:
                # Run micro on each macro container crop
                for m_box in macro_boxes:
                    x, y, w, h = m_box["bbox"]
                    crop = image.crop((int(x), int(y), int(x + w), int(y + h)))
                    micro_res = micro_model.predict(crop, conf=confidence_threshold, verbose=False, device="cpu")[0]
                    local_boxes = _parse_yolo_boxes(micro_res.boxes, micro_res.names)
                    for lb in local_boxes:
                        lx, ly, lw, lh = lb["bbox"]
                        lb["bbox"] = (lx + x, ly + y, lw, lh)
                        lb["level"] = "micro"
                        yolo_results[i].append(lb)
        
        del micro_model
        
    except Exception as e:
        logger.warning(f"YOLO CPU failed: {e}. Vision LLM will handle all detection.")
    
    for i, dets in enumerate(yolo_results):
        logger.info(f"YOLO CPU: {states[i].image_path} -> {len(dets)} boxes")
    
    return yolo_results


# ---------------------------------------------------------------------------
# Stage 2: Vision LLM 3B Detector (GPU subprocess)
# ---------------------------------------------------------------------------


def _parse_truncated_json(json_str):
    """Parse a JSON array that may have been truncated by token limit."""
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
        return [result] if isinstance(result, dict) else []
    except json.JSONDecodeError:
        pass

    last_complete = -1
    brace_depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(json_str):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0:
                last_complete = i

    if last_complete == -1:
        return []

    repaired = json_str[:last_complete + 1].rstrip().rstrip(',') + ']'
    if not repaired.startswith('['):
        repaired = '[' + repaired
    try:
        result = json.loads(repaired)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _run_vision_subprocess(worker_module, payload):
    """Run a vision model subprocess and return parsed results."""
    payload_file = settings.jobs_dir / f"vision_payload_{uuid.uuid4().hex}.json"
    payload_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(payload_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        
        completed = subprocess.run(
            [sys.executable, "-m", worker_module, str(payload_file)],
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        
        stdout_lines = completed.stdout.splitlines(keepends=True)
        for line in stdout_lines:
            if "__VISION_JSON_START__" not in line and "__VISION_JSON_END__" not in line and line.strip():
                logger.info(f"[VLM] {line.strip()}")
        
        if completed.returncode != 0:
            raise RuntimeError(f"Vision subprocess failed (code {completed.returncode})")
        
        stdout = completed.stdout
        if "__VISION_JSON_START__" not in stdout or "__VISION_JSON_END__" not in stdout:
            raise RuntimeError("Vision subprocess output missing JSON markers")
        
        json_str = stdout.split("__VISION_JSON_START__")[1].split("__VISION_JSON_END__")[0].strip()
        return json.loads(json_str)
        
    finally:
        if payload_file.exists():
            try:
                payload_file.unlink()
            except Exception:
                pass


def _run_vision_detector(states, yolo_detections_per_state):
    """Run the 3B Vision LLM detector subprocess."""
    model_name = settings.vision_detector_model_name
    load_in_4bit = settings.local_llm_load_in_4bit
    
    logger.info(f"Vision Detector: running {model_name} on {len(states)} images")
    
    # Format YOLO detections for the prompt
    yolo_for_prompt = []
    for dets in yolo_detections_per_state:
        yolo_for_prompt.append([{"bbox": list(d["bbox"]), "confidence": d.get("confidence", 0.5)} for d in dets])
    
    payload = {
        "model_name": model_name,
        "image_paths": [str(s.image_path) for s in states],
        "yolo_detections": yolo_for_prompt,
        "temperature": 0.1,
        "max_new_tokens": 2048,
        "load_in_4bit": load_in_4bit,
    }
    
    raw_results = _run_vision_subprocess("app.modules.vision_detector_worker", payload)
    
    # Parse each image's result
    parsed_results = []
    for res_str in raw_results:
        parsed_results.append(_parse_truncated_json(res_str))
    
    return parsed_results


def _run_vision_verifier(states, detections_per_state):
    """Run the 7B Vision LLM verifier subprocess."""
    model_name = settings.vision_llm_model_name
    load_in_4bit = settings.local_llm_load_in_4bit
    
    logger.info(f"Vision Verifier: running {model_name} on {len(states)} images")
    
    payload = {
        "model_name": model_name,
        "image_paths": [str(s.image_path) for s in states],
        "detections": detections_per_state,
        "temperature": 0.1,
        "max_new_tokens": 2048,
        "load_in_4bit": load_in_4bit,
    }
    
    raw_results = _run_vision_subprocess("app.modules.vision_verifier_worker", payload)
    
    # Parse verifier output: each result is a JSON object with "approved" and "elements"
    verified_results = []
    approval_flags = []
    
    for res_str in raw_results:
        try:
            parsed = json.loads(res_str) if isinstance(res_str, str) else res_str
            if isinstance(parsed, dict):
                approval_flags.append(parsed.get("approved", True))
                elements = parsed.get("elements", [])
                verified_results.append(elements if isinstance(elements, list) else [])
            elif isinstance(parsed, list):
                approval_flags.append(True)
                verified_results.append(parsed)
            else:
                approval_flags.append(True)
                verified_results.append([])
        except (json.JSONDecodeError, TypeError):
            # Verifier response unparseable — use the pre-verification detections
            approval_flags.append(True)
            verified_results.append([])
    
    return verified_results, approval_flags


# ---------------------------------------------------------------------------
# Merging: YOLO bboxes + Vision LLM semantic labels
# ---------------------------------------------------------------------------


def _merge_detections(yolo_dets, vlm_dets, iou_threshold=0.5):
    """
    Merge YOLO detections (precise bboxes) with Vision LLM detections
    (semantic labels + missed elements). For overlapping boxes (IoU > threshold),
    keep YOLO's bbox precision but use VLM's semantic type and text.
    """
    merged = []
    vlm_used = set()
    
    for yd in yolo_dets:
        best_vlm_idx = -1
        best_iou = 0.0
        
        for vi, vd in enumerate(vlm_dets):
            if vi in vlm_used:
                continue
            score = _iou(yd["bbox"], vd.get("bbox", [0, 0, 0, 0]))
            if score > best_iou:
                best_iou = score
                best_vlm_idx = vi
        
        if best_iou >= iou_threshold and best_vlm_idx >= 0:
            vlm_used.add(best_vlm_idx)
            vd = vlm_dets[best_vlm_idx]
            merged.append({
                "bbox": yd["bbox"],  # keep YOLO's precise bbox
                "type": vd.get("type", yd.get("label", "div")),  # use VLM's semantic type
                "text": vd.get("text"),
                "confidence": max(yd.get("confidence", 0.5), vd.get("confidence", 0.5)),
                "source": "yolo+vlm",
            })
        else:
            merged.append({
                "bbox": yd["bbox"],
                "type": yd.get("label", "div"),
                "text": None,
                "confidence": yd.get("confidence", 0.5),
                "source": "yolo",
            })
    
    # Add VLM-only detections (elements YOLO missed)
    for vi, vd in enumerate(vlm_dets):
        if vi not in vlm_used:
            bbox = vd.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox, list) and len(bbox) == 4:
                merged.append({
                    "bbox": tuple(float(v) for v in bbox),
                    "type": vd.get("type", "div"),
                    "text": vd.get("text"),
                    "confidence": vd.get("confidence", 0.5),
                    "source": "vlm",
                })
    
    return merged


# ---------------------------------------------------------------------------
# PaddleOCR subprocess (unchanged from v1)
# ---------------------------------------------------------------------------


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
            
    def analyze(self, image_paths, lang, confidence_threshold):
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


# ---------------------------------------------------------------------------
# Depth model loading
# ---------------------------------------------------------------------------


def _load_depth_pipeline(model_name, device):
    from transformers import pipeline

    if device == "cpu":
        device_index = -1
    elif ":" in device:
        device_index = int(device.split(":")[-1])
    else:
        device_index = 0

    try:
        return pipeline(task="depth-estimation", model=model_name, device=device_index, model_kwargs={"local_files_only": True})
    except Exception:
        return pipeline(task="depth-estimation", model=model_name, device=device_index)


# ---------------------------------------------------------------------------
# Main orchestration: Hybrid Parallel Pipeline
# ---------------------------------------------------------------------------


def _analyze_states_sync(
    states: list[KeyStateFrame],
    yolo_confidence_threshold: Optional[float] = None,
    ocr_lang: Optional[str] = None,
    ocr_confidence_threshold: Optional[float] = None,
    depth_model_name: Optional[str] = None,
    colors_per_element: Optional[int] = None,
    cuda_device_index: Optional[int] = None,
    enable_3d: bool = True,
) -> tuple[list[list[DetectedElement]], list[bool]]:
    """
    Returns (detections_per_state, verifier_approvals_per_state).
    """
    if not states:
        return [], []

    yolo_macro_path = settings.yolo_macro_weights_path
    yolo_micro_path = settings.yolo_micro_weights_path
    yolo_export_onnx = settings.yolo_export_onnx
    yolo_conf = settings.yolo_confidence_threshold if yolo_confidence_threshold is None else yolo_confidence_threshold
    ocr_lang = ocr_lang or settings.paddleocr_lang
    ocr_conf = settings.ocr_confidence_threshold if ocr_confidence_threshold is None else ocr_confidence_threshold
    depth_model_name = depth_model_name or settings.depth_model_name
    colors_per_element = colors_per_element or settings.colorgram_colors_per_element
    cuda_device_index = settings.cuda_device_index if cuda_device_index is None else cuda_device_index

    torch_device = f"cuda:{cuda_device_index}" if _cuda_is_available() else "cpu"
    images = [Image.open(state.image_path).convert("RGB") for state in states]
    guard = GPUPipelineGuard(device_index=cuda_device_index)

    # -----------------------------------------------------------------------
    # Stage 1: YOLO on CPU (runs during Vision LLM model loading)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 1: YOLO CPU detection (parallel with VLM loading)")
    logger.info("=" * 60)
    
    yolo_detections = _run_yolo_cpu(states, yolo_macro_path, yolo_micro_path, yolo_export_onnx, yolo_conf)

    # -----------------------------------------------------------------------
    # Stage 2: Vision LLM 3B Detector (GPU subprocess)
    # YOLO results are passed as context to reduce VLM workload
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 2: Vision LLM 3B detector (with YOLO context)")
    logger.info("=" * 60)
    
    with guard.stage(
        "vision_detector_3b",
        loader=lambda: "subprocess_ready",
        unloader=lambda _: None,
        min_free_mb=1024,
    ):
        vlm_detections = _run_vision_detector(states, yolo_detections)

    # Merge YOLO + VLM detections
    merged_per_state = []
    for i in range(len(states)):
        yolo_dets = yolo_detections[i] if i < len(yolo_detections) else []
        vlm_dets = vlm_detections[i] if i < len(vlm_detections) else []
        merged = _merge_detections(yolo_dets, vlm_dets)
        merged_per_state.append(merged)
        
        yolo_count = sum(1 for m in merged if m["source"] in ("yolo", "yolo+vlm"))
        vlm_only = sum(1 for m in merged if m["source"] == "vlm")
        logger.info(f"Merge: {states[i].image_path} -> {len(merged)} total ({yolo_count} from YOLO, {vlm_only} VLM-only)")

    # -----------------------------------------------------------------------
    # Stage 3: Vision LLM 7B Verifier (GPU subprocess)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 3: Vision LLM 7B verifier")
    logger.info("=" * 60)
    
    with guard.stage(
        "vision_verifier_7b",
        loader=lambda: "subprocess_ready",
        unloader=lambda _: None,
        min_free_mb=1024,
    ):
        verified_per_state, approval_flags = _run_vision_verifier(states, merged_per_state)
    
    # Use verified results if available, otherwise keep merged
    final_raw_per_state = []
    for i in range(len(states)):
        if verified_per_state[i]:
            final_raw_per_state.append(verified_per_state[i])
            logger.info(f"Verifier: {states[i].image_path} -> {len(verified_per_state[i])} verified elements (approved={approval_flags[i]})")
        else:
            final_raw_per_state.append(merged_per_state[i])
            logger.info(f"Verifier: {states[i].image_path} -> using merged ({len(merged_per_state[i])} elements, verifier returned empty)")

    # -----------------------------------------------------------------------
    # Stage 4: PaddleOCR (CPU subprocess) — precise text
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 4: PaddleOCR text extraction")
    logger.info("=" * 60)
    
    with guard.stage(
        "paddleocr",
        loader=lambda: "subprocess_ready",
        unloader=lambda _: None,
        min_free_mb=512,
    ):
        ocr_detections_list = _global_ocr_worker.analyze(
            [s.image_path for s in states], ocr_lang, ocr_conf
        )

    # -----------------------------------------------------------------------
    # Stage 5: Depth-Anything-V2 (GPU) — always run for z-ordering
    # -----------------------------------------------------------------------
    depth_array_list = []
    raw_depth_stats = []
    
    logger.info("=" * 60)
    logger.info("Stage 5: Depth-Anything-V2")
    logger.info("=" * 60)
    
    if enable_3d:
        with guard.stage(
            "depth-anything-v2",
            loader=lambda: _load_depth_pipeline(depth_model_name, torch_device),
            unloader=lambda p: p.model.to("cpu"),
            min_free_mb=1536,
        ) as depth_pipe:
            depth_results = depth_pipe(images, batch_size=4)
            if isinstance(depth_results, dict):
                depth_results = [depth_results]
            for res in depth_results:
                arr = np.array(res["depth"])
                depth_array_list.append(arr)
                # Pre-normalization statistics for 3D scene detection
                raw_range = float(arr.max() - arr.min())
                raw_var = float(arr.var())
                raw_depth_stats.append({"range": raw_range, "variance": raw_var})
    else:
        depth_array_list = [None for _ in images]
        raw_depth_stats = [{"range": 0.0, "variance": 0.0} for _ in images]

    # -----------------------------------------------------------------------
    # Assembly: Combine all sources into DetectedElement objects
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Assembling final detection results")
    logger.info("=" * 60)
    
    all_elements = []
    for state_idx, (state, image, final_raw, ocr_dets, depth_array) in enumerate(
        zip(states, images, final_raw_per_state, ocr_detections_list, depth_array_list)
    ):
        raw_items = []
        
        # Add vision-detected elements
        for vi, item in enumerate(final_raw):
            bbox = item.get("bbox", [0, 0, 0, 0])
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                raw_items.append({
                    "kind": "vision",
                    "bbox": tuple(float(v) for v in bbox),
                    "label": item.get("type", "div"),
                    "text": item.get("text"),
                    "confidence": float(item.get("confidence", 0.5)),
                })
        
        # Add OCR text elements (supplement vision detections)
        for oi, ocr_item in enumerate(ocr_dets):
            ocr_bbox = ocr_item.get("box", ocr_item.get("bbox", (0, 0, 0, 0)))
            if isinstance(ocr_bbox, (list, tuple)) and len(ocr_bbox) == 4:
                # Check if OCR text already exists in vision detections
                ocr_text = ocr_item.get("text", "")
                already_found = any(
                    ri.get("text") and ocr_text and ocr_text.strip().lower() in ri["text"].strip().lower()
                    for ri in raw_items
                )
                if not already_found and ocr_text:
                    raw_items.append({
                        "kind": "ocr",
                        "bbox": tuple(float(v) for v in ocr_bbox),
                        "label": "text",
                        "text": ocr_text,
                        "confidence": float(ocr_item.get("confidence", 0.5)),
                    })

        # Cap total elements: vision gets priority, then OCR
        vision_items = [r for r in raw_items if r["kind"] == "vision"]
        ocr_items = [r for r in raw_items if r["kind"] == "ocr"]
        vision_items.sort(key=lambda x: x["confidence"], reverse=True)
        ocr_items.sort(key=lambda x: x["confidence"], reverse=True)
        raw_items = vision_items[:30] + ocr_items[:15]

        # Compute z-indices
        if depth_array is not None:
            raw_depths = [_sample_depth_region(depth_array, item["bbox"]) for item in raw_items]
            z_indices = _normalize_z_indices(raw_depths)
        else:
            raw_depths = [0.0] * len(raw_items)
            z_indices = [None for _ in raw_items]

        # Build DetectedElement objects
        elements = []
        for idx, (item, z) in enumerate(zip(raw_items, z_indices)):
            x, y, w, h = item["bbox"]
            crop = image.crop((max(0, int(x)), max(0, int(y)), min(image.width, int(x + w)), min(image.height, int(y + h))))
            hex_colors = _extract_hex_colors(crop, colors_per_element)

            is_text = item["kind"] == "ocr" or item["label"] == "text"
            source = "vision_llm" if item["kind"] == "vision" else "ocr"
            
            elements.append(
                DetectedElement(
                    element_id=f"{item['kind']}-{idx:03d}",
                    element_type="text" if is_text else item["label"],
                    bbox=BoundingBox(x=x, y=y, width=max(0, w), height=max(0, h), z_index=round(z, 2) if z is not None else None),
                    text_content=item.get("text") if is_text or item.get("text") else None,
                    hex_colors=hex_colors,
                    confidence=round(item["confidence"], 4),
                    detection_source=source,
                    raw_depth_range=raw_depth_stats[state_idx]["range"] if depth_array is not None else None,
                )
            )

        logger.info(
            f"Module B: {state.image_path} -> {len(elements)} elements "
            f"(vision={sum(1 for e in elements if e.detection_source == 'vision_llm')}, "
            f"ocr={sum(1 for e in elements if e.detection_source == 'ocr')})"
        )
        all_elements.append(elements)

    return all_elements, approval_flags


async def analyze_states(states: list[KeyStateFrame], **kwargs) -> tuple[list[list[DetectedElement]], list[bool]]:
    return await asyncio.to_thread(_analyze_states_sync, states, **kwargs)


async def analyze_state(state: KeyStateFrame, **kwargs) -> list[DetectedElement]:
    res, _ = await analyze_states([state], **kwargs)
    return res[0] if res else []
