"""Sequential local UI detection, spatial OCR fusion, strict geometry validation, measured styles and explicit degradation diagnostics."""
import json
import os
import sys
import subprocess
import uuid
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.core.process_runner import run_worker
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
    from pathlib import Path
    from ultralytics import YOLO
    target = Path(weights_path)
    if not target.is_file():
        raise FileNotFoundError(f"UI checkpoint not found: {target}")
    model = YOLO(str(target),task="detect")
    model.overrides["device"] = "cpu"
    _require_ui_taxonomy(model)
    if export_onnx and target.suffix == ".pt":
        onnx = target.with_suffix(".onnx")
        if not onnx.exists():
            model.export(format="onnx",device="cpu")
        model = YOLO(str(onnx),task="detect")
        model.overrides["device"] = "cpu"
    return model


def _require_ui_taxonomy(model):
    names = getattr(model,"names",{})
    labels = {str(v).lower().replace(" ","_") for v in (names.values() if isinstance(names,dict) else names)}
    if not labels.intersection({"button","text","input","text_button","icon","container","card","image_view"}):
        raise ValueError("YOLO checkpoint has no UI taxonomy; configure UI-trained weights (COCO weights are unsupported)")


def _run_yolo_cpu(states, yolo_macro_path, yolo_micro_path, export_onnx, confidence_threshold):
    """Each optional UI checkpoint covers the full image independently."""
    results = [[] for _ in states]
    for level, weights in (("macro",yolo_macro_path),("micro",yolo_micro_path)):
        try:
            model = _load_yolo_cpu(weights,export_onnx)
        except Exception as exc:
            logger.warning("Skipping %s detector: %s",level,exc)
            continue
        try:
            for index,state in enumerate(states):
                result = model.predict(state.image_path,conf=confidence_threshold,verbose=False,device="cpu")[0]
                boxes = _parse_yolo_boxes(result.boxes,result.names)
                results[index].extend({**box,"level":level} for box in boxes)
        finally:
            del model
    return results


# ---------------------------------------------------------------------------
# Stage 2: Vision LLM 7B Detector (GPU subprocess)
# ---------------------------------------------------------------------------


def _parse_truncated_json(json_str):
    # A partial array is incomplete detection, never a successful extraction.
    result = json.loads(json_str)
    if not isinstance(result,list) or not all(isinstance(item,dict) for item in result):
        raise ValueError("Vision detector must return a complete JSON array")
    return result


def _run_vision_subprocess(worker_module, payload):
    """Run a vision model subprocess and return parsed results."""
    payload_file = settings.jobs_dir / f"vision_payload_{uuid.uuid4().hex}.json"
    payload_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(payload_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        
        completed = run_worker([sys.executable, "-m", worker_module, str(payload_file)],
                               timeout=settings.worker_timeout_sec)

        stdout_lines = completed.stdout.splitlines(keepends=True)
        for line in stdout_lines:
            if "__VISION_JSON_START__" not in line and "__VISION_JSON_END__" not in line and line.strip():
                logger.debug(f"[VLM] {line.strip()[:500]}")
        
        if completed.returncode != 0:
            raise RuntimeError(f"Vision subprocess failed (code {completed.returncode}): {completed.stderr[-1500:]}")
        
        stdout = completed.stdout
        if "__VISION_JSON_START__" not in stdout or "__VISION_JSON_END__" not in stdout:
            raise RuntimeError("Vision subprocess output missing JSON markers")
        
        json_str = stdout.split("__VISION_JSON_START__")[1].split("__VISION_JSON_END__")[0].strip()
        result = json.loads(json_str)
        if payload.get("image_paths"):
            from pathlib import Path
            target = Path(payload["image_paths"][0]).parent / (worker_module.rsplit(".",1)[-1] + "-raw.json")
            target.write_text(json.dumps(result,indent=2),encoding="utf-8")
        return result

    finally:
        if payload_file.exists():
            try:
                payload_file.unlink()
            except Exception:
                pass


def _supported_role(kind, anchor):
    """Reject unsupported control guesses for OCR-only anchors.

    Trained UI boxes may be unlabeled. Text-only regions need a plausible field
    or command cue; decorative labels must not become dozens of fake inputs.
    """
    import re
    from difflib import SequenceMatcher
    if anchor.get("source") != "ocr":
        return kind
    text = str(anchor.get("text") or "").strip().casefold()
    input_cue = bool(re.match(r"^(paste|provide|enter|type|search|email|e-mail|password|username|name|query|message|address|phone)\b",text))
    if kind == "input" and not input_cue:
        return "text"
    if kind in {"checkbox","radio","switch"} and not re.match(r"^(remember|agree|accept|enable|disable|subscribe|select|show|hide|allow|notify|notifications)\b",text):
        return "text"
    if kind == "slider":
        return "text"  # OCR labels cannot establish the track/thumb geometry.
    if kind == 'dropdown' and not re.match(r'^(select|choose|sort|filter|language|country|category|mode|voice)\b',text):
        return 'text'  # A noun alone does not establish a selector/arrow control.
    words = re.findall(r"[a-z]+",text)
    verbs = {"save","submit","next","back","cancel","close","attach","browse","upload","download",
             "execute","run","start","stop","continue","sign","log","login","create","delete","remove",
             "send","search","open","apply","confirm","yes","no","voice","scan","inspect","analyze","analyse","verify","check"}
    command = bool(words) and (words[0] in verbs or
        any(abs(len(words[0])-len(verb))<=2 and SequenceMatcher(None,words[0],verb).ratio() >= .70 for verb in verbs if len(verb)>4))
    if kind in {"button","link","tab"} and not command:
        return "text"
    # Obvious placeholder/imperative cues remain usable if a weak model chooses
    # 'static'. These are explicit heuristics, not verified application semantics.
    if kind == "text" and re.match(r"^(enter|type)\b",text):
        return "input"
    if kind == "text" and command:
        return "button"
    return kind


def _run_vision_detector(states, yolo_detections_per_state, ocr_detections_per_state=None):
    """Classify grounded anchors, never ask the small VLM to invent geometry/text."""
    ocr_detections_per_state = ocr_detections_per_state or [[] for _ in states]
    anchors_per_state = []
    for yolo, ocr in zip(yolo_detections_per_state,ocr_detections_per_state):
        anchors = [{**item,"type":item.get("label","container"),"text":None,"source":"yolo"} for item in yolo]
        anchors.extend({"bbox":item.get("bbox",item.get("box")),"text":item.get("text"),
                        "confidence":item.get("confidence",.5),"type":"text","source":"ocr"} for item in ocr)
        anchors_per_state.append(anchors)
    if not any(anchors_per_state):
        return [[] for _ in states]
    payload = {"model_name":settings.vision_detector_model_name,
        "image_paths":[str(s.image_path) for s in states],"anchors":anchors_per_state,
        "temperature":0.0,"max_new_tokens":settings.vision_max_new_tokens,
        "load_in_4bit":True,"metrics_path":str(Path(states[0].image_path).parent/"gpu_profile.json")}
    raw = _run_vision_subprocess("app.modules.vision_detector_worker",payload)
    if len(raw) != len(states):
        raise ValueError("Semantic classifier state count mismatch")
    result = []
    allowed = {"button","input","checkbox","radio","switch","slider","dropdown","link","tab","text","image","icon","container"}
    for state,anchors,response in zip(states,anchors_per_state,raw):
        try:
            labels=json.loads(response)
            if not isinstance(labels,dict):
                raise ValueError("Expected anchor ID to semantic type mapping")
        except (ValueError,TypeError):
            labels={}
            logger.warning("Semantic classification incomplete for frame %s",state.frame_index)
        with Image.open(state.image_path) as source:
            size=source.size
        items=[]
        for index,anchor in enumerate(anchors):
            kind=labels.get(str(index),anchor["type"])
            kind=kind if isinstance(kind,str) and kind in allowed else anchor["type"]
            kind=_supported_role(kind,anchor)
            item={**anchor,"type":kind,"text_bbox":anchor["bbox"]}
            if anchor["source"] == "yolo":
                item["id"]=index
            if kind in {"button","input","checkbox","radio","switch","slider","dropdown","link","tab"} and anchor["source"] == "ocr":
                x,y,w,h=map(float,anchor["bbox"])
                pad=max(3,h*.4)
                item["bbox"]=(max(0,x-pad),max(0,y-pad),min(size[0],x+w+pad)-max(0,x-pad),min(size[1],y+h+pad)-max(0,y-pad))
            items.append(item)
        result.append(items)
    return result


def _apply_verifier_patches(original_dets, verifier_output):
    if not isinstance(verifier_output, dict):
        return original_dets
        
    dets_by_id = {i: dict(d) for i, d in enumerate(original_dets)}
    
    for del_id in verifier_output.get("deleted_ids", []):
        dets_by_id.pop(del_id, None)
        
    for corr in verifier_output.get("corrections", []):
        if isinstance(corr, dict) and "id" in corr and corr["id"] in dets_by_id:
            for k, v in corr.items():
                if k == "type" and isinstance(v,str):
                    dets_by_id[corr["id"]][k] = _supported_role(v,dets_by_id[corr["id"]])
                    
    final_dets = list(dets_by_id.values())
    # The semantic checker cannot introduce ungrounded boxes. Missed objects
    # must come from another measured OCR/UI-detector pass, not free-form JSON.
    return final_dets


def _run_vision_verifier(states: list, detections_per_state: list[list[dict]]) -> tuple[list[list[dict]], list[bool]]:
    """Compatibility shim: independent verification is structural/browser based."""
    return detections_per_state, [False for _ in states]


# ---------------------------------------------------------------------------
# Merging: YOLO bboxes + Vision LLM semantic labels
# ---------------------------------------------------------------------------


def _valid_detection(item, image_size=None):
    import math
    if not isinstance(item, dict):
        return None
    try:
        x, y, w, h = map(float, item["bbox"])
        confidence = float(item.get("confidence", 0.5))
        if not all(math.isfinite(v) for v in (x, y, w, h, confidence)) or w <= 0 or h <= 0:
            return None
        if image_size:
            iw, ih = image_size
            right, bottom = min(iw, x+w), min(ih, y+h)
            x, y = max(0, x), max(0, y)
            w, h = right-x, bottom-y
            if w <= 0 or h <= 0:
                return None
        label = str(item.get("type", item.get("label", "container"))).strip().lower().replace(" ", "_")
        return {**item, "bbox": (x,y,w,h), "type": label,
                "text": item.get("text") if isinstance(item.get("text"), str) else None,
                "confidence": min(1.0, max(0.0, confidence))}
    except (KeyError, TypeError, ValueError):
        return None


def _merge_detections(yolo_dets, vlm_dets, iou_threshold=0.5):
    merged = []
    remaining = {i: d for i, d in enumerate(yolo_dets)}
    for vd in vlm_dets if isinstance(vlm_dets, list) else []:
        if not isinstance(vd, dict):
            continue
        match = vd.get("id")
        if not isinstance(match, int) or match not in remaining:
            match = None
            if "bbox" in vd and _valid_detection(vd):
                scores = [(i, _iou(d["bbox"], vd["bbox"])) for i,d in remaining.items()]
                if scores:
                    candidate, score = max(scores, key=lambda v:v[1])
                    if score >= iou_threshold:
                        match = candidate
        if match is not None:
            yd = remaining.pop(match)
            item = {**yd, **vd, "bbox": yd["bbox"], "source": "yolo+vlm"}
        else:
            item = {**vd, "source": vd.get("source", "vlm")}
        normalized = _valid_detection(item)
        if normalized and not any(_iou(normalized["bbox"], m["bbox"]) > .9 and
                                 normalized["type"] == m["type"] for m in merged):
            merged.append(normalized)
    for yd in remaining.values():
        normalized = _valid_detection({**yd,"source":"yolo"})
        if normalized:
            merged.append(normalized)
    return merged


# ---------------------------------------------------------------------------
# PaddleOCR subprocess (unchanged from v1)
# ---------------------------------------------------------------------------


class PersistentOCRWorker:
    """One bounded subprocess per batch: no leaked framework state between jobs."""
    def analyze(self, image_paths, lang, confidence_threshold):
        from app.modules import ocr_cache
        cached = [ocr_cache.load(path,lang,confidence_threshold) for path in image_paths]
        missing = [i for i,value in enumerate(cached) if value is None]
        if not missing:
            return cached
        payload = {"image_paths":[image_paths[i] for i in missing],"lang":lang,"confidence_threshold":confidence_threshold}
        completed = run_worker([sys.executable,"-m","app.modules.ocr_subprocess_worker"],
            input=json.dumps(payload)+"\n"+json.dumps({"command":"shutdown"})+"\n",
            timeout=settings.worker_timeout_sec)

        try:
            result = json.loads(completed.stdout.split("__OCR_JSON_START__")[1].split("__OCR_JSON_END__")[0])
        except (IndexError, ValueError) as exc:
            raise RuntimeError("OCR worker returned no valid result: " + completed.stderr[-2000:]) from exc
        if result.get("status") != "success" or completed.returncode:
            raise RuntimeError("OCR failed: " + str(result.get("message",completed.stderr[-2000:])))
        detections = result.get("detections")
        if not isinstance(detections,list) or len(detections) != len(missing):
            raise RuntimeError("OCR returned a different number of states")
        for i,detection in zip(missing,detections):
            cached[i] = detection
            ocr_cache.save(image_paths[i],lang,confidence_threshold,detection)
        return cached

    def shutdown(self):
        pass


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
    states: list[KeyStateFrame], yolo_confidence_threshold=None, ocr_lang=None,
    ocr_confidence_threshold=None, depth_model_name=None, colors_per_element=None,
    cuda_device_index=None, enable_3d=False,
) -> tuple[list[list[DetectedElement]], list[bool]]:
    """Detect, validate and fuse all regions without dropping dense UI content.

    Model failures are persisted as degradation warnings, never approvals.
    Source assets still allow a useful preview if semantic extraction is partial.
    Natural-scene depth is intentionally not used as CSS stacking order.
    """
    from pathlib import Path
    from app.modules.visual_styles import measure_styles, control_rectangles, snap_control_bbox
    if not states:
        return [], []
    diagnostics = {"warnings": [], "states": []}
    yolo = _run_yolo_cpu(states, settings.yolo_macro_weights_path, settings.yolo_micro_weights_path,
                         settings.yolo_export_onnx, yolo_confidence_threshold or settings.yolo_confidence_threshold)
    if not any(yolo):
        diagnostics["warnings"].append("No compatible UI-trained YOLO detections; using vision/OCR")
    try:
        ocr = _global_ocr_worker.analyze([s.image_path for s in states], ocr_lang or settings.paddleocr_lang,
                                       settings.ocr_confidence_threshold if ocr_confidence_threshold is None else ocr_confidence_threshold)
    except Exception as exc:
        diagnostics["warnings"].append(f"OCR unavailable: {exc}")
        ocr = [[] for _ in states]
    if len(ocr) != len(states):
        raise RuntimeError("OCR state count mismatch")
    try:
        vlm = _run_vision_detector(states,yolo,ocr)
        if len(vlm) != len(states):
            raise RuntimeError("Vision detector state count mismatch")
    except Exception as exc:
        diagnostics["warnings"].append(f"Vision classification unavailable: {exc}")
        vlm = [[] for _ in states]
    merged_states = []
    for state, yd, vd, text_items in zip(states, yolo, vlm, ocr):
        with Image.open(state.image_path) as image:
            size = image.size
        merged = [d for item in _merge_detections(yd, vd) if (d := _valid_detection(item,size))]
        for item in text_items:
            text = _valid_detection({**item, "bbox":item.get("bbox", item.get("box")), "type":"text","source":"ocr"},size)
            if not text or not text["text"]:
                continue
            matched = False
            for existing in merged:
                # Same words elsewhere on screen are not duplicates.
                if existing.get("text", "") and existing["text"].strip().casefold() == text["text"].strip().casefold():
                    ax,ay,aw,ah = existing["bbox"]; bx,by,bw,bh = text["bbox"]
                    overlap = max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
                    if overlap/(bw*bh) > .7:
                        if existing["type"] == "text":
                            existing.update(text)
                        else:
                            existing["text"] = text["text"]
                        matched = True
                        break
            if not matched:
                merged.append(text)
        merged_states.append(merged)
    verified, approvals = merged_states, [False]*len(states)
    diagnostics['recognition_model_count']=1
    diagnostics['verification']='geometry, native-control and browser checks; no model self-approval'
    all_elements = []
    for index, (state, raw) in enumerate(zip(states, verified)):
        elements = []
        rejected = 0
        with Image.open(state.image_path) as original:
            image = original.convert("RGB")
        rectangles = control_rectangles(image)
        for item in raw:
            item = _valid_detection(item, image.size)
            if not item:
                rejected += 1
                continue
            if item["type"] in {"button","input","dropdown","select","tab","link"}:
                item["bbox"] = snap_control_bbox(item["bbox"],rectangles,image.size)
            source = {"yolo":"yolo","yolo+vlm":"vision_llm","vlm":"vision_llm",
                      "ocr":"ocr","verifier":"verifier"}.get(item.get("source"),"vision_llm")
            x,y,w,h = item["bbox"]
            crop = image.crop((int(x),int(y),int(x+w),int(y+h)))
            styles=measure_styles(image,item["bbox"],item["text"],item["type"])
            if item.get("text") and item.get("text_bbox"):
                styles["font-size"] = f"{max(8,round(float(item['text_bbox'][3])*.8))}px"
            elements.append(DetectedElement(element_id=f"{source}-{len(elements):04d}",
                element_type=item["type"], bbox=BoundingBox(x=x,y=y,width=w,height=h),
                text_content=item["text"], confidence=item["confidence"], detection_source=source,
                hex_colors=_extract_hex_colors(crop,colors_per_element or settings.colorgram_colors_per_element),
                styles=styles,attributes={"data-text-bbox":json.dumps(item.get('text_bbox',item['bbox']))}))
        all_elements.append(elements)
        diagnostics["states"].append({"frame_index":state.frame_index,"elements":len(elements),
            "rejected_geometry":rejected,"verifier_approved":approvals[index],
            "raw_detector":vlm[index],"final_detections":[e.model_dump() for e in elements]})
    diagnostics_path = Path(states[0].image_path).parent / "detection_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    if diagnostics["warnings"]:
        logger.warning("Detection degraded: %s", diagnostics["warnings"])
    return all_elements, approvals


async def analyze_states(states: list[KeyStateFrame], **kwargs) -> tuple[list[list[DetectedElement]], list[bool]]:
    return await asyncio.to_thread(_analyze_states_sync, states, **kwargs)


async def analyze_state(state: KeyStateFrame, **kwargs) -> list[DetectedElement]:
    res, _ = await analyze_states([state], **kwargs)
    return res[0] if res else []
