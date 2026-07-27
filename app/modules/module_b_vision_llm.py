"""
Module B (Fallback): Vision LLM Spatial Engine
Uses Qwen2-VL-7B-Instruct to parse UI screenshots and emit bounding boxes,
text, and element types when the standard YOLO+OCR pipeline fails.
"""
import asyncio
import json
import sys
import uuid
from typing import Optional

from app.config import settings
from app.models.schemas import BoundingBox, DetectedElement, KeyStateFrame
from app.utils.logger import get_logger
from app.core.vram_manager import GPUPipelineGuard

logger = get_logger("omniui.module_b_vision")

_VISION_PROMPT = """\
Analyze this user interface screenshot. Identify every distinct UI element (buttons, text fields, icons, images, text blocks, navbars, cards, etc.).
For each element, provide:
- type: A short string identifying the element (e.g. "button", "text", "image", "input", "navbar")
- bbox: The bounding box in pixels [x, y, width, height]
- text: The visible text if applicable, otherwise null
- confidence: A float between 0 and 1 indicating your confidence

Return a JSON array of these elements. Example:
[
  {"type": "button", "bbox": [10, 20, 100, 40], "text": "Submit", "confidence": 0.95},
  {"type": "text", "bbox": [120, 20, 200, 20], "text": "Hello World", "confidence": 0.99}
]
"""

class VisionLLMError(RuntimeError):
    pass

async def _call_vision_llm(image_path: str, model_name: str, load_in_4bit: bool) -> list[dict]:
    payload_file = settings.jobs_dir / f"vision_payload_{uuid.uuid4().hex}.json"
    payload_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(payload_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "system_prompt": _VISION_PROMPT,
                    "image_path": str(image_path),
                    "temperature": 0.1,
                    "max_new_tokens": 1500,
                    "load_in_4bit": load_in_4bit,
                },
                f,
            )

        def _run_subprocess():
            import subprocess
            return subprocess.run(
                [sys.executable, "-m", "app.modules.vision_llm_worker", str(payload_file)],
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

        completed_process = await asyncio.to_thread(_run_subprocess)
        
        stdout_lines = completed_process.stdout.splitlines(keepends=True)
        for line in stdout_lines:
            if "__VISION_JSON_START__" not in line and "__VISION_JSON_END__" not in line and line.strip():
                logger.info(f"[Vision LLM STDOUT] {line.strip()}")
                
        if completed_process.returncode != 0:
            raise VisionLLMError(f"Vision LLM worker failed. Code: {completed_process.returncode}")

        stdout = completed_process.stdout
        if "__VISION_JSON_START__" not in stdout or "__VISION_JSON_END__" not in stdout:
            raise VisionLLMError("Vision LLM response missing JSON markers.")

        json_str = stdout.split("__VISION_JSON_START__")[1].split("__VISION_JSON_END__")[0].strip()
        return json.loads(json_str)

    finally:
        if payload_file.exists():
            try:
                payload_file.unlink()
            except Exception:
                pass


async def analyze_state_fallback(
    state: KeyStateFrame, 
    cuda_device_index: Optional[int] = None
) -> list[DetectedElement]:
    model_name = settings.vision_llm_model_name
    load_in_4bit = settings.local_llm_load_in_4bit
    cuda_device_index = settings.cuda_device_index if cuda_device_index is None else cuda_device_index

    logger.info(f"Module B Fallback: Running Vision LLM ({model_name}) on {state.image_path}")

    guard = GPUPipelineGuard(device_index=cuda_device_index)
    
    with guard.stage(
        "qwen2-vl",
        loader=lambda: "subprocess_ready",
        unloader=lambda _x: None,
        min_free_mb=1024,
    ):
        raw_items = await _call_vision_llm(state.image_path, model_name, load_in_4bit)

    elements: list[DetectedElement] = []
    for idx, item in enumerate(raw_items):
        try:
            x, y, w, h = item["bbox"]
            el_type = item.get("type", "div")
            text = item.get("text", None)
            conf = float(item.get("confidence", 0.5))
            
            elements.append(
                DetectedElement(
                    element_id=f"vision-{idx:03d}",
                    element_type=el_type,
                    bbox=BoundingBox(x=x, y=y, width=w, height=h, z_index=50.0),
                    text_content=text,
                    hex_colors=[],
                    confidence=round(conf, 4),
                )
            )
        except Exception as e:
            logger.warning(f"Failed to parse Vision LLM item {item}: {e}")

    logger.info(f"Module B Fallback: {state.image_path} -> {len(elements)} element(s) detected via Vision LLM")
    return elements
