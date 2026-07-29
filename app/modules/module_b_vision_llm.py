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


def _parse_truncated_json(json_str: str) -> list[dict]:
    """Parse a JSON array that may have been truncated by the token limit.
    
    The Vision LLM often hits max_new_tokens mid-JSON, producing output like:
        [{"type":"button",...}, {"type":"text",...}, {"type":"text", "bbox": [10,
    This function salvages all complete objects from the array.
    """
    # First, try parsing as-is
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
        return [result] if isinstance(result, dict) else []
    except json.JSONDecodeError:
        pass

    # Find the last complete JSON object by searching for "}," or "}" + "]"
    # Working backwards from the end, find the last closing brace that completes an object
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
        logger.warning("Vision LLM output contained no complete JSON objects.")
        return []

    # Truncate to last complete object and close the array
    repaired = json_str[:last_complete + 1].rstrip().rstrip(',') + ']'
    if not repaired.startswith('['):
        repaired = '[' + repaired

    try:
        result = json.loads(repaired)
        logger.info(f"Repaired truncated Vision LLM JSON: salvaged {len(result)} complete element(s).")
        return result if isinstance(result, list) else []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to repair truncated JSON: {e}")
        raise VisionLLMError(f"Vision LLM produced unparseable JSON: {e}") from e

async def _call_vision_llm(image_paths: list[str], model_name: str, load_in_4bit: bool) -> list[list[dict]]:
    payload_file = settings.jobs_dir / f"vision_payload_{uuid.uuid4().hex}.json"
    payload_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(payload_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": model_name,
                    "system_prompt": _VISION_PROMPT,
                    "image_paths": [str(p) for p in image_paths],
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
        
        try:
            results_list = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise VisionLLMError(f"Vision LLM worker returned invalid JSON array: {e}\n{json_str}")
            
        parsed_results = []
        for res_str in results_list:
            parsed_results.append(_parse_truncated_json(res_str))
            
        return parsed_results

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
    res = await analyze_states_fallback([state], cuda_device_index)
    return res[0] if res else []


async def analyze_states_fallback(
    states: list[KeyStateFrame], 
    cuda_device_index: Optional[int] = None
) -> list[list[DetectedElement]]:
    if not states:
        return []
        
    model_name = settings.vision_llm_model_name
    load_in_4bit = settings.local_llm_load_in_4bit
    cuda_device_index = settings.cuda_device_index if cuda_device_index is None else cuda_device_index

    logger.info(f"Module B Fallback: Running Vision LLM ({model_name}) on {len(states)} image(s)")

    guard = GPUPipelineGuard(device_index=cuda_device_index)
    
    with guard.stage(
        "qwen2-vl",
        loader=lambda: "subprocess_ready",
        unloader=lambda _x: None,
        min_free_mb=1024,
    ):
        raw_items_list = await _call_vision_llm([s.image_path for s in states], model_name, load_in_4bit)

    all_elements: list[list[DetectedElement]] = []
    
    # Maximum elements per frame to keep code tree prompts within 8GB VRAM limits
    MAX_ELEMENTS_PER_FRAME = 20

    for state, raw_items in zip(states, raw_items_list):
        elements: list[DetectedElement] = []
        seen_texts: set[str] = set()
        for idx, item in enumerate(raw_items):
            try:
                x, y, w, h = item["bbox"]
                el_type = item.get("type", "div")
                text = item.get("text", None)
                conf = float(item.get("confidence", 0.5))

                # Deduplicate: skip elements with identical text content
                # The Vision LLM often hallucinates repeated text blocks
                if text:
                    text_key = text.strip().lower()
                    if text_key in seen_texts:
                        continue
                    seen_texts.add(text_key)
                
                elements.append(
                    DetectedElement(
                        element_id=f"vision-{idx:03d}",
                        element_type=el_type,
                        text_content=text,
                        bbox=BoundingBox(x=x, y=y, width=w, height=h, z_index=50.0),
                        hex_colors=[],
                        confidence=round(conf, 4),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to parse Vision LLM item {item}: {e}")

        # Cap element count to prevent OOM in Module D's code generation
        if len(elements) > MAX_ELEMENTS_PER_FRAME:
            # Sort by confidence descending, keep the best ones
            elements.sort(key=lambda el: el.confidence, reverse=True)
            dropped = len(elements) - MAX_ELEMENTS_PER_FRAME
            elements = elements[:MAX_ELEMENTS_PER_FRAME]
            logger.info(f"Module B Fallback: capped {state.image_path} from {dropped + MAX_ELEMENTS_PER_FRAME} to {MAX_ELEMENTS_PER_FRAME} elements (dropped {dropped} lowest-confidence)")

        logger.info(f"Module B Fallback: {state.image_path} -> {len(elements)} element(s) detected via Vision LLM")
        all_elements.append(elements)
        
    return all_elements
