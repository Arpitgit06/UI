"""
Vision Detector Worker: Qwen2.5-VL-3B-Instruct subprocess for UI element detection.

This is the Stage 1 detector in the hybrid parallel pipeline. It receives:
  - An image path (key-state screenshot)
  - Optional YOLO pre-detections (bounding boxes found by YOLO on CPU)

It returns a structured JSON array of detected UI elements with type, bbox,
text content, and confidence. When YOLO pre-detections are provided, the
prompt instructs the model to:
  1. Assign semantic labels to YOLO's existing boxes
  2. Detect any additional UI elements YOLO missed
This reduces output tokens and processing time vs. full-frame detection.

Runs in an isolated subprocess for 100% VRAM cleanup on exit.
"""
import json
import os
import sys
from pathlib import Path

# Better CUDA memory allocation for GPUs with limited VRAM (e.g. 8GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Setup strict local cache paths before importing torch/transformers
project_root = Path(__file__).resolve().parent.parent.parent
models_cache_dir = project_root / "models_cache"
models_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(models_cache_dir))
os.environ.setdefault("TRANSFORMERS_CACHE", str(models_cache_dir))
os.environ.setdefault("TORCH_HOME", str(models_cache_dir))


_DETECTOR_PROMPT_WITH_YOLO = """\
You are a precise UI element detector analyzing a screenshot of a user interface.

A fast pre-detector has already found these bounding boxes in the image:
{yolo_detections}

For each pre-detected box above, determine its UI element type from this list:
button, text, input, checkbox, radio, switch, slider, icon, image, navbar, toolbar, card, modal, drawer, list, list_item, video, canvas, container, header, footer, sidebar, tab, dropdown, tooltip, badge, avatar, progress_bar, divider, link

Also carefully scan the entire screenshot for any ADDITIONAL UI elements that the pre-detector missed. Common things the pre-detector misses: text labels, small icons, input placeholders, navigation items, status indicators, background images, gradient overlays.

For EVERY element (both pre-detected and newly found), provide:
- "type": the UI element type from the list above
- "bbox": bounding box as [x, y, width, height] in pixels
- "text": any visible text content (null if none)
- "confidence": your confidence from 0.0 to 1.0

Return ONLY a JSON array. Example:
[
  {"type": "button", "bbox": [10, 20, 100, 40], "text": "Submit", "confidence": 0.95},
  {"type": "text", "bbox": [120, 20, 200, 20], "text": "Hello World", "confidence": 0.99}
]
"""

_DETECTOR_PROMPT_NO_YOLO = """\
You are a precise UI element detector analyzing a screenshot of a user interface.

Carefully scan the entire screenshot and detect every distinct UI element. Element types to look for:
button, text, input, checkbox, radio, switch, slider, icon, image, navbar, toolbar, card, modal, drawer, list, list_item, video, canvas, container, header, footer, sidebar, tab, dropdown, tooltip, badge, avatar, progress_bar, divider, link

For EVERY element found, provide:
- "type": the UI element type from the list above
- "bbox": bounding box as [x, y, width, height] in pixels
- "text": any visible text content (null if none)
- "confidence": your confidence from 0.0 to 1.0

Return ONLY a JSON array. Example:
[
  {"type": "button", "bbox": [10, 20, 100, 40], "text": "Submit", "confidence": 0.95},
  {"type": "text", "bbox": [120, 20, 200, 20], "text": "Hello World", "confidence": 0.99}
]
"""


def _resolve_model_path(model_name: str) -> str:
    """Resolve HuggingFace model name to local snapshot path if available."""
    hub_dir = models_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if hub_dir.exists():
        snapshots = list(hub_dir.iterdir())
        if snapshots:
            resolved = str(snapshots[0])
            print(f"[Vision Detector] Found local snapshot: {resolved}", file=sys.stderr, flush=True)
            return resolved
    return model_name


def main() -> None:
    if len(sys.argv) < 2:
        print("[Vision Detector] Error: payload file path required as argument", file=sys.stderr, flush=True)
        sys.exit(1)

    payload_file = sys.argv[1]
    with open(payload_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model_name = payload["model_name"]
    image_paths = payload.get("image_paths", [])
    if "image_path" in payload and not image_paths:
        image_paths = [payload["image_path"]]
    yolo_detections_per_image = payload.get("yolo_detections", [])  # list of lists
    temperature = float(payload.get("temperature", 0.1))
    max_new_tokens = int(payload.get("max_new_tokens", 2048))
    load_in_4bit = bool(payload.get("load_in_4bit", True))

    try:
        import torch
        from transformers import AutoProcessor

        # Try Qwen2.5-VL first (newer), fall back to Qwen2-VL
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
            print("[Vision Detector] Using Qwen2.5-VL model class", file=sys.stderr, flush=True)
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as VLModel
            print("[Vision Detector] Falling back to Qwen2-VL model class", file=sys.stderr, flush=True)

        from qwen_vl_utils import process_vision_info

        resolved_name = _resolve_model_path(model_name)

        print(f"[Vision Detector] Loading processor for {model_name}...", file=sys.stderr, flush=True)
        processor = AutoProcessor.from_pretrained(
            resolved_name,
            cache_dir=str(models_cache_dir),
            local_files_only=False,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        model_kwargs = {
            "cache_dir": str(models_cache_dir),
            "local_files_only": False,
        }
        if load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
                model_kwargs["device_map"] = {"": "cuda:0"}
            except Exception as exc:
                print(f"[Vision Detector] 4-bit unavailable ({exc}), using fp16", file=sys.stderr, flush=True)
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            model_kwargs["device_map"] = "auto"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[Vision Detector] Loading 3B model weights...", file=sys.stderr, flush=True)
        model = VLModel.from_pretrained(resolved_name, **model_kwargs)
        print("[Vision Detector] Model loaded into VRAM.", file=sys.stderr, flush=True)

        results = []
        for img_idx, img_path in enumerate(image_paths):
            # Build prompt based on whether YOLO pre-detections are available
            yolo_dets = yolo_detections_per_image[img_idx] if img_idx < len(yolo_detections_per_image) else []
            
            if yolo_dets:
                yolo_summary = json.dumps([{"bbox": d["bbox"], "confidence": d.get("confidence", 0.5)} for d in yolo_dets], indent=2)
                prompt_text = _DETECTOR_PROMPT_WITH_YOLO.format(yolo_detections=yolo_summary)
            else:
                prompt_text = _DETECTOR_PROMPT_NO_YOLO

            prompt_text += "\nIMPORTANT: Output ONLY valid JSON array, no markdown fences or extra text."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            vision_info = process_vision_info(messages)
            if len(vision_info) == 3:
                image_inputs, video_inputs, video_kwargs = vision_info
            else:
                image_inputs, video_inputs = vision_info
                video_kwargs = {}

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **(video_kwargs or {}),
            )
            inputs = inputs.to(model.device)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vram_alloc = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0
            print(f"[Vision Detector] Image {img_idx+1}/{len(image_paths)}: generating ({vram_alloc:.1f}GB VRAM)...", file=sys.stderr, flush=True)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0.0,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            print(f"[Vision Detector] Done image {img_idx+1}.", file=sys.stderr, flush=True)

            # Clean markdown fences
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            results.append(response_text)

            # Cleanup tensors
            del inputs, generated_ids, generated_ids_trimmed
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("__VISION_JSON_START__")
        print(json.dumps(results))
        print("__VISION_JSON_END__")

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
