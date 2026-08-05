"""
Vision Verifier Worker: Qwen2-VL-7B-Instruct subprocess for detection verification.

This is the Stage 2 verifier in the hybrid parallel pipeline. It receives:
  - An image path (key-state screenshot)
  - The merged detection list from Stage 1 (YOLO + 3B detector)

Its ONLY job is verification and correction — not full re-detection:
  1. Check if any major visible elements were missed
  2. Verify element types make sense
  3. Catch any wrong text transcriptions
  4. Approve or correct the detection list

Runs in an isolated subprocess for 100% VRAM cleanup on exit.
Uses the existing Qwen2-VL-7B-Instruct model (already cached by setup.bat).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

project_root = Path(__file__).resolve().parent.parent.parent
models_cache_dir = project_root / "models_cache"
models_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(models_cache_dir))
os.environ.setdefault("TRANSFORMERS_CACHE", str(models_cache_dir))
os.environ.setdefault("TORCH_HOME", str(models_cache_dir))


_VERIFIER_PROMPT = """\
You are a UI detection verifier. A detection system has identified these UI elements in the screenshot:

{detections}

Your task is to VERIFY and CORRECT this detection list by comparing it against the actual screenshot:

1. MISSED ELEMENTS: Are there any clearly visible UI elements in the screenshot that are NOT in the list above? If so, add them.
2. WRONG TYPES: Are any element types obviously wrong? (e.g., a navbar labeled as "button"). If so, correct them.
3. WRONG TEXT: Is any text content incorrect? If so, fix it.
4. BAD BBOXES: Are any bounding boxes wildly off from the visible element? If so, adjust them.
5. FALSE POSITIVES: Are any detections clearly wrong (detecting something that isn't there)? If so, remove them.

Return a JSON object with exactly two fields:
- "approved": true if the detections are good enough (minor issues are OK), false if major problems found
- "elements": the COMPLETE corrected element list (include ALL elements — both verified originals and any new ones you added)

Each element in the list must have: "type", "bbox" [x, y, width, height], "text" (or null), "confidence".

Return ONLY valid JSON, no markdown fences.
"""


def _resolve_model_path(model_name: str) -> str:
    hub_dir = models_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if hub_dir.exists():
        snapshots = list(hub_dir.iterdir())
        if snapshots:
            resolved = str(snapshots[0])
            print(f"[Vision Verifier] Found local snapshot: {resolved}", file=sys.stderr, flush=True)
            return resolved
    return model_name


def main() -> None:
    if len(sys.argv) < 2:
        print("[Vision Verifier] Error: payload file path required", file=sys.stderr, flush=True)
        sys.exit(1)

    payload_file = sys.argv[1]
    with open(payload_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model_name = payload["model_name"]
    image_paths = payload.get("image_paths", [])
    detections_per_image = payload.get("detections", [])  # list of lists
    temperature = float(payload.get("temperature", 0.1))
    max_new_tokens = int(payload.get("max_new_tokens", 2048))
    load_in_4bit = bool(payload.get("load_in_4bit", True))

    try:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        resolved_name = _resolve_model_path(model_name)

        print(f"[Vision Verifier] Loading processor for {model_name}...", file=sys.stderr, flush=True)
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
                print(f"[Vision Verifier] 4-bit unavailable ({exc}), using fp16", file=sys.stderr, flush=True)
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            model_kwargs["device_map"] = "auto"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[Vision Verifier] Loading 7B verifier model...", file=sys.stderr, flush=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(resolved_name, **model_kwargs)
        print("[Vision Verifier] Model loaded into VRAM.", file=sys.stderr, flush=True)

        results = []
        for img_idx, img_path in enumerate(image_paths):
            dets = detections_per_image[img_idx] if img_idx < len(detections_per_image) else []

            det_summary = json.dumps(
                [{"type": d.get("type", "unknown"), "bbox": d["bbox"], "text": d.get("text"), "confidence": d.get("confidence", 0.5)} for d in dets],
                indent=2,
            )
            prompt_text = _VERIFIER_PROMPT.format(detections=det_summary)
            prompt_text += "\nIMPORTANT: Output ONLY valid JSON object, no markdown fences."

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

            print(f"[Vision Verifier] Image {img_idx+1}/{len(image_paths)}: verifying {len(dets)} elements...", file=sys.stderr, flush=True)

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

            print(f"[Vision Verifier] Done image {img_idx+1}.", file=sys.stderr, flush=True)

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            results.append(response_text)

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
