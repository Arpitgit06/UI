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
You are a UI detection verifier. A detection system has identified these UI elements:

{detections}

VERIFY and CORRECT this detection list by comparing it against the actual screenshot.
Do NOT re-output the entire list. Only output the differences:
1. "corrections": Elements whose type, text, or bbox need fixing. Reference them by "id".
2. "missed_elements": New elements that were completely missed. Provide their "type", "bbox", and "text".
3. "deleted_ids": A list of IDs of elements that are false positives and should be removed.

Return a JSON object with exactly these fields:
- "approved": true if the original detections are mostly good, false if they have major issues
- "corrections": list of objects with "id" and the fields to correct (e.g., {{"id": 2, "type": "button"}})
- "missed_elements": list of new elements (e.g., {{"type": "icon", "bbox": [10, 10, 20, 20]}})
- "deleted_ids": list of IDs to remove (e.g., [4, 7])

Return ONLY valid JSON.
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
            max_pixels=384 * 28 * 28,  # Reduced from 512 to prevent 8GB VRAM OOM on 7B model
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
        for img_idx, (img_path, dets) in enumerate(zip(image_paths, detections_per_image)):
            
            # Add IDs, convert bboxes to int, and minify JSON to heavily reduce input tokens
            dets_with_ids = []
            for i, d in enumerate(dets):
                bbox = d.get("bbox")
                if bbox and len(bbox) == 4:
                    bbox = [int(v) for v in bbox]
                dets_with_ids.append({
                    "id": i,
                    "type": d.get("type", "div"),
                    "bbox": bbox,
                    "text": d.get("text")
                })
            dets_summary = json.dumps(dets_with_ids, separators=(',', ':'))
            
            prompt_text = _VERIFIER_PROMPT.format(detections=dets_summary)
            prompt_text += "\nIMPORTANT: Output ONLY valid JSON, no markdown fences."

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
