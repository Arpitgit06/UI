"""
Standalone worker script for running local Qwen2-VL-7B-Instruct
in an isolated Windows subprocess. This guarantees 100% GPU VRAM cleanup.
"""
import json
import os
import sys
from pathlib import Path

# Setup strict local cache paths before importing torch/transformers
project_root = Path(__file__).resolve().parent.parent.parent
models_cache_dir = project_root / "models_cache"
models_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(models_cache_dir))
os.environ.setdefault("TRANSFORMERS_CACHE", str(models_cache_dir))
os.environ.setdefault("TORCH_HOME", str(models_cache_dir))

def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(1)

    payload_file = sys.argv[1]
    with open(payload_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model_name = payload["model_name"]
    system_prompt = payload["system_prompt"]
    image_path = payload["image_path"]
    temperature = float(payload.get("temperature", 0.1))
    max_new_tokens = int(payload.get("max_new_tokens", 2048))
    load_in_4bit = bool(payload.get("load_in_4bit", True))

    try:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        print(f"[Vision LLM Worker] Resolving model path for {model_name}...", file=sys.stderr, flush=True)
        hub_dir = models_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
        if hub_dir.exists():
            snapshots = list(hub_dir.iterdir())
            if snapshots:
                model_name = str(snapshots[0])
                print(f"[Vision LLM Worker] Found local snapshot, loading directly from: {model_name}", file=sys.stderr, flush=True)

        print(f"[Vision LLM Worker] Loading processor...", file=sys.stderr, flush=True)
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=str(models_cache_dir), local_files_only=False)
        
        model_kwargs = {
            "cache_dir": str(models_cache_dir),
            "local_files_only": False,
        }
        if load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
                model_kwargs["device_map"] = {"": "cuda:0"}
            except Exception as exc:
                print(f"[Vision LLM Worker] Notice: 4-bit quantization config unavailable ({exc}), falling back to half-precision with auto device_map.", file=sys.stderr, flush=True)
                model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            model_kwargs["device_map"] = "auto"

        print(f"[Vision LLM Worker] Loading weights onto GPU ({model_kwargs.get('device_map')}) in 4-bit mode...", file=sys.stderr, flush=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        print("[Vision LLM Worker] Model successfully loaded into VRAM. Preparing prompt...", file=sys.stderr, flush=True)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": system_prompt + "\nIMPORTANT: You must output ONLY valid JSON matching the requested structure, without markdown code block fences or extra text."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        print(f"[Vision LLM Worker] Generating up to {max_new_tokens} tokens on {model.device}...", file=sys.stderr, flush=True)
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

        print(f"[Vision LLM Worker] Generation completed.", file=sys.stderr, flush=True)

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        print("__VISION_JSON_START__")
        print(response_text)
        print("__VISION_JSON_END__")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
