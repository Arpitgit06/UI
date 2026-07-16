"""
Standalone worker script for running local Hugging Face causal language models (e.g., Qwen2.5-Coder-7B-Instruct)
in an isolated Windows subprocess. This guarantees 100% GPU VRAM cleanup when generation finishes.
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
    user_content = payload["user_content"]
    temperature = float(payload.get("temperature", 0.1))
    max_new_tokens = int(payload.get("max_new_tokens", 2048))
    load_in_4bit = bool(payload.get("load_in_4bit", True))

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[Local LLM Worker] Resolving model path for {model_name}...", file=sys.stderr, flush=True)
        # Bypassing huggingface_hub cache validation: On Windows without Administrator/Developer Mode, 
        # symlinks fail, causing the blobs/ directory to remain empty. This tricks HF into thinking the
        # cache is corrupt and forces a 15GB re-download every run. By passing the direct snapshot path,
        # we treat it as a local offline model and bypass HF Hub completely.
        hub_dir = models_cache_dir / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
        if hub_dir.exists():
            snapshots = list(hub_dir.iterdir())
            if snapshots:
                # Use the latest or only snapshot directory available
                model_name = str(snapshots[0])
                print(f"[Local LLM Worker] Found local snapshot, loading directly from: {model_name}", file=sys.stderr, flush=True)

        print(f"[Local LLM Worker] Loading tokenizer...", file=sys.stderr, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(models_cache_dir), local_files_only=False)
        
        # Load model using 4-bit quantization directly onto cuda:0 to prevent accelerate from CPU-offloading unquantized shard estimates
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
                print(f"[Local LLM Worker] Notice: 4-bit quantization config unavailable ({exc}), falling back to half-precision with auto device_map.", file=sys.stderr, flush=True)
                model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            model_kwargs["device_map"] = "auto"

        print(f"[Local LLM Worker] Loading weights onto GPU ({model_kwargs.get('device_map')}) in 4-bit mode...", file=sys.stderr, flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print("[Local LLM Worker] Model successfully loaded into VRAM. Preparing prompt...", file=sys.stderr, flush=True)

        messages = [
            {"role": "system", "content": system_prompt + "\nIMPORTANT: You must output ONLY valid JSON matching the requested structure, without markdown code block fences or extra text."},
            {"role": "user", "content": user_content},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{system_prompt}\n\nUser: {user_content}\n\nAssistant:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print(f"[Local LLM Worker] Generating up to {max_new_tokens} tokens on {model.device}...", file=sys.stderr, flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"[Local LLM Worker] Generation completed ({len(generated_ids)} tokens produced).", file=sys.stderr, flush=True)

        # Clean up markdown code blocks if the model wrapped the JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        print("__LLM_JSON_START__")
        print(response_text)
        print("__LLM_JSON_END__")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
