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

# Reduce VRAM fragmentation on Windows (expandable_segments not supported)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")


def _fix_json_string_escaping(text: str) -> str:
    """
    Fix unescaped newlines, tabs, and carriage returns inside JSON string values.
    The LLM outputs correct JSON structure but with literal newlines in string
    values (e.g. JSX code), which is invalid JSON. This walks the text character
    by character, tracking whether we're inside a string, and escapes any raw
    control characters found within strings.
    """
    result = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\' and in_string and i + 1 < len(text):
            # Already an escape sequence — keep it as-is
            result.append(c)
            result.append(text[i + 1])
            i += 2
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
            i += 1
            continue
        if in_string:
            if c == '\n':
                result.append('\\n')
            elif c == '\r':
                result.append('\\r')
            elif c == '\t':
                result.append('\\t')
            else:
                result.append(c)
        else:
            result.append(c)
        i += 1
    return ''.join(result)


def _recover_json(response_text: str, system_prompt: str) -> str:
    """
    Attempt to parse response_text as JSON. If it fails, apply recovery
    heuristics based on common local-LLM failure modes:
      1. Unescaped newlines/tabs inside JSON string values
      2. Trailing garbage after the closing brace
      3. Raw JSX/CSS code without the JSON wrapper
    """

    # Fast path: already valid JSON
    try:
        json.loads(response_text)
        return response_text
    except (json.JSONDecodeError, ValueError):
        pass

    # Heuristic 1 (most common): Fix unescaped newlines inside JSON strings
    # The LLM outputs {"component_jsx": "<div>\n<button>\n</div>"} with literal
    # newlines instead of the escaped \\n that JSON requires
    if response_text.lstrip().startswith('{'):
        fixed = _fix_json_string_escaping(response_text)
        try:
            json.loads(fixed)
            print("[Local LLM Worker] Fixed unescaped newlines in JSON string values", file=sys.stderr, flush=True)
            return fixed
        except (json.JSONDecodeError, ValueError):
            pass

        # Heuristic 2: The fixed text might have trailing garbage — find balanced braces
        first_brace = fixed.find('{')
        if first_brace != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(first_brace, len(fixed)):
                c = fixed[i]
                if esc:
                    esc = False
                    continue
                if c == '\\':
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = fixed[first_brace:i + 1]
                        try:
                            json.loads(candidate)
                            print("[Local LLM Worker] Extracted balanced JSON object", file=sys.stderr, flush=True)
                            return candidate
                        except (json.JSONDecodeError, ValueError):
                            break

    # Heuristic 3: The model output raw code instead of JSON.
    # Only apply this if the model DID NOT try to output JSON (i.e. doesn't start with { )
    if not response_text.lstrip().startswith('{'):
        is_2d_prompt = "component_jsx" in system_prompt and "styles_css" in system_prompt
        is_3d_prompt = "scene3d_jsx" in system_prompt

        if is_2d_prompt:
            jsx_part = response_text
            css_part = ""

            # Look for CSS-like content after the component
            css_markers = [".state_", ".el-", "position: absolute"]
            for marker in css_markers:
                idx = response_text.rfind(marker)
                if idx != -1:
                    line_start = response_text.rfind('\n', 0, idx)
                    if line_start != -1:
                        jsx_part = response_text[:line_start].strip()
                        css_part = response_text[line_start:].strip()
                        break

            result = json.dumps({"component_jsx": jsx_part, "styles_css": css_part})
            print(f"[Local LLM Worker] Wrapped raw output into JSON (jsx={len(jsx_part)} chars, css={len(css_part)} chars)", file=sys.stderr, flush=True)
            return result

        if is_3d_prompt:
            result = json.dumps({"scene3d_jsx": response_text})
            print("[Local LLM Worker] Wrapped raw 3D output into JSON", file=sys.stderr, flush=True)
            return result

    # Last resort: return as-is
    return response_text


def main() -> None:
    # Read the initialization payload
    init_line = sys.stdin.readline()
    if not init_line:
        sys.exit(1)
        
    payload = json.loads(init_line)
    model_name = payload["model_name"]
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
            "attn_implementation": "sdpa",
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
                print(f"[Local LLM Worker] Notice: 4-bit quantization config unavailable ({exc}), falling back to half-precision with auto device_map.", file=sys.stderr, flush=True)
                model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
                model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            model_kwargs["device_map"] = "auto"

        # Clear any stale VRAM allocations before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[Local LLM Worker] Loading weights onto GPU ({model_kwargs.get('device_map')}) in 4-bit mode...", file=sys.stderr, flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print("[Local LLM Worker] Model successfully loaded into VRAM. Preparing prompt...", file=sys.stderr, flush=True)

        while True:
            line = sys.stdin.readline()
            if not line:
                break
                
            payload = json.loads(line)
            if payload.get("command") == "shutdown":
                break
                
            system_prompt = payload["system_prompt"]
            user_content = payload["user_content"]
            temperature = float(payload.get("temperature", 0.1))
            max_new_tokens = int(payload.get("max_new_tokens", 2048))

            messages = [
                {"role": "system", "content": system_prompt + "\nIMPORTANT: You must output ONLY valid JSON matching the requested structure, without markdown code block fences or extra text."},
                {"role": "user", "content": user_content},
            ]

            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = f"{system_prompt}\n\nUser: {user_content}\n\nAssistant:\n"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Log VRAM and token stats for debugging OOM
            input_len = inputs["input_ids"].shape[1]
            if torch.cuda.is_available():
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_alloc = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"[Local LLM Worker] VRAM: {vram_alloc:.1f}GB allocated / {vram_total:.1f}GB total ({vram_total - vram_alloc:.1f}GB free)", file=sys.stderr, flush=True)
            print(f"[Local LLM Worker] Input: {input_len} tokens. Generating up to {max_new_tokens} tokens on {model.device}...", file=sys.stderr, flush=True)
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
                parts = response_text.split("```")
                if len(parts) >= 3:
                    response_text = parts[1].strip()
                    # Remove language tag if present (e.g. 'jsx\n...')
                    if response_text and not response_text.startswith('{'):
                        newline_idx = response_text.find('\n')
                        if newline_idx != -1 and newline_idx < 20:
                            response_text = response_text[newline_idx+1:].strip()

            # Attempt to parse as JSON; if it fails, try to recover
            response_text = _recover_json(response_text, system_prompt)

            print("__LLM_JSON_START__\n" + response_text + "\n__LLM_JSON_END__", flush=True)

            del inputs, outputs, generated_ids
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
