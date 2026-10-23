"""Short-lived local NF4 coder; process exit releases all model allocations."""
import json
import os
from pathlib import Path
import sys
from app.config import settings


def main():
    payload = json.loads(sys.stdin.read())
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from huggingface_hub import snapshot_download
    from app.core.model_budget import configure_coder
    model_name = payload['model']
    profile = configure_coder(torch, model_name, settings.cuda_device_index)
    profile['worker_pid'] = os.getpid()
    resolved = snapshot_download(model_name, cache_dir=str(settings.models_cache_dir/'hub'), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(resolved, local_files_only=True,
        device_map={'':f'cuda:{settings.cuda_device_index}'},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype))
    text = tokenizer.apply_chat_template(payload['messages'], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    if inputs.input_ids.shape[-1] > 6144:
        raise ValueError('Code request exceeds the bounded 6144-token input budget')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=min(2048,settings.local_llm_max_new_tokens),
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    source = tokenizer.decode(output[0,inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    profile.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(settings.cuda_device_index),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(settings.cuda_device_index))
    print('__CODER_JSON_START__'+json.dumps({'source':source,'profile':profile})+'__CODER_JSON_END__')


if __name__ == '__main__':
    main()
