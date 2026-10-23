"""Bounded per-region semantic classification; text/geometry are never generated."""
import json
import os
import sys
from pathlib import Path

from app.config import settings
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","garbage_collection_threshold:0.6")


def main(payload_override=None):
    if payload_override is None:
        with open(sys.argv[1],encoding="utf-8") as source:
            payload=json.load(source)
    else:
        payload=payload_override
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from huggingface_hub import snapshot_download
    from qwen_vl_utils import process_vision_info
    model_name=payload["model_name"]
    from app.core.model_budget import configure_recognition
    profile=configure_recognition(torch,model_name,settings.cuda_device_index)
    profile['worker_pid']=os.getpid()
    resolved=snapshot_download(repo_id=model_name,cache_dir=str(settings.models_cache_dir/"hub"),local_files_only=settings.offline)
    processor=AutoProcessor.from_pretrained(resolved,local_files_only=True,min_pixels=56*56,max_pixels=448*448)
    kwargs={"local_files_only":True,"device_map":"auto"}
    if torch.cuda.is_available():
        kwargs.update({"device_map":{"":f"cuda:{settings.cuda_device_index}"},"quantization_config":BitsAndBytesConfig(
            load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)})
    else:
        kwargs["torch_dtype"]=torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading grounded semantic classifier {model_name}",file=sys.stderr,flush=True)
    model=AutoModelForImageTextToText.from_pretrained(resolved,**kwargs)
    codes={"A":"text","B":"button","C":"input","D":"checkbox","E":"slider","F":"dropdown"}
    allowed={}
    for code,kind in codes.items():
        encoded=processor.tokenizer.encode(" "+code,add_special_tokens=False)
        if len(encoded)!=1:
            encoded=processor.tokenizer.encode(code,add_special_tokens=False)
        if len(encoded)!=1:
            raise RuntimeError("Semantic label cannot be represented by one token")
        allowed[encoded[0]]=kind
    results=[]
    for image_index,(path,anchors) in enumerate(zip(payload["image_paths"],payload["anchors"])):
        with Image.open(path) as original:
            image=original.convert("RGB")
        labels={}
        for index,anchor in enumerate(anchors):
            x,y,w,h=map(float,anchor["bbox"])
            # A local context query avoids parroting an example sequence across
            # dozens of regions. Each classification starts a fresh conversation.
            crop=image.crop((max(0,int(x-max(12,w*.3))),max(0,int(y-max(12,h*1.5))),
                             min(image.width,int(x+w+max(12,w*.3))),min(image.height,int(y+h+max(12,h*1.5)))))
            prompt=("Classify the region labeled "+json.dumps(str(anchor.get("text") or "")[:160])+". "
                    "Choose ONE letter: A=static text/heading/artwork, B=clickable button/link, "
                    "C=editable text input, D=checkbox/radio/switch, E=slider, F=dropdown. "
                    "Use the cropped screenshot as context. Reply with one letter only.")
            messages=[{"role":"user","content":[{"type":"image","image":crop},{"type":"text","text":prompt}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            info=process_vision_info(messages)
            inputs=processor(text=[text],images=info[0],padding=True,return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=1,min_new_tokens=1,do_sample=False,
                    prefix_allowed_tokens_fn=lambda batch,input_ids:list(allowed))
            token=int(output[0,-1])
            labels[str(index)]=allowed[token]
            del inputs,output
            if (index+1)%10==0:
                print(f"Image {image_index+1}: classified {index+1}/{len(anchors)} anchors",file=sys.stderr,flush=True)
        if payload.get("mode")=="verify":
            corrections=[{"id":i,"type":labels[str(i)]} for i,a in enumerate(anchors) if a.get("type","text")!=labels[str(i)]]
            result={"approved":not corrections,"corrections":corrections,"missed_elements":[],"deleted_ids":[]}
        else:
            result=labels
        results.append(json.dumps(result))
        print(f"Image {image_index+1}: complete",file=sys.stderr,flush=True)
    print("__VISION_JSON_START__")
    profile.update({'peak_allocated_bytes':torch.cuda.max_memory_allocated(settings.cuda_device_index),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(settings.cuda_device_index)})
    if payload.get('metrics_path'):
        Path(payload['metrics_path']).write_text(json.dumps(profile,indent=2),encoding='utf-8')
    print(json.dumps(results))
    print("__VISION_JSON_END__")


if __name__=="__main__":
    main()
