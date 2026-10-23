"""One GPU stage per process; no layer/CPU/disk model-offloading hooks."""
import json
import sys
import os
from pathlib import Path
from app.config import settings


def main():
    import torch
    from app.core.model_budget import configure_cuda_stage
    request=json.loads(Path(sys.argv[1]).read_text());phase=sys.argv[2]
    profile=configure_cuda_stage(torch,'ltx-'+phase,settings.cuda_device_index)
    profile['worker_pid']=os.getpid()
    base=Path(request['base_path']);q8=Path(request['q8_path']);out=Path(request['output_dir'])
    if not (base/'model_index.json').exists() or not q8.is_file():
        raise RuntimeError('LTX assets not cached. Run tools/setup_ltx.py in the generation environment first.')
    dtype=torch.float16
    device=f'cuda:{settings.cuda_device_index}'
    if phase=='encode':
        from transformers import T5EncoderModel,AutoTokenizer,BitsAndBytesConfig
        tokenizer=AutoTokenizer.from_pretrained(base,subfolder='tokenizer',local_files_only=True)
        encoder=T5EncoderModel.from_pretrained(base,subfolder='text_encoder',local_files_only=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),device_map={'':device},torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else dtype)
        tokens=tokenizer(request['prompt'],padding='max_length',max_length=128,truncation=True,return_tensors='pt').to(device)
        with torch.inference_mode():embeds=encoder(**tokens).last_hidden_state
        if not torch.isfinite(embeds).all():raise RuntimeError('Text encoder produced non-finite embeddings')
        torch.save({'prompt_embeds':embeds.cpu(),'prompt_attention_mask':tokens.attention_mask.cpu()},out/'prompt.pt')
        profile['model']='T5 INT8 prompt encoder'
    elif phase=='generate':
        from diffusers import LTXImageToVideoPipeline,LTXVideoTransformer3DModel,GGUFQuantizationConfig
        from diffusers.utils import export_to_video
        from PIL import Image
        transformer=LTXVideoTransformer3DModel.from_single_file(str(q8),config=str(base),subfolder='transformer',
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),torch_dtype=dtype,local_files_only=True)
        pipe=LTXImageToVideoPipeline.from_pretrained(base,transformer=transformer,text_encoder=None,tokenizer=None,
            torch_dtype=dtype,local_files_only=True).to(device)
        pipe.vae.enable_tiling()
        embeddings=torch.load(out/'prompt.pt',map_location=device,weights_only=True)
        embeddings['prompt_embeds']=embeddings['prompt_embeds'].to(dtype=dtype)
        image=Image.open(request['reference']).convert('RGB').resize((request['width'],request['height']))
        with torch.inference_mode():
            frames=pipe(image=image,**embeddings,guidance_scale=1.0,width=request['width'],height=request['height'],
                num_frames=request['frames'],num_inference_steps=request['steps'],generator=torch.Generator(device).manual_seed(request['seed'])).frames[0]
        export_to_video(frames,str(out/'generated.mp4'),fps=request['fps'])
        profile['model']='LTX-Video 2B v0.9 '+request.get('precision','q8')+' + VAE'
    else:raise ValueError('Invalid LTX phase')
    profile.update({'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'offloading':False})
    (out/f'{phase}-profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')


if __name__=='__main__':main()
