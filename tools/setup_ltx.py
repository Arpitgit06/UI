"""Download the original LTX-Video 2B FP16/Q8 stack and its required T5/VAE."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import settings
from app.modules.ltx_assets import BASE_REPO,Q8_REPO,Q8_FILE,FP16_FILE


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision',choices=['fp16','q8'],default='fp16')
    args=parser.parse_args()
    from huggingface_hub import hf_hub_download,snapshot_download,HfApi
    root=settings.models_cache_dir/'ltx-2b';root.mkdir(parents=True,exist_ok=True)
    files=HfApi().list_repo_files(BASE_REPO)
    allowed=[]
    for name in files:
        if name=='model_index.json' or name.startswith(('tokenizer/','scheduler/','vae/','text_encoder/')) or name=='transformer/config.json':
            if name.endswith('.bin') and any(p.startswith(name.split('/')[0]+'/') and p.endswith('.safetensors') for p in files):continue
            allowed.append(name)
    print('Downloading base tokenizer/T5/VAE and '+args.precision+' transformer only.',flush=True)
    snapshot_download(BASE_REPO,allow_patterns=allowed,local_dir=root/'base',max_workers=1)
    hf_hub_download(Q8_REPO,FP16_FILE if args.precision=='fp16' else Q8_FILE,local_dir=root)
    print('Cached LTX-Video 2B '+args.precision+'; T5 and the video generator run in separate processes.')


if __name__=='__main__':main()
