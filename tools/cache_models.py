"""Explicit online setup; runtime inference never downloads missing weights."""
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from app.config import settings


def main():
    from huggingface_hub import snapshot_download
    from app.core.model_budget import SUPPORTED_MODEL,CODER_MODEL
    if (settings.vision_detector_model_name,settings.local_llm_model_name)!=(SUPPORTED_MODEL,CODER_MODEL):
        raise SystemExit('Stale model configuration: run setup.bat to configure VL-7B and Coder-7B.')
    for model in [SUPPORTED_MODEL,CODER_MODEL]:
        print(f"Caching {model}",flush=True)
        snapshot_download(repo_id=model,cache_dir=str(settings.models_cache_dir/"hub"),max_workers=1)
    # Keep torch DLL registration before paddle on Windows.
    import torch
    from paddleocr import PaddleOCR
    PaddleOCR(lang=settings.paddleocr_lang,device="cpu",enable_mkldnn=False,cpu_threads=4,
              use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False)
    print("VL-7B, Coder-7B and CPU OCR cached. LTX is downloaded by setup_ltx.py.",flush=True)


if __name__ == "__main__":
    main()
