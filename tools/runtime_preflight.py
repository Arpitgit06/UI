"""Validate offline assets and dependencies without loading GPU models."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def weights_present(directory):
    directory = Path(directory)
    indexes = list(directory.glob('*.safetensors.index.json'))
    if indexes:
        try:
            shards = set(json.loads(indexes[0].read_text(encoding='utf-8'))['weight_map'].values())
            return bool(shards) and all((directory/name).is_file() and (directory/name).stat().st_size > 1024 for name in shards)
        except (KeyError, ValueError, OSError):
            return False
    return any(path.stat().st_size > 1024 for path in directory.glob('*.safetensors'))


def check_runtime(root):
    import sys
    sys.path.insert(0, str(root))
    from app.config import settings
    from app.core.model_budget import SUPPORTED_MODEL, CODER_MODEL
    errors = []
    for module in ('torch', 'torchvision', 'paddle', 'paddleocr', 'transformers', 'bitsandbytes', 'uvicorn', 'diffusers', 'gguf', 'sentencepiece', 'imageio_ffmpeg'):
        if importlib.util.find_spec(module) is None:
            errors.append(f'Missing Python dependency: {module}')
    if settings.vision_detector_model_name != SUPPORTED_MODEL or settings.local_llm_model_name != CODER_MODEL:
        errors.append('Stale model choices in .env; run setup.bat to migrate to VL-7B + Coder-7B')
    for model in (SUPPORTED_MODEL, CODER_MODEL):
        folder = settings.models_cache_dir/'hub'/('models--'+model.replace('/', '--'))/'snapshots'
        if not any((snapshot/'config.json').is_file() and weights_present(snapshot) for snapshot in folder.glob('*')):
            errors.append(f'Missing/incomplete cached model: {model}')
    if settings.code_generator not in {'qwen', 'deterministic'}:
        errors.append('OMNIUI_CODE_GENERATOR must be qwen or deterministic')
    if settings.artwork_generator != 'preserve':
        from app.modules.ltx_assets import FP16_FILE, Q8_FILE
        base = settings.models_cache_dir/'ltx-2b'/'base'
        filename = FP16_FILE if settings.artwork_generator == 'ltx-2b-fp16' else Q8_FILE
        if settings.artwork_generator not in {'ltx-2b-fp16', 'ltx-2b-q8'}:
            errors.append('Unsupported artwork generator')
        for relative in ('model_index.json', 'tokenizer/spiece.model', 'scheduler/scheduler_config.json', 'transformer/config.json', 'text_encoder/config.json', 'vae/config.json'):
            if not (base/relative).is_file():
                errors.append(f'Missing LTX component: {relative}')
        if not weights_present(base/'text_encoder') or not weights_present(base/'vae'):
            errors.append('Missing/incomplete LTX T5 or VAE weights')
        path = base.parent/filename
        if not path.is_file() or path.stat().st_size < 1024**3:
            errors.append(f'Missing/incomplete LTX transformer: {filename}')
    node = shutil.which('node')
    frontend = root/'tools'/'frontend'
    if not node:
        errors.append('Node.js is missing')
    elif not (frontend/'node_modules/esbuild/package.json').is_file() or not (frontend/'node_modules/playwright/package.json').is_file():
        errors.append('Browser/build dependencies missing; run setup.bat')
    elif settings.visual_verification:
        script = "const fs=require('fs');const p=process.env.OMNIUI_CHROMIUM_PATH||require('playwright').chromium.executablePath();process.exit(fs.existsSync(p)?0:1)"
        result = subprocess.run([node, '-e', script], cwd=frontend, capture_output=True, timeout=15)
        if result.returncode:
            errors.append('Chromium missing; run setup.bat or set OMNIUI_CHROMIUM_PATH')
    return errors
