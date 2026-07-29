"""
Central configuration for OmniUI. Values can be overridden via
environment variables (see .env.example) without touching code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op if .env doesn't exist


class Settings:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.storage_root = self.project_root / "storage"
        self.uploads_dir = self.storage_root / "uploads"
        self.jobs_dir = self.storage_root / "jobs"
        self.outputs_dir = self.storage_root / "outputs"
        self.models_cache_dir = self.project_root / "models_cache"
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("HF_HOME", str(self.models_cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.models_cache_dir))
        os.environ.setdefault("TORCH_HOME", str(self.models_cache_dir))
        os.environ.setdefault("PADDLEX_HOME", str(self.models_cache_dir / "paddlex"))
        os.environ.setdefault("PADDLE_HOME", str(self.models_cache_dir / "paddle"))
        os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(self.models_cache_dir / "ultralytics"))

        # Auto-migrate yolov10n.pt from root directory into models_cache, or remove duplicate from root
        root_yolo = self.project_root / "yolov10n.pt"
        cache_yolo = self.models_cache_dir / "yolov10n.pt"
        if root_yolo.exists():
            try:
                if not cache_yolo.exists():
                    import shutil
                    shutil.move(str(root_yolo), str(cache_yolo))
                else:
                    root_yolo.unlink()
            except Exception:
                pass

        # Auto-migrate ~/.paddlex official models into project models_cache/paddlex if present outside
        user_paddlex = Path.home() / ".paddlex"
        cache_paddlex = self.models_cache_dir / "paddlex"
        if user_paddlex.exists() and not cache_paddlex.exists():
            try:
                import shutil
                shutil.copytree(str(user_paddlex), str(cache_paddlex), dirs_exist_ok=True)
            except Exception:
                pass

        # VRAM safety
        self.min_free_vram_mb = float(os.environ.get("OMNIUI_MIN_FREE_VRAM_MB", 512.0))
        self.cuda_device_index = int(os.environ.get("OMNIUI_CUDA_DEVICE", 0))

        # Module B: Spatial Vision & Detection
        # NOTE on yolo_weights_path: the stock "yolov10n.pt" checkpoint is
        # COCO-pretrained (person/car/dog/...) -- it does NOT know what a
        # button or navbar is. Real UI detection needs a checkpoint
        # fine-tuned on a UI dataset (e.g. Rico). Point this at that
        # checkpoint (under models_cache/) once you have one; until then,
        # this wires up the real inference/VRAM-gate plumbing against
        # placeholder weights so the pipeline is testable end-to-end.
        self.yolo_weights_path = os.environ.get(
            "OMNIUI_YOLO_WEIGHTS",
            str(self.models_cache_dir / "yolov10n.pt"),
        )
        self.yolo_confidence_threshold = float(os.environ.get("OMNIUI_YOLO_CONF", 0.25))
        self.paddleocr_lang = os.environ.get("OMNIUI_OCR_LANG", "en")
        self.ocr_confidence_threshold = float(os.environ.get("OMNIUI_OCR_CONF", 0.5))
        self.depth_model_name = os.environ.get(
            "OMNIUI_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"
        )
        self.colorgram_colors_per_element = int(os.environ.get("OMNIUI_COLORS_PER_ELEMENT", 3))

        # Module C: DOM Synthesizer
        # Fraction of a candidate child's area that must overlap a candidate
        # parent for it to count as "contained" -- see module_c's docstring
        # for why this is a ratio rather than strict corner-containment.
        self.dom_containment_threshold = float(os.environ.get("OMNIUI_DOM_CONTAINMENT_THRESHOLD", 0.8))

        # Local LLM (Module D) — loaded directly via Hugging Face / transformers
        self.local_llm_model_name = os.environ.get("OMNIUI_LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.vision_llm_model_name = os.environ.get("OMNIUI_VISION_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
        self.local_llm_load_in_4bit = os.environ.get("OMNIUI_LLM_4BIT", "true").lower() == "true"
        self.local_llm_temperature = float(os.environ.get("OMNIUI_LLM_TEMPERATURE", 0.1))
        self.local_llm_max_new_tokens = int(os.environ.get("OMNIUI_LLM_MAX_TOKENS", 4096))
        self.local_llm_max_retries = int(os.environ.get("OMNIUI_LLM_MAX_RETRIES", 1))
        self.local_llm_strict_validation = os.environ.get("OMNIUI_LLM_STRICT_VALIDATION", "false").lower() == "true"

    def ensure_directories(self) -> None:
        for d in (self.uploads_dir, self.jobs_dir, self.outputs_dir, self.models_cache_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
