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
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(self.models_cache_dir / "browsers"))
        os.environ.setdefault("PADDLEX_HOME", str(self.models_cache_dir / "paddlex"))
        os.environ.setdefault("PADDLE_HOME", str(self.models_cache_dir / "paddle"))
        os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(self.models_cache_dir / "ultralytics"))

        # VRAM safety
        self.min_free_vram_mb = float(os.environ.get("OMNIUI_MIN_FREE_VRAM_MB", 512.0))
        self.cuda_device_index = int(os.environ.get("OMNIUI_CUDA_DEVICE", 0))

        # Module B: Spatial Vision & Detection (Hybrid ONNX Architecture)
        # Macro model detects large layout containers (cards, modals)
        self.yolo_macro_weights_path = os.environ.get(
            "OMNIUI_YOLO_MACRO_WEIGHTS",
            str(self.models_cache_dir / "yolov8_macro.pt"),
        )
        # Micro model detects dense nested elements (icons, text inputs)
        self.yolo_micro_weights_path = os.environ.get(
            "OMNIUI_YOLO_MICRO_WEIGHTS",
            str(self.models_cache_dir / "yolov10_micro.pt"),
        )
        self.yolo_export_onnx = os.environ.get("OMNIUI_YOLO_EXPORT_ONNX", "true").lower() == "true"
        self.yolo_on_cpu = os.environ.get("OMNIUI_YOLO_ON_CPU", "true").lower() == "true"
        
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

        # Models execute in separate, sequential processes. Only the Qwen stages use NF4.
        self.vision_detector_model_name = os.environ.get("OMNIUI_VISION_DETECTOR_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
        self.local_llm_model_name = os.environ.get("OMNIUI_CODE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.code_generator = os.environ.get("OMNIUI_CODE_GENERATOR", "qwen")
        self.local_llm_load_in_4bit = os.environ.get("OMNIUI_LLM_4BIT", "true").lower() == "true"
        self.local_llm_temperature = float(os.environ.get("OMNIUI_LLM_TEMPERATURE", 0.1))
        self.local_llm_max_new_tokens = int(os.environ.get("OMNIUI_LLM_MAX_TOKENS", 4096))
        self.local_llm_max_retries = int(os.environ.get("OMNIUI_LLM_MAX_RETRIES", 2))
        self.local_llm_strict_validation = os.environ.get("OMNIUI_LLM_STRICT_VALIDATION", "true").lower() == "true"
        self.offline = os.environ.get("OMNIUI_OFFLINE", "true").lower() == "true"
        self.worker_timeout_sec = int(os.environ.get("OMNIUI_WORKER_TIMEOUT_SEC", 600))
        self.vision_max_new_tokens = int(os.environ.get("OMNIUI_VISION_MAX_TOKENS", 4096))
        self.vision_max_pixels = int(os.environ.get("OMNIUI_VISION_MAX_PIXELS", 1280 * 720))
        self.max_upload_bytes = int(os.environ.get("OMNIUI_MAX_UPLOAD_MB", 512)) * 1024**2
        self.max_video_seconds = int(os.environ.get("OMNIUI_MAX_VIDEO_SECONDS", 600))
        self.max_video_pixels = int(os.environ.get("OMNIUI_MAX_VIDEO_PIXELS", 3840 * 2160))
        self.max_pending_jobs = int(os.environ.get("OMNIUI_MAX_PENDING_JOBS", 8))
        self.max_key_states = int(os.environ.get("OMNIUI_MAX_KEY_STATES", 12))
        self.motion_max_samples = int(os.environ.get('OMNIUI_MOTION_MAX_SAMPLES',80))
        self.motion_max_regions = int(os.environ.get('OMNIUI_MOTION_MAX_REGIONS',4))
        self.artwork_generator=os.environ.get('OMNIUI_ARTWORK_GENERATOR','ltx-2b-fp16')
        self.visual_verification = os.environ.get("OMNIUI_VISUAL_VERIFICATION", "true").lower() == "true"
        self.require_visual_verification = os.environ.get("OMNIUI_REQUIRE_VISUAL_VERIFICATION", "true").lower() == "true"

    def ensure_directories(self) -> None:
        for d in (self.uploads_dir, self.jobs_dir, self.outputs_dir, self.models_cache_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
