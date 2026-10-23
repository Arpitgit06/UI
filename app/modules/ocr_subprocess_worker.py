"""
Standalone worker script for executing PaddleOCR in an isolated Windows subprocess.
Crucial behavior: This script forces PaddleOCR to run on CPU to avoid DLL collisions
with PyTorch (WinError 127 on cuDNN 9.5) and to preserve VRAM for other models.
"""
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# WINDOWS DLL CONFLICT FIX: PaddleOCR -> paddlex -> modelscope transitively
# imports torch. If paddle's C++ runtime loads first, it blocks torch's shm.dll
# (WinError 127). We must (1) register torch's DLL directory and (2) import torch
# BEFORE any paddle code runs.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_GPU"] = "0"
os.environ["FLAGS_use_gpu"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "1"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

# WINDOWS DLL CONFLICT FIX: PaddleOCR -> paddlex -> modelscope transitively
# imports torch. If paddle's C++ runtime loads first, it blocks torch's shm.dll
# (WinError 127). We must (1) register torch's DLL directory and (2) import torch
# BEFORE any paddle code runs.
_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    try:
        os.add_dll_directory(_torch_lib)
    except (OSError, AttributeError):
        pass
try:
    import torch  # noqa: F401 — must load torch DLLs before paddle
except ImportError:
    pass

# Setup strict local cache paths before importing paddle/paddleocr
project_root = Path(__file__).resolve().parent.parent.parent
models_cache_dir = project_root / "models_cache"
models_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PADDLEX_HOME", str(models_cache_dir / "paddlex"))
os.environ.setdefault("PADDLE_HOME", str(models_cache_dir / "paddle"))
os.environ.setdefault("HF_HOME", str(models_cache_dir))


def _xyxy_to_bbox_tuple(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


from app.modules.ocr_adapter import _parse_paddleocr_result


def main() -> None:
    # Note: CUDA_VISIBLE_DEVICES, USE_GPU, FLAGS_* are all set at module level
    # (before `import paddle`) to prevent DLL loading issues on Windows.
    try:
        import paddle
        
        paddle.set_flags({
            "FLAGS_enable_pir_in_executor": 0,
            "FLAGS_use_mkldnn": 0,
        })
        from paddleocr import PaddleOCR

        # Always run on CPU: it takes <200ms per image and saves VRAM for YOLO/LLM
        ocr = None

        while True:
            line = sys.stdin.readline()
            if not line:
                break
                
            payload = json.loads(line)
            if payload.get("command") == "shutdown":
                break
                
            image_paths = payload.get("image_paths", [])
            lang = payload.get("lang", "en")
            confidence_threshold = float(payload.get("confidence_threshold", 0.5))
            
            if ocr is None:
                from app.config import settings
                model_options = {}
                if settings.offline:
                    base = models_cache_dir / "paddlex" / "official_models"
                    detections = sorted(base.glob("*det"), reverse=True)
                    recognitions = sorted(base.glob("*rec"), reverse=True)
                    if not detections or not recognitions:
                        raise RuntimeError("OCR models are not cached; run setup.bat online once")
                    model_options = {"text_detection_model_dir":str(detections[0]),
                                     "text_recognition_model_dir":str(recognitions[0]),
                                     "text_detection_model_name":detections[0].name,
                                     "text_recognition_model_name":recognitions[0].name}
                ocr = PaddleOCR(
                    lang=lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                    cpu_threads=4,
                    **model_options,
                    device="cpu",
                )
            
            all_ocr_detections = []
            total_images = len(image_paths)
            for i, img_path in enumerate(image_paths):
                ocr_raw = ocr.predict(img_path)
                ocr_detections = []
                for res in ocr_raw:
                    ocr_detections.extend(_parse_paddleocr_result(res))
                ocr_detections = [d for d in ocr_detections if d.get("confidence", 0.0) >= confidence_threshold]
                all_ocr_detections.append(ocr_detections)
                
                print(f"__OCR_PROGRESS__{i+1}/{total_images}__\n", flush=True)

            response = {"status": "success", "detections": all_ocr_detections}
            print("__OCR_JSON_START__\n" + json.dumps(response) + "\n__OCR_JSON_END__", flush=True)

    except Exception as e:
        response = {"status": "error", "message": str(e)}
        print("__OCR_JSON_START__\n" + json.dumps(response) + "\n__OCR_JSON_END__", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
