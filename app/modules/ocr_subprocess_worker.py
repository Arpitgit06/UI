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

# BULLETPROOF DLL FIX: PyTorch and PaddlePaddle both ship with cuDNN DLLs.
# If Paddle loads first, it pollutes the DLL search path and crashes Torch
# with WinError 127. By eagerly importing Torch here at the very top, we
# force Windows to load Torch's DLLs into memory safely before Paddle starts.
try:
    import torch
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


def _parse_paddleocr_result(res) -> list[dict]:
    if res is None:
        return []

    # Case 1: Classic list-of-lines `[[box, [text, score]], ...]`
    if isinstance(res, (list, tuple)):
        parsed = []
        for item in res:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            box, (text, score) = item[0], item[1]
            if isinstance(box, (list, tuple)) and len(box) == 4:
                if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in box):
                    x1 = min(float(p[0]) for p in box)
                    y1 = min(float(p[1]) for p in box)
                    x2 = max(float(p[0]) for p in box)
                    y2 = max(float(p[1]) for p in box)
                elif all(isinstance(v, (int, float)) for v in box):
                    x1, y1, x2, y2 = [float(v) for v in box]
                else:
                    continue
                parsed.append(
                    {
                        "text": text,
                        "confidence": float(score),
                        "box": _xyxy_to_bbox_tuple(x1, y1, x2, y2),
                    }
                )
        return parsed
    
    # Case 2: Dict format (paddlex v3 style)
    parsed = []
    if hasattr(res, "get"):
        text_polys = res.get("text_polys", [])
        rec_text = res.get("rec_text", [])
        rec_score = res.get("rec_score", [])
        
        for i in range(min(len(text_polys), len(rec_text), len(rec_score))):
            poly = text_polys[i]
            if not isinstance(poly, (list, tuple)) or len(poly) != 4:
                continue
            x1 = min(float(p[0]) for p in poly)
            y1 = min(float(p[1]) for p in poly)
            x2 = max(float(p[0]) for p in poly)
            y2 = max(float(p[1]) for p in poly)
            
            parsed.append(
                {
                    "text": rec_text[i],
                    "confidence": float(rec_score[i]),
                    "box": _xyxy_to_bbox_tuple(x1, y1, x2, y2),
                }
            )
    return parsed


def main() -> None:
    if len(sys.argv) < 5:
        sys.exit(1)

    image_path = sys.argv[1]
    # We ignore sys.argv[2] (device) and force CPU to avoid PyTorch DLL conflicts
    lang = sys.argv[3]
    confidence_threshold = float(sys.argv[4])

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["USE_GPU"] = "0"
    # Disable PIR executor — crashes with oneDNN on CPU/Windows
    # See: https://github.com/PaddlePaddle/Paddle/issues/70255
    os.environ["FLAGS_enable_pir_in_executor"] = "0"
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

    try:
        import paddle
        
        paddle.set_flags({
            "FLAGS_enable_pir_in_executor": 0,
            "FLAGS_use_mkldnn": 0,
        })
        from paddleocr import PaddleOCR

        # Always run on CPU: it takes <200ms per image and saves VRAM for YOLO/LLM
        ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="cpu",
        )
        
        ocr_raw = ocr.predict(image_path)
        ocr_detections = []
        for res in ocr_raw:
            ocr_detections.extend(_parse_paddleocr_result(res))
        ocr_detections = [d for d in ocr_detections if d.get("confidence", 0.0) >= confidence_threshold]

        print("__OCR_JSON_START__")
        print(json.dumps(ocr_detections))
        print("__OCR_JSON_END__")
    except Exception as e:
        print(f"PaddleOCR Subprocess Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
