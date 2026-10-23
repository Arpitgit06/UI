"""Reuse real OCR only when image bytes, language, threshold and model version match."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from app.config import settings


def cache_path(image, lang, threshold):
    identity = json.dumps({"adapter":2,"lang":lang,"threshold":threshold,
                           "paddleocr":version("paddleocr"),"paddlex":version("paddlex")},sort_keys=True)
    digest = hashlib.sha256(identity.encode()+Path(image).read_bytes()).hexdigest()
    return settings.storage_root / "ocr_cache" / f"{digest}.json"


def load(image,lang,threshold):
    path=cache_path(image,lang,threshold)
    try:
        result=json.loads(path.read_text(encoding="utf-8"))
        if isinstance(result,list):
            return result
    except (OSError,ValueError):
        pass
    return None


def save(image,lang,threshold,result):
    path=cache_path(image,lang,threshold)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=True),encoding="utf-8")
