"""Normalize real PaddleOCR 2/3 results without importing either ML framework."""
import json
import math


class PaddleOCRResultError(ValueError):
    pass


def _parse_paddleocr_result(result) -> list[dict]:
    if result is None:
        raise PaddleOCRResultError("OCR returned None instead of a result")
    if hasattr(result, "json"):
        result = result.json
        if callable(result):
            result = result()
    if isinstance(result, str):
        result = json.loads(result)
    if isinstance(result, dict) and "res" in result:
        result = result["res"]

    def box(values):
        if hasattr(values, "tolist"):
            values = values.tolist()
        if len(values) != 4:
            raise PaddleOCRResultError("OCR box must have four corners or xyxy values")
        if isinstance(values[0], (list, tuple)):
            xs, ys = zip(*values)
            values = [min(xs), min(ys), max(xs), max(ys)]
        x1, y1, x2, y2 = map(float, values)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            raise PaddleOCRResultError("OCR returned invalid box geometry")
        return (x1, y1, x2-x1, y2-y1)

    if isinstance(result, dict):
        if not {"rec_texts", "rec_scores"}.issubset(result):
            raise PaddleOCRResultError("Unknown PaddleOCR format; inspect print(res.json)")
        boxes = result.get("rec_boxes", result.get("rec_polys"))
        texts, scores = result["rec_texts"], result["rec_scores"]
        if boxes is None or not len(boxes) == len(texts) == len(scores):
            raise PaddleOCRResultError("OCR text, confidence and box counts differ")
        return [{"text": str(t), "confidence": float(s), "bbox": box(b)}
                for t, s, b in zip(texts, scores, boxes)]
    if isinstance(result, (list, tuple)):
        return [{"text": str(item[1][0]), "confidence": float(item[1][1]), "bbox": box(item[0])}
                for item in result]
    raise PaddleOCRResultError("Unknown PaddleOCR result object; inspect print(res.json)")
