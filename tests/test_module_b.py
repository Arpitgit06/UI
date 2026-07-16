"""
Tests for Module B. Two tiers:

  1. Pure functions (bbox math, depth sampling/normalization, PaddleOCR
     result parsing, hex formatting) -- tested directly against
     hand-built stand-ins matching each library's documented output shape.
  2. Full _analyze_state_sync orchestration -- ultralytics/paddleocr/
     transformers aren't installed in the environment this was written
     in, so the module-level `_load_yolo` / `_load_paddleocr` /
     `_load_depth_pipeline` functions are monkeypatched to return small
     fake models with the same call interface. This verifies the GLUE
     logic (guard usage, combining YOLO+OCR into one flat list, z-index
     normalization, cropping) for real; it does not and cannot verify
     that a real YOLOv10/PaddleOCR/Depth-Anything-V2 call behaves the
     way the fakes assume -- see module_b_spatial_vision.py's module
     docstring for exactly what was and wasn't checked against live
     libraries, and what to verify first on a real run.
"""
import numpy as np
import pytest
from PIL import Image

from app.modules import module_b_spatial_vision as module_b
from app.modules.module_b_spatial_vision import (
    PaddleOCRResultError,
    _normalize_z_indices,
    _parse_paddleocr_result,
    _parse_yolo_boxes,
    _rgb_to_hex,
    _sample_depth_region,
    _xyxy_to_bbox_tuple,
)
from app.models.schemas import KeyStateFrame


# --- pure function tests -----------------------------------------------------


def test_xyxy_to_bbox_tuple():
    assert _xyxy_to_bbox_tuple(10, 20, 60, 90) == (10.0, 20.0, 50.0, 70.0)


def test_rgb_to_hex():
    assert _rgb_to_hex(255, 0, 128) == "#ff0080"
    assert _rgb_to_hex(0, 0, 0) == "#000000"


class _FakeYoloBox:
    """Duck-types ultralytics' Boxes-per-detection interface: .xyxy[0], .cls[0], .conf[0]."""

    def __init__(self, xyxy, cls_idx, conf):
        self.xyxy = [xyxy]
        self.cls = [cls_idx]
        self.conf = [conf]


def test_parse_yolo_boxes():
    boxes = [
        _FakeYoloBox([10, 20, 60, 80], 0, 0.91),
        _FakeYoloBox([100, 100, 150, 140], 1, 0.77),
    ]
    names = {0: "button", 1: "input"}

    parsed = _parse_yolo_boxes(boxes, names)

    assert parsed == [
        {"bbox": (10.0, 20.0, 50.0, 60.0), "label": "button", "confidence": pytest.approx(0.91)},
        {"bbox": (100.0, 100.0, 50.0, 40.0), "label": "input", "confidence": pytest.approx(0.77)},
    ]


class _FakePaddleResult:
    """Duck-types a PaddleOCR 3.x result object: a `.json` dict attribute."""

    def __init__(self, json_dict):
        self.json = json_dict


def test_parse_paddleocr_result_matches_documented_shape():
    res = _FakePaddleResult(
        {
            "res": {
                "rec_texts": ["Submit", "Cancel"],
                "rec_boxes": [[10, 200, 80, 220], [100, 200, 170, 220]],
                "rec_scores": [0.95, 0.88],
            }
        }
    )

    parsed = _parse_paddleocr_result(res)

    assert len(parsed) == 2
    assert parsed[0]["text"] == "Submit"
    assert parsed[0]["bbox"] == (10.0, 200.0, 70.0, 20.0)
    assert parsed[0]["confidence"] == pytest.approx(0.95)


def test_parse_paddleocr_result_raises_helpful_error_on_unexpected_shape():
    res = _FakePaddleResult({"unexpected": "shape"})
    with pytest.raises(PaddleOCRResultError, match=r"print\(res\.json\)"):
        _parse_paddleocr_result(res)


def test_sample_depth_region_averages_within_bbox():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[0:50, :] = 10.0
    depth[50:100, :] = 90.0

    assert _sample_depth_region(depth, (0, 0, 100, 50)) == pytest.approx(10.0)
    assert _sample_depth_region(depth, (0, 50, 100, 50)) == pytest.approx(90.0)


def test_sample_depth_region_clamps_out_of_bounds_bbox():
    depth = np.full((10, 10), 5.0, dtype=np.float32)
    # a bbox mostly outside the array shouldn't crash -- clamped to valid indices
    assert _sample_depth_region(depth, (8, 8, 20, 20)) == pytest.approx(5.0)


def test_normalize_z_indices_min_max_scales_to_0_100():
    result = _normalize_z_indices([10.0, 30.0, 50.0])
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(50.0)
    assert result[2] == pytest.approx(100.0)


def test_normalize_z_indices_handles_uniform_depths():
    assert _normalize_z_indices([42.0, 42.0, 42.0]) == [50.0, 50.0, 50.0]


def test_normalize_z_indices_handles_empty_list():
    assert _normalize_z_indices([]) == []


# --- full orchestration test (monkeypatched loaders) ------------------------


class _FakeYoloModel:
    def __init__(self, boxes, names):
        self._boxes = boxes
        self._names = names

    def predict(self, image_path, conf=0.25, verbose=False):
        class _Result:
            pass

        r = _Result()
        r.boxes = self._boxes
        r.names = self._names
        return [r]

    def to(self, device):
        return self


class _FakePaddleOCR:
    def __init__(self, results):
        self._results = results

    def predict(self, image_path):
        return self._results


class _FakeDepthPipeline:
    def __init__(self, depth_image):
        self._depth_image = depth_image
        self.model = self  # so the unloader's p.model.to("cpu") has something to call

    def __call__(self, image):
        return {"depth": self._depth_image}

    def to(self, device):
        return self


def test_analyze_state_sync_combines_yolo_and_ocr_into_flat_list(tmp_path, monkeypatch):
    # a 100x100 test image with a "button" region (top half) and a
    # "text" region (bottom half), so depth sampling should tell them apart
    img_path = tmp_path / "state.png"
    Image.new("RGB", (100, 100), color=(20, 20, 20)).save(img_path)

    fake_yolo = _FakeYoloModel([_FakeYoloBox([10, 10, 90, 40], 0, 0.9)], {0: "button"})
    fake_ocr_result = _FakePaddleResult(
        {"res": {"rec_texts": ["Click me"], "rec_boxes": [[20, 60, 80, 80]], "rec_scores": [0.93]}}
    )
    fake_ocr = _FakePaddleOCR([fake_ocr_result])

    depth_arr = np.zeros((100, 100), dtype=np.uint8)
    depth_arr[0:50, :] = 200  # button region: "closer"
    depth_arr[50:100, :] = 50  # text region: "farther"
    fake_depth = _FakeDepthPipeline(Image.fromarray(depth_arr))

    monkeypatch.setattr(module_b, "_load_yolo", lambda *a, **kw: fake_yolo)
    monkeypatch.setattr(module_b, "_load_paddleocr", lambda *a, **kw: fake_ocr)
    monkeypatch.setattr(module_b, "_load_depth_pipeline", lambda *a, **kw: fake_depth)

    state = KeyStateFrame(frame_index=0, timestamp_sec=0.0, image_path=str(img_path))
    elements = module_b._analyze_state_sync(state)

    assert len(elements) == 2
    button_el = next(e for e in elements if e.element_type == "button")
    text_el = next(e for e in elements if e.element_type == "text")

    assert button_el.bbox.width == pytest.approx(80.0)
    assert text_el.text_content == "Click me"
    assert button_el.confidence == pytest.approx(0.9)
    assert text_el.confidence == pytest.approx(0.93)
    # the "closer" button region should get a higher z_index than the "farther" text region
    assert button_el.bbox.z_index > text_el.bbox.z_index
