"""
Tests for Module C. Unlike Modules A/B, this module has no ML/GPU
dependencies at all -- it's pure geometry plus a PIL image-size read --
so every test here runs against the real, unmodified implementation.
No stand-ins, no monkeypatching.
"""
import asyncio
import json

import pytest
from PIL import Image

from app.models.schemas import BoundingBox, DetectedElement, KeyStateFrame
from app.modules.module_c_dom_synthesizer import (
    _area,
    _assign_parents,
    _containment_ratio,
    _intersection_area,
    _root_bbox_from_image,
    _synthesize_dom_sync,
    _tag_hint_for,
    synthesize_dom,
    write_layout_json,
)


def _el(id_, x, y, w, h, etype="div", text=None, z=None, conf=0.9):
    return DetectedElement(
        element_id=id_,
        element_type=etype,
        bbox=BoundingBox(x=x, y=y, width=w, height=h, z_index=z),
        text_content=text,
        confidence=conf,
    )


def _frame(image_path, frame_index=0):
    return KeyStateFrame(frame_index=frame_index, timestamp_sec=0.0, image_path=image_path)


# --- geometry primitives -----------------------------------------------------


def test_area_and_intersection():
    outer = BoundingBox(x=0, y=0, width=100, height=100)
    inner = BoundingBox(x=25, y=25, width=50, height=50)
    assert _area(outer) == 10000.0
    assert _intersection_area(outer, inner) == 2500.0
    assert _containment_ratio(inner, outer) == 1.0


def test_containment_ratio_partial_and_zero_overlap():
    outer = BoundingBox(x=0, y=0, width=100, height=100)
    mostly_outside = BoundingBox(x=90, y=90, width=50, height=50)
    no_overlap = BoundingBox(x=200, y=200, width=10, height=10)
    assert 0.0 < _containment_ratio(mostly_outside, outer) < 1.0
    assert _containment_ratio(no_overlap, outer) == 0.0


def test_tag_hint_mapping():
    assert _tag_hint_for("button") == "button"
    assert _tag_hint_for("Text Button") == "button"  # normalizes case/spaces
    assert _tag_hint_for("person") == "div"  # unrecognized (e.g. stock COCO class) -> fallback


# --- parent assignment --------------------------------------------------------


def test_simple_containment_text_inside_button():
    button = _el("btn-0", 10, 10, 100, 40, etype="button")
    text = _el("ocr-0", 20, 20, 60, 20, etype="text", text="Click me")

    parent_of = _assign_parents([button, text], containment_threshold=0.8)

    assert parent_of["ocr-0"] == "btn-0"
    assert parent_of["btn-0"] is None


def test_multi_level_nesting_attaches_to_tightest_parent():
    card = _el("card-0", 0, 0, 500, 500, etype="card")
    button = _el("btn-1", 20, 20, 100, 40, etype="button")
    text = _el("ocr-1", 30, 30, 60, 20, etype="text", text="Submit")

    parent_of = _assign_parents([card, button, text], containment_threshold=0.8)

    assert parent_of["ocr-1"] == "btn-1", "text should attach to its direct (tightest) parent, not skip to the card"
    assert parent_of["btn-1"] == "card-0"
    assert parent_of["card-0"] is None


def test_unrelated_elements_become_siblings():
    navbar = _el("nav-0", 0, 0, 500, 50, etype="navbar")
    card = _el("card-1", 0, 60, 500, 400, etype="card")

    parent_of = _assign_parents([navbar, card], containment_threshold=0.8)

    assert parent_of["nav-0"] is None
    assert parent_of["card-1"] is None


def test_near_identical_boxes_do_not_nest_into_each_other():
    box_a = _el("a-0", 10, 10, 50, 20, etype="button")
    box_b = _el("b-0", 10, 10, 50, 20, etype="text", text="label")  # identical bbox

    parent_of = _assign_parents([box_a, box_b], containment_threshold=0.8)

    assert parent_of["a-0"] is None
    assert parent_of["b-0"] is None


def test_partial_overlap_below_threshold_does_not_nest():
    big = _el("big-0", 0, 0, 100, 100, etype="card")
    mostly_outside = _el("half-0", 80, 80, 40, 40, etype="text", text="edge")  # ~1/4 overlaps

    parent_of = _assign_parents([big, mostly_outside], containment_threshold=0.8)

    assert parent_of["half-0"] is None


# --- tree assembly / z-ordering / root bbox ----------------------------------


def test_children_sorted_by_z_index_ascending():
    parent = _el("p-0", 0, 0, 200, 200, etype="card")
    far = _el("c-far", 10, 10, 30, 30, etype="text", z=10.0)
    near = _el("c-near", 50, 50, 30, 30, etype="text", z=90.0)

    layout = _synthesize_dom_sync(_frame("/nonexistent.png"), [parent, far, near])

    parent_node = layout.root.children[0]
    assert [c.node_id for c in parent_node.children] == ["c-far", "c-near"]


def test_empty_detections_produce_a_childless_root():
    layout = _synthesize_dom_sync(_frame("/nonexistent.png"), [])
    assert layout.root.children == []


def test_root_bbox_uses_real_image_dimensions(tmp_path):
    img_path = tmp_path / "state.png"
    Image.new("RGB", (640, 480)).save(img_path)

    root_bbox = _root_bbox_from_image(str(img_path))

    assert root_bbox.width == 640
    assert root_bbox.height == 480


def test_root_bbox_falls_back_to_union_when_image_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.png"
    assert _root_bbox_from_image(str(missing_path)) is None

    layout = _synthesize_dom_sync(_frame(str(missing_path)), [_el("x-0", 5, 5, 20, 20)])
    assert layout.root.bbox.width == 20
    assert layout.root.bbox.height == 20


def test_duplicate_element_id_is_deduplicated_not_crashed(tmp_path):
    img_path = tmp_path / "state.png"
    Image.new("RGB", (200, 200)).save(img_path)
    dup_a = _el("dup-0", 10, 10, 50, 20, etype="button")
    dup_b = _el("dup-0", 10, 10, 50, 20, etype="button")  # same id, would corrupt the tree if not caught

    layout = _synthesize_dom_sync(_frame(str(img_path)), [dup_a, dup_b])

    assert len(layout.root.children) == 1


# --- JSON export --------------------------------------------------------------


def test_write_layout_json_round_trips(tmp_path):
    img_path = tmp_path / "state.png"
    Image.new("RGB", (300, 200)).save(img_path)
    button = _el("btn-0", 10, 10, 100, 40, etype="button")
    text = _el("ocr-0", 20, 20, 60, 20, etype="text", text="Click me")

    layout = _synthesize_dom_sync(_frame(str(img_path), frame_index=3), [button, text])
    json_path = write_layout_json(layout, tmp_path / "layouts")

    assert json_path.name == "state_3_layout.json"
    with open(json_path) as f:
        parsed = json.load(f)
    assert "root" in parsed
    assert len(parsed["root"]["children"]) == 1
    assert parsed["root"]["children"][0]["children"][0]["text_content"] == "Click me"


# --- async entrypoint ----------------------------------------------------------


def test_synthesize_dom_async_writes_file_when_output_dir_given(tmp_path):
    img_path = tmp_path / "state.png"
    Image.new("RGB", (300, 200)).save(img_path)
    state = _frame(str(img_path), frame_index=7)
    elements = [_el("btn-0", 10, 10, 100, 40, etype="button")]
    out_dir = tmp_path / "layouts"

    layout = asyncio.run(synthesize_dom(state, elements, out_dir))

    assert layout.root.node_id == "root"
    assert (out_dir / "state_7_layout.json").exists()


def test_synthesize_dom_async_skips_write_when_output_dir_none(tmp_path):
    img_path = tmp_path / "state.png"
    Image.new("RGB", (300, 200)).save(img_path)
    state = _frame(str(img_path))

    layout = asyncio.run(synthesize_dom(state, [], output_dir=None))

    assert layout.root.node_id == "root"
