"""
Tests for Module D. Two tiers:

  1. Pure/deterministic functions (class name sanitization, percentage/3D
     coordinate math, CSS block extraction, the validator itself against
     both a faithful and a deliberately corrupted response) -- tested
     directly against the real implementation.
  2. Full generate_code() orchestration -- we mock `_call_local_llm_structured`
     with a hand-built fake. This verifies retry-on-validation-failure,
     the 3D branch, and general error handling -- it does not and cannot verify 
     that a real local LLM would actually produce valid output for a real prompt.

Async entrypoints are exercised with asyncio.run() inside plain sync
test functions (matching tests/test_module_c.py) rather than
pytest-asyncio, to avoid a dependency the rest of the suite doesn't need.
"""
import asyncio
import json

import pytest

from app.config import settings
from app.models.schemas import BoundingBox, DOMNode, KeyStateFrame, LayoutState
from app.modules import module_d_code_generator as module_d
from app.modules.module_d_code_generator import (
    LocalLLMConnectionError,
    LocalLLMValidationError,
    _build_code_ready_tree,
    _build_index_scaffold,
    _build_package_json,
    _collect_class_names,
    _extract_css_block,
    _node_css,
    _pascal_case,
    _project_to_3d,
    _sanitize_class_name,
    _validate_component,
    generate_code,
    package_output,
)

ROOT_BBOX = BoundingBox(x=0, y=0, width=1000, height=500)
CHILD_BBOX = BoundingBox(x=100, y=50, width=200, height=100, z_index=75.0)


# --- pure function tests -----------------------------------------------------


def test_sanitize_class_name():
    assert _sanitize_class_name("state_0-btn-0") == "state_0-btn-0"
    assert _sanitize_class_name("a b/c!d") == "a-b-c-d"
    assert _sanitize_class_name("0abc")[0].isalpha()


def test_pascal_case():
    assert _pascal_case("state_0") == "State0"
    assert _pascal_case("---") == "State"  # empty after stripping -> fallback


def test_project_to_3d_scales_correctly():
    proj = _project_to_3d(CHILD_BBOX, ROOT_BBOX)
    assert set(proj.keys()) == {"position", "width", "height"}
    assert proj["width"] == pytest.approx(200 / 1000 * 10.0)


def test_node_css_root_uses_relative_fixed_size():
    root_node = DOMNode(node_id="root", tag_hint="div", bbox=ROOT_BBOX)
    css = _node_css(root_node, None)
    assert css["position"] == "relative"
    assert css["width"].endswith("px")


def test_node_css_child_uses_absolute_percentages():
    child = DOMNode(node_id="btn", tag_hint="button", bbox=CHILD_BBOX, hex_colors=["#ff0000"])
    css = _node_css(child, ROOT_BBOX)
    assert css["position"] == "absolute"
    assert css["left"] == "10.00%"
    assert css["background-color"] == "#ff0000"  # no text -> background, not foreground color


def test_node_css_text_node_uses_foreground_color():
    text_node = DOMNode(node_id="txt", tag_hint="span", bbox=CHILD_BBOX, text_content="Hi", hex_colors=["#00ff00"])
    css = _node_css(text_node, ROOT_BBOX)
    assert css["color"] == "#00ff00"
    assert "background-color" not in css


def _sample_tree() -> DOMNode:
    return DOMNode(
        node_id="root",
        tag_hint="div",
        bbox=ROOT_BBOX,
        children=[
            DOMNode(
                node_id="btn-0",
                tag_hint="button",
                bbox=CHILD_BBOX,
                hex_colors=["#123456"],
                children=[
                    DOMNode(
                        node_id="ocr-0",
                        tag_hint="span",
                        bbox=BoundingBox(x=110, y=60, width=50, height=20, z_index=80.0),
                        text_content="Click",
                    )
                ],
            )
        ],
    )


def test_build_code_ready_tree_prefixes_class_names_and_nests():
    code_tree = _build_code_ready_tree(_sample_tree(), "state_0")
    assert code_tree["class_name"] == "state_0-root"
    assert len(code_tree["children"]) == 1
    assert len(code_tree["children"][0]["children"]) == 1
    assert _collect_class_names(code_tree) == {"state_0-root", "state_0-btn-0", "state_0-ocr-0"}


def test_package_json_includes_r3f_only_when_needed():
    assert "@react-three/fiber" not in json.loads(_build_package_json(False))["dependencies"]
    assert "@react-three/fiber" in json.loads(_build_package_json(True))["dependencies"]


def test_index_scaffold_imports_every_state():
    frame = KeyStateFrame(frame_index=0, timestamp_sec=0.0, image_path="/x.png")
    layout = LayoutState(state_name="state_0", source_frame=frame, root=_sample_tree())
    src = _build_index_scaffold([layout])
    assert "import State0 from './State0';" in src


def test_extract_css_block():
    css = ".a { position: relative; width: 1000px; }\n.b { position: absolute; left: 10.00%; }"
    assert "left: 10.00%" in _extract_css_block(css, "b")
    assert _extract_css_block(css, "missing") is None


# --- validation: faithful vs. corrupted --------------------------------------


def _build_faithful_response(node: dict) -> tuple[str, str]:
    def walk_css(n):
        rule = f".{n['class_name']} {{ " + " ".join(f"{k}: {v};" for k, v in n["css"].items()) + " }"
        return [rule] + [r for c in n["children"] for r in walk_css(c)]

    def walk_jsx(n):
        text = n["text"] or ""
        kids = "".join(walk_jsx(c) for c in n["children"])
        return f'<{n["tag"]} className="{n["class_name"]}">{text}{kids}</{n["tag"]}>'

    return walk_jsx(node), "\n".join(walk_css(node))


def test_validate_component_accepts_faithful_response():
    code_tree = _build_code_ready_tree(_sample_tree(), "state_0")
    jsx, css = _build_faithful_response(code_tree)
    assert _validate_component(code_tree, jsx, css) == []


def test_validate_component_catches_corrupted_value_even_with_coincidental_duplicate():
    # btn-0's left AND top both happen to be "10.00%" here -- a naive
    # "is this bare value present anywhere in the block" check would miss
    # a corrupted `left` because `top`'s correct value masks it. This is
    # exactly the false negative caught and fixed during development;
    # guarding against a regression.
    code_tree = _build_code_ready_tree(_sample_tree(), "state_0")
    jsx, css = _build_faithful_response(code_tree)
    corrupted_css = css.replace("left: 10.00%;", "left: 55.00%;")

    problems = _validate_component(code_tree, jsx, corrupted_css)

    assert any("left" in p and "10.00%" in p for p in problems)


def test_validate_component_catches_missing_class_reference():
    code_tree = _build_code_ready_tree(_sample_tree(), "state_0")
    jsx, css = _build_faithful_response(code_tree)
    missing_class_jsx = jsx.replace('className="state_0-ocr-0"', 'className="totally-different"')

    problems = _validate_component(code_tree, missing_class_jsx, css)

    assert len(problems) > 0


# --- full orchestration with a fake LLM ----------------------------


class _FakeLLM:
    def __init__(self, responses=None, raise_exc=None):
        self.calls = []
        self._responses = list(responses) if responses else []
        self._raise_exc = raise_exc

    async def call(self, model, system_prompt, user_content, schema, temperature, load_in_4bit):
        self.calls.append({"model": model})
        if self._raise_exc:
            raise self._raise_exc
        if not self._responses:
            raise RuntimeError("fake LLM ran out of canned responses")
        return json.loads(self._responses.pop(0))


def _make_layout(state_name: str, is_3d: bool = False) -> LayoutState:
    root = DOMNode(
        node_id="root",
        tag_hint="div",
        bbox=ROOT_BBOX,
        children=[DOMNode(node_id="btn-0", tag_hint="button", bbox=CHILD_BBOX, hex_colors=["#123456"])],
    )
    frame = KeyStateFrame(frame_index=0, timestamp_sec=0.0, image_path="/x.png")
    return LayoutState(state_name=state_name, source_frame=frame, root=root, is_3d_scene=is_3d)





def test_happy_path_produces_expected_files(monkeypatch):
    layout = _make_layout("state_0")
    tree = _build_code_ready_tree(layout.root, layout.state_name)
    jsx, css = _build_faithful_response(tree)
    fake_llm = _FakeLLM(responses=[json.dumps({"component_jsx": jsx, "styles_css": css})])
    monkeypatch.setattr(module_d, "_call_local_llm_structured", fake_llm.call)

    files = asyncio.run(generate_code([layout]))

    assert "State0.jsx" in files
    assert "styles.css" in files
    assert "index.jsx" in files
    assert "package.json" in files


def test_retries_once_then_succeeds(monkeypatch):
    layout = _make_layout("state_1")
    tree = _build_code_ready_tree(layout.root, layout.state_name)
    jsx, css = _build_faithful_response(tree)
    bad_response = json.dumps({"component_jsx": "<div></div>", "styles_css": ""})
    good_response = json.dumps({"component_jsx": jsx, "styles_css": css})
    fake_llm = _FakeLLM(responses=[bad_response, good_response])
    monkeypatch.setattr(module_d, "_call_local_llm_structured", fake_llm.call)

    files = asyncio.run(generate_code([layout], max_retries=2))

    assert "State1.jsx" in files
    assert len(fake_llm.calls) == 2


def test_raises_validation_error_after_exhausting_retries(monkeypatch):
    layout = _make_layout("state_2")
    bad_response = json.dumps({"component_jsx": "<div></div>", "styles_css": ""})
    fake_llm = _FakeLLM(responses=[bad_response, bad_response])
    monkeypatch.setattr(module_d, "_call_local_llm_structured", fake_llm.call)

    with pytest.raises(LocalLLMValidationError, match="failed validation"):
        asyncio.run(generate_code([layout], max_retries=2))


def test_connection_error_is_not_retried(monkeypatch):
    layout = _make_layout("state_3")
    fake_llm = _FakeLLM(raise_exc=LocalLLMConnectionError("no subprocess"))
    monkeypatch.setattr(module_d, "_call_local_llm_structured", fake_llm.call)

    with pytest.raises(LocalLLMConnectionError, match="no subprocess"):
        asyncio.run(generate_code([layout], max_retries=3))

    assert len(fake_llm.calls) == 1, "a broken worker won't come up between retries, so this shouldn't retry"


def test_3d_scene_generates_extra_file_and_r3f_dependency(monkeypatch):
    layout = _make_layout("state_4", is_3d=True)
    tree = _build_code_ready_tree(layout.root, layout.state_name)
    jsx, css = _build_faithful_response(tree)
    component_response = json.dumps({"component_jsx": jsx, "styles_css": css})
    scene_jsx = "".join(
        f'<mesh position={{{node["position_3d"]["position"]}}}></mesh>' for node in [tree] + tree["children"]
    )
    scene_response = json.dumps({"scene3d_jsx": scene_jsx})
    fake_llm = _FakeLLM(responses=[component_response, scene_response])
    monkeypatch.setattr(module_d, "_call_local_llm_structured", fake_llm.call)

    files = asyncio.run(generate_code([layout]))

    assert "Scene3D_State4.jsx" in files
    assert "@react-three/fiber" in json.loads(files["package.json"])["dependencies"]





def test_package_output_zips_all_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "outputs_dir", tmp_path)

    zip_path = asyncio.run(package_output("job123", {"a.txt": "hello", "b.txt": "world"}))

    assert zip_path.exists()
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        assert set(zf.namelist()) == {"a.txt", "b.txt"}
