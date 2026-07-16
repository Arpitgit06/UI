"""
Module D: Local Code Generation Engine

Loads Module C's LayoutState tree(s) and emits React components, CSS,
and (for states flagged as 3D) a React Three Fiber scene, using a local
LLM (Qwen2.5-Coder-7B-Instruct) strictly as a syntax compiler.

Why "syntax compiler" is taken literally here, not just as prompt
flavor text: the architecture doc's own Stage 3 already separates
"convert absolute coordinates into layout logic" (a Python step) from
"generate the component" (the LLM step) -- and structured-output
research is explicit that grammar-constrained JSON decoding guarantees
syntactic validity, NOT value accuracy (a schema-constrained model can
still emit a syntactically perfect but numerically wrong percentage or
color). So every number and color the model would otherwise have to
"convert" or "guess" is computed in Python BEFORE the prompt is built:

  - Position/size: each node's bbox is expressed as a percentage of its
    DIRECT PARENT's box (not the root), so `position: absolute` +
    percentage left/top/width/height reproduces the captured layout
    exactly and scales proportionally if the root container resizes.
    This deliberately does not attempt Flexbox/Grid inference -- guessing
    which elements "belong" in a flex row is exactly the kind of
    interpretation this project's CV-first philosophy exists to avoid.
  - 3D projection: each node's 2D bbox + z_index is projected into R3F
    scene units (see _project_to_3d) once, in Python.
  - Color: the node's own hex_colors (from Module B's Colorgram pass).

The LLM's job is to transcribe this already-resolved tree into
syntactically correct JSX/CSS/R3F -- matching the doc's framing that the
LLM should "format the extracted logic," not invent it. Because
grammar-constrained decoding doesn't guarantee value fidelity, every
generation is mechanically re-checked afterward (_validate_component /
_validate_scene3d): does every node's class name appear in both outputs,
does its CSS block contain every property value exactly as computed,
does its text content appear verbatim? A failed check retries (in case
of a one-off slip) and ultimately raises a clear, itemized error rather
than silently shipping code that doesn't match the input.

Verification status: every deterministic function (percentage/3D-
coordinate math, class name sanitization, the CSS-block extractor, and
the validator itself against both a "faithful" and a deliberately
corrupted fake response) plus the full retry/validation orchestration
were verified -- see tests/test_module_d.py.
"""
import asyncio
import json
import re
import zipfile
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models.schemas import BoundingBox, DOMNode, LayoutState
from app.utils.logger import get_logger

logger = get_logger("omniui.module_d")

class LocalLLMConnectionError(RuntimeError):
    """Raised when the local LLM worker subprocess cannot execute or load weights."""


class LocalLLMGenerationError(RuntimeError):
    """Raised when local LLM response can't be parsed as the requested structured JSON."""


class LocalLLMValidationError(RuntimeError):
    """Raised when generated code fails the mechanical no-hallucination check after all retries."""




# Scene-unit constants for the 2D-pixel -> R3F-scene projection. Arbitrary
# but consistent choices; a captured UI fills roughly a 10x6 unit area
# centered at the origin, with depth spanning 0-2 units.
_SCENE_WIDTH = 10.0
_SCENE_HEIGHT = 6.0
_MAX_DEPTH = 2.0

_COMPONENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "component_jsx": {"type": "string"},
        "styles_css": {"type": "string"},
    },
    "required": ["component_jsx", "styles_css"],
}

_SCENE3D_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"scene3d_jsx": {"type": "string"}},
    "required": ["scene3d_jsx"],
}

_SYSTEM_PROMPT_2D = """\
You are a strict, deterministic syntax compiler, not a UI designer.

You will receive a JSON tree where every element's position, size, color, \
and text has ALREADY been computed. Your ONLY job is to transcribe it into \
syntactically correct React JSX and CSS.

Rules you must follow exactly:
- Do not invent, omit, rename, or reorder any element in the tree.
- Do not change any numeric value (percentages, z-index) or hex color from \
what is given -- copy each node's "css" values exactly into its CSS rule.
- Do not add placeholder or lorem-ipsum text. Only use the "text" field's \
exact value, and only emit a text child when it is not null.
- Do not add elements, wrappers, or styling not implied by the tree.
- Every node's "class_name" must appear as both a CSS selector \
(".class_name { ... }") in styles_css and a className reference in \
component_jsx.

Respond with a JSON object with exactly two fields: "component_jsx" (a \
complete, self-contained functional React component) and "styles_css" \
(one CSS rule block per node, using each node's exact "css" values).\
"""

_SYSTEM_PROMPT_3D = """\
You are a strict, deterministic syntax compiler generating a React Three \
Fiber (@react-three/fiber) scene, not a UI designer.

You will receive the same JSON tree; every node also has a "position_3d" \
field: {"position": [x, y, z], "width": w, "height": h}, already computed \
from the 2D layout and its depth information. Your ONLY job is to \
transcribe this into a syntactically correct R3F scene.

Rules you must follow exactly:
- One <mesh> per node, with a <planeGeometry args={[width, height]} /> \
sized from that node's position_3d.width/height.
- Set each mesh's position prop to that node's position_3d.position \
exactly -- do not recompute, guess, or adjust these coordinates.
- Use each node's first hex color (if present) as its material color \
(<meshStandardMaterial color="..." />); use <meshStandardMaterial /> with \
no color prop if the node has none.
- Wrap everything in a <Canvas> from '@react-three/fiber', with one \
<ambientLight /> and one <directionalLight position={[5, 5, 5]} />.
- Do not invent additional meshes, geometry, lights, or elements not \
present in the tree.

Respond with a JSON object with exactly one field: "scene3d_jsx" (a \
complete, self-contained functional React component default-exporting \
the scene).\
"""


# ---------------------------------------------------------------------------
# Pure deterministic pre-computation -- no LLM involved in any of this
# ---------------------------------------------------------------------------


def _sanitize_class_name(raw: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", raw).strip("-").lower()
    if not sanitized:
        return "el"
    if sanitized[0].isdigit():
        sanitized = f"el-{sanitized}"
    return sanitized


def _pascal_case(raw: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", raw)
    name = "".join(p.capitalize() for p in parts if p)
    return name or "State"


def _project_to_3d(bbox: BoundingBox, root_bbox: BoundingBox) -> dict:
    """
    Projects a 2D pixel bbox (plus its already-normalized 0-100 z_index)
    into R3F scene units: centered on the origin, sized to _SCENE_WIDTH x
    _SCENE_HEIGHT, depth spanning 0-_MAX_DEPTH. Screen Y is flipped since
    it grows downward while 3D Y conventionally grows upward.
    """
    root_w = root_bbox.width or 1.0
    root_h = root_bbox.height or 1.0
    cx = bbox.x + bbox.width / 2
    cy = bbox.y + bbox.height / 2

    x3d = (cx - root_w / 2) / root_w * _SCENE_WIDTH
    y3d = -(cy - root_h / 2) / root_h * _SCENE_HEIGHT
    z3d = ((bbox.z_index or 0.0) / 100.0) * _MAX_DEPTH
    width3d = bbox.width / root_w * _SCENE_WIDTH
    height3d = bbox.height / root_h * _SCENE_HEIGHT

    return {
        "position": [round(x3d, 3), round(y3d, 3), round(z3d, 3)],
        "width": round(width3d, 3),
        "height": round(height3d, 3),
    }


def _node_css(node: DOMNode, parent_bbox: Optional[BoundingBox]) -> dict[str, str]:
    css: dict[str, str] = {}
    if parent_bbox is None:
        # root: fixed intrinsic size that scales proportionally via max-width
        css["position"] = "relative"
        css["width"] = f"{node.bbox.width:.0f}px"
        css["max-width"] = "100%"
        if node.bbox.height:
            css["aspect-ratio"] = f"{node.bbox.width:.0f} / {node.bbox.height:.0f}"
    else:
        parent_w = parent_bbox.width or 1.0
        parent_h = parent_bbox.height or 1.0
        css["position"] = "absolute"
        css["left"] = f"{(node.bbox.x - parent_bbox.x) / parent_w * 100:.2f}%"
        css["top"] = f"{(node.bbox.y - parent_bbox.y) / parent_h * 100:.2f}%"
        css["width"] = f"{node.bbox.width / parent_w * 100:.2f}%"
        css["height"] = f"{node.bbox.height / parent_h * 100:.2f}%"

    if node.bbox.z_index is not None:
        css["z-index"] = str(int(round(node.bbox.z_index)))
    if node.hex_colors:
        css["color" if node.text_content else "background-color"] = node.hex_colors[0]

    return css


def _build_code_ready_tree(root: DOMNode, state_prefix: str) -> dict:
    """
    Walks the DOMNode tree once, computing every value the LLM will need
    and nothing it will have to infer: a sanitized class name (prefixed
    with state_prefix since Module B's element ids are only unique
    within one state, not across a whole job), resolved CSS properties,
    and a 3D projection for every node regardless of whether this
    particular state ends up using it.
    """

    def _walk(node: DOMNode, parent_bbox: Optional[BoundingBox]) -> dict:
        class_name = _sanitize_class_name(f"{state_prefix}-{node.node_id}")
        return {
            "node_id": node.node_id,
            "class_name": class_name,
            "tag": node.tag_hint,
            "text": node.text_content,
            "css": _node_css(node, parent_bbox),
            "position_3d": _project_to_3d(node.bbox, root.bbox),
            "children": [_walk(child, node.bbox) for child in node.children],
        }

    return _walk(root, None)


def _collect_class_names(code_tree: dict) -> set[str]:
    names = {code_tree["class_name"]}
    for child in code_tree["children"]:
        names |= _collect_class_names(child)
    return names


def _build_package_json(needs_r3f: bool) -> str:
    dependencies = {"react": "^18.3.0", "react-dom": "^18.3.0"}
    if needs_r3f:
        dependencies["@react-three/fiber"] = "^8.17.0"
        dependencies["three"] = "^0.166.0"
    return json.dumps(
        {"name": "omniui-generated-ui", "private": True, "version": "0.1.0", "type": "module", "dependencies": dependencies},
        indent=2,
    )


def _build_index_scaffold(layouts: list[LayoutState]) -> str:
    component_names = [_pascal_case(layout.state_name) for layout in layouts]
    imports = "\n".join(f"import {name} from './{name}';" for name in component_names)
    renders = "\n".join(f"      <{name} />" for name in component_names)
    return (
        f"{imports}\n"
        f"import './styles.css';\n\n"
        f"// Auto-generated gallery of every captured UI state. This does NOT attempt\n"
        f"// to reconstruct the interactions between states (e.g. which state is a\n"
        f"// hover/click result of which) -- that would require inferring transition\n"
        f"// logic this pipeline doesn't attempt. Each state renders as its own\n"
        f"// independent, pixel-accurate snapshot; wire up real navigation yourself.\n"
        f"export default function App() {{\n"
        f"  return (\n"
        f"    <div className=\"omniui-state-gallery\">\n"
        f"{renders}\n"
        f"    </div>\n"
        f"  );\n"
        f"}}\n"
    )


# ---------------------------------------------------------------------------
# Mechanical post-generation validation -- the "trust but verify" layer.
# Grammar-constrained JSON decoding guarantees syntactic validity, not
# value fidelity, so every generation is checked against the exact values
# that were computed above before it's accepted.
# ---------------------------------------------------------------------------


def _extract_css_block(css_text: str, class_name: str) -> Optional[str]:
    match = re.search(re.escape(f".{class_name}") + r"\s*\{([^}]*)\}", css_text)
    return match.group(1) if match else None


def _validate_component(code_tree: dict, component_jsx: str, styles_css: str) -> list[str]:
    problems: list[str] = []

    def _walk(node: dict) -> None:
        cls = node["class_name"]
        if cls not in component_jsx:
            problems.append(f"class '{cls}' not referenced in component_jsx")
        block = _extract_css_block(styles_css, cls)
        if block is None:
            problems.append(f"no CSS rule found for '.{cls}' in styles_css")
        else:
            for prop, value in node["css"].items():
                # Tied together deliberately: checking `value` alone would miss a
                # corrupted property whose WRONG value happens to match some OTHER
                # property's correct value in the same block (e.g. left and top
                # coincidentally both being "10.00%") -- pairing them closes that gap.
                if not re.search(re.escape(prop) + r"\s*:\s*" + re.escape(value), block):
                    problems.append(f"'.{cls}' CSS block missing expected {prop}: {value}")
        if node["text"] and node["text"] not in component_jsx:
            problems.append(f"expected text {node['text']!r} not found in component_jsx")
        for child in node["children"]:
            _walk(child)

    _walk(code_tree)
    return problems


def _validate_scene3d(code_tree: dict, scene3d_jsx: str) -> list[str]:
    problems: list[str] = []

    def _walk(node: dict) -> None:
        pos = node["position_3d"]["position"]
        pos_str = f"[{pos[0]}, {pos[1]}, {pos[2]}]"
        # tolerate either bracket spacing style a model might emit
        pos_compact = pos_str.replace(", ", ",")
        if pos_str not in scene3d_jsx and pos_compact not in scene3d_jsx:
            problems.append(f"node '{node['node_id']}' position {pos_str} not found in scene3d_jsx")
        for child in node["children"]:
            _walk(child)

    _walk(code_tree)
    return problems


# ---------------------------------------------------------------------------
# Local LLM calls
# ---------------------------------------------------------------------------


async def _call_local_llm_structured(
    model: str, system_prompt: str, user_content: str, schema: dict, temperature: float, load_in_4bit: bool
) -> dict:
    import subprocess
    import sys
    import uuid

    payload_file = settings.jobs_dir / f"llm_payload_{uuid.uuid4().hex}.json"
    payload_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(payload_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": model,
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "temperature": temperature,
                    "max_new_tokens": settings.local_llm_max_new_tokens,
                    "load_in_4bit": load_in_4bit,
                },
                f,
            )

        def _run_subprocess():
            import subprocess
            return subprocess.run(
                [sys.executable, "-m", "app.modules.llm_subprocess_worker", str(payload_file)],
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

        completed_process = await asyncio.to_thread(_run_subprocess)
        
        stdout_lines = completed_process.stdout.splitlines(keepends=True)
        
        for line in stdout_lines:
            if "__LLM_JSON_START__" not in line and "__LLM_JSON_END__" not in line and line.strip():
                logger.info(f"[LLM Worker STDOUT] {line.strip()}")
                
        process_returncode = completed_process.returncode
        stdout = completed_process.stdout
        stderr = ""
    finally:
        if payload_file.exists():
            try:
                payload_file.unlink()
            except Exception:
                pass

    if process_returncode != 0:
        raise LocalLLMConnectionError(
            f"Local LLM worker subprocess failed (model={model!r}). "
            f"Underlying error/stderr: {stderr or stdout}"
        )

    if "__LLM_JSON_START__" not in stdout or "__LLM_JSON_END__" not in stdout:
        raise LocalLLMGenerationError(
            f"Local LLM response did not contain expected JSON markers. Raw stdout: {stdout[:500]}"
        )

    json_str = stdout.split("__LLM_JSON_START__")[1].split("__LLM_JSON_END__")[0].strip()
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError) as exc:
        raise LocalLLMGenerationError(f"Local LLM response wasn't valid JSON: {exc}. Raw: {json_str[:200]}") from exc


async def _generate_component_and_styles(
    model: str,
    component_name: str,
    code_tree: dict,
    temperature: float,
    max_retries: int,
    strict_validation: bool,
    load_in_4bit: bool,
) -> tuple[str, str]:
    user_content = json.dumps({"component_name": component_name, "tree": code_tree}, indent=2)

    last_problems: list[str] = []
    for attempt in range(1, max_retries + 1):
        result = await _call_local_llm_structured(
            model, _SYSTEM_PROMPT_2D, user_content, _COMPONENT_RESPONSE_SCHEMA, temperature, load_in_4bit
        )
        try:
            component_jsx, styles_css = result["component_jsx"], result["styles_css"]
        except KeyError as exc:
            last_problems = [f"response missing expected key: {exc}"]
            logger.warning(f"{component_name}: attempt {attempt}/{max_retries} missing key {exc}")
            continue

        problems = _validate_component(code_tree, component_jsx, styles_css)
        if not problems:
            return component_jsx, styles_css
        if not strict_validation:
            logger.warning(f"{component_name}: proceeding despite validation issues (strict_validation=False): {problems}")
            return component_jsx, styles_css
        last_problems = problems
        logger.warning(f"{component_name}: validation failed on attempt {attempt}/{max_retries}: {problems}")

    raise LocalLLMValidationError(
        f"Generated code for {component_name!r} failed validation after {max_retries} attempt(s): {last_problems}"
    )


async def _generate_scene3d(
    model: str,
    component_name: str,
    code_tree: dict,
    temperature: float,
    max_retries: int,
    strict_validation: bool,
    load_in_4bit: bool,
) -> str:
    user_content = json.dumps({"component_name": component_name, "tree": code_tree}, indent=2)

    last_problems: list[str] = []
    for attempt in range(1, max_retries + 1):
        result = await _call_local_llm_structured(
            model, _SYSTEM_PROMPT_3D, user_content, _SCENE3D_RESPONSE_SCHEMA, temperature, load_in_4bit
        )
        try:
            scene3d_jsx = result["scene3d_jsx"]
        except KeyError as exc:
            last_problems = [f"response missing expected key: {exc}"]
            logger.warning(f"{component_name} (3D): attempt {attempt}/{max_retries} missing key {exc}")
            continue

        problems = _validate_scene3d(code_tree, scene3d_jsx)
        if not problems:
            return scene3d_jsx
        if not strict_validation:
            logger.warning(f"{component_name} (3D): proceeding despite validation issues: {problems}")
            return scene3d_jsx
        last_problems = problems
        logger.warning(f"{component_name} (3D): validation failed on attempt {attempt}/{max_retries}: {problems}")

    raise LocalLLMValidationError(
        f"Generated 3D scene for {component_name!r} failed validation after {max_retries} attempt(s): {last_problems}"
    )


# ---------------------------------------------------------------------------
# Local Module Entry Points (Called exclusively by local pipeline.py)
# ---------------------------------------------------------------------------





async def generate_code(
    layouts: list[LayoutState],
    code_model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_retries: Optional[int] = None,
    strict_validation: Optional[bool] = None,
) -> dict[str, str]:
    code_model = code_model or settings.local_llm_model_name
    temperature = settings.local_llm_temperature if temperature is None else temperature
    max_retries = max_retries or settings.local_llm_max_retries
    strict_validation = settings.local_llm_strict_validation if strict_validation is None else strict_validation
    load_in_4bit = settings.local_llm_load_in_4bit

    logger.info(f"Module D executing local code generation with model: {code_model} (4-bit={load_in_4bit})")

    files: dict[str, str] = {}
    css_blocks: list[str] = []
    any_3d = False

    for layout in layouts:
        code_tree = _build_code_ready_tree(layout.root, layout.state_name)
        component_name = _pascal_case(layout.state_name)

        component_jsx, styles_css = await _generate_component_and_styles(
            code_model, component_name, code_tree, temperature, max_retries, strict_validation, load_in_4bit
        )
        files[f"{component_name}.jsx"] = component_jsx
        css_blocks.append(styles_css)

        if layout.is_3d_scene:
            any_3d = True
            scene3d_jsx = await _generate_scene3d(
                code_model, component_name, code_tree, temperature, max_retries, strict_validation, load_in_4bit
            )
            files[f"Scene3D_{component_name}.jsx"] = scene3d_jsx

        logger.info(f"Module D: generated {component_name} ({len(code_tree['children'])} top-level child node(s))")

    files["styles.css"] = "\n\n".join(css_blocks)
    files["index.jsx"] = _build_index_scaffold(layouts)
    files["package.json"] = _build_package_json(any_3d)
    return files


def _package_output_sync(job_id: str, files: dict[str, str]) -> Path:
    zip_path = settings.outputs_dir / f"{job_id}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return zip_path


async def package_output(job_id: str, files: dict[str, str]) -> Path:
    return await asyncio.to_thread(_package_output_sync, job_id, files)
