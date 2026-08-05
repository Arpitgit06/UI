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


def _fix_json_string_escaping(text: str) -> str:
    """Fix unescaped newlines/tabs/CR inside JSON string values."""
    result = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\' and in_string and i + 1 < len(text):
            result.append(c)
            result.append(text[i + 1])
            i += 2
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
            i += 1
            continue
        if in_string:
            if c == '\n':
                result.append('\\n')
            elif c == '\r':
                result.append('\\r')
            elif c == '\t':
                result.append('\\t')
            else:
                result.append(c)
        else:
            result.append(c)
        i += 1
    return ''.join(result)



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
    dev_dependencies = {"vite": "^6.0.0", "@vitejs/plugin-react": "^4.3.0"}
    if needs_r3f:
        dependencies["@react-three/fiber"] = "^8.17.0"
        dependencies["three"] = "^0.166.0"
    return json.dumps(
        {
            "name": "omniui-generated-ui",
            "private": True,
            "version": "0.1.0",
            "type": "module",
            "scripts": {
                "dev": "vite",
                "build": "vite build",
                "preview": "vite preview",
            },
            "dependencies": dependencies,
            "devDependencies": dev_dependencies,
        },
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


def _build_index_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>OmniUI Generated UI</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/main.jsx"></script>
  </body>
</html>
"""


def _build_main_jsx() -> str:
    return """import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
"""


def _build_vite_config() -> str:
    return """import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
});
"""


def _build_readme() -> str:
    return """# OmniUI Generated UI

This project was auto-generated by the OmniUI video-to-UI pipeline.

## Getting Started

```bash
npm install
npm run dev
```

Open the URL shown in the terminal (usually http://localhost:5173).

## Project Structure

- `App.jsx` — Gallery of all captured UI states
- `State*.jsx` — Individual state components (pixel-accurate layout)
- `styles.css` — All CSS rules (using CSS custom properties for shared palette)
- `transitions.json` — Cursor/action metadata per state transition
- `verification_report.json` — Detection quality report

## Notes

- Each state renders independently — no transition logic is inferred
- Layout uses `position: absolute` with percentage values for proportional scaling
- Colors are extracted from the original video frames via Colorgram
"""


def _build_transitions_json(layouts: list[LayoutState]) -> str:
    transitions = {}
    for layout in layouts:
        frame = layout.source_frame
        transitions[layout.state_name] = {
            "timestamp_sec": frame.timestamp_sec,
            "frame_index": frame.frame_index,
            "inferred_action": frame.inferred_action,
            "cursor_position": list(frame.cursor_position) if frame.cursor_position else None,
            "ssim_delta": frame.ssim_delta_from_previous,
            "ambient_motion_regions": [
                list(r) for r in frame.ambient_motion_regions
            ] if frame.ambient_motion_regions else [],
        }
    return json.dumps(transitions, indent=2)


def _extract_color_palette(
    css_blocks: list[str],
    delta_e_threshold: float = 10.0,
) -> tuple[dict[str, str], list[str]]:
    """
    Extract unique colors from all CSS blocks, cluster similar colors,
    and return:
      1. A mapping of original hex -> CSS variable name
      2. Updated CSS blocks with hex values replaced by var() references
    """
    import re as _re
    
    # Extract all hex colors from CSS
    hex_pattern = _re.compile(r'#[0-9a-fA-F]{6}')
    all_colors = set()
    for block in css_blocks:
        all_colors.update(hex_pattern.findall(block))
    
    if not all_colors:
        return {}, css_blocks
    
    # Simple color clustering: group colors by proximity (using channel diff)
    def _color_distance(c1: str, c2: str) -> float:
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        return ((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2) ** 0.5
    
    sorted_colors = sorted(all_colors)
    clusters: list[list[str]] = []
    used = set()
    
    for color in sorted_colors:
        if color in used:
            continue
        cluster = [color]
        used.add(color)
        for other in sorted_colors:
            if other not in used and _color_distance(color, other) < delta_e_threshold:
                cluster.append(other)
                used.add(other)
        clusters.append(cluster)
    
    # Build palette mapping: each cluster gets one CSS var
    color_to_var = {}
    for i, cluster in enumerate(clusters):
        var_name = f"--omni-{i}"
        for color in cluster:
            color_to_var[color] = var_name
    
    # Build :root declaration
    palette_lines = [":root {"]
    for i, cluster in enumerate(clusters):
        palette_lines.append(f"  --omni-{i}: {cluster[0]};")
    palette_lines.append("}")
    palette_css = "\n".join(palette_lines)
    
    # Replace hex colors in CSS blocks with var() references
    updated_blocks = []
    for block in css_blocks:
        updated = block
        for color, var_name in color_to_var.items():
            updated = updated.replace(color, f"var({var_name})")
        updated_blocks.append(updated)
    
    return color_to_var, [palette_css] + updated_blocks


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
# Fast Deterministic Python Code Generator
# (Bypasses LLM token generation latency and JSON formatting bugs)
# ---------------------------------------------------------------------------


def _render_tree_to_jsx_and_css(code_tree: dict, component_name: str) -> tuple[str, str]:
    css_rules = []

    def _render_node(node: dict, indent: int = 4) -> str:
        cls = node["class_name"]
        tag = node["tag"] or "div"
        text = node["text"] or ""

        rule_lines = [f".{cls} {{"]
        for k, v in node["css"].items():
            rule_lines.append(f"  {k}: {v};")
        rule_lines.append("}")
        css_rules.append("\n".join(rule_lines))

        ind = " " * indent
        children_jsx = []
        for child in node["children"]:
            children_jsx.append(_render_node(child, indent + 2))

        if text and children_jsx:
            content = f"\n{ind}  {text}" + "".join(children_jsx) + f"\n{ind}"
        elif text:
            content = text
        elif children_jsx:
            content = "".join(children_jsx) + f"\n{ind}"
        else:
            content = ""

        if content:
            return f"\n{ind}<{tag} className=\"{cls}\">{content}</{tag}>"
        else:
            return f"\n{ind}<{tag} className=\"{cls}\" />"

    root_jsx = _render_node(code_tree, indent=6)

    # Add transition comment if cursor/action data available
    transition_comment = ""
    
    component_jsx = (
        f"import React from 'react';\n\n"
        f"{transition_comment}"
        f"export default function {component_name}() {{\n"
        f"  return ({root_jsx}\n"
        f"  );\n"
        f"}}\n"
    )
    styles_css = "\n\n".join(css_rules)
    return component_jsx, styles_css


def _render_tree_to_scene3d_jsx(code_tree: dict, component_name: str) -> str:
    meshes = []
    
    def _walk(node: dict):
        if "position_3d" in node and node["position_3d"]:
            p = node["position_3d"]
            x, y, z = p["position"]
            w, h = p["width"], p["height"]
            color = node["css"].get("background-color", "#cccccc")
            if not color.startswith("#"):
                color = "#cccccc"
            meshes.append(
                f"        <mesh position={{[{x:.2f}, {y:.2f}, {z:.2f}]}}>\n"
                f"          <planeGeometry args={{[{w:.2f}, {h:.2f}]}} />\n"
                f"          <meshStandardMaterial color=\"{color}\" />\n"
                f"        </mesh>"
            )
        for child in node.get("children", []):
            _walk(child)
            
    _walk(code_tree)
    
    meshes_str = "\n".join(meshes)
    return (
        f"import React from 'react';\n"
        f"import {{ Canvas }} from '@react-three/fiber';\n\n"
        f"export default function Scene3D_{component_name}() {{\n"
        f"  return (\n"
        f"    <div style={{{{ width: '100vw', height: '100vh' }}}}>\n"
        f"      <Canvas camera={{{{ position: [0, 0, 1500] }}}}>\n"
        f"        <ambientLight intensity={{0.5}} />\n"
        f"        <directionalLight position={{[10, 10, 5]}} intensity={{1}} />\n"
        f"{meshes_str}\n"
        f"      </Canvas>\n"
        f"    </div>\n"
        f"  );\n"
        f"}}\n"
    )


# ---------------------------------------------------------------------------
# Local LLM calls
# ---------------------------------------------------------------------------


import subprocess
import threading
import sys

class PersistentLLMWorker:
    def __init__(self, max_jobs=4):
        self.process = None
        self.lock = threading.Lock()
        self.jobs_processed = 0
        self.max_jobs = max_jobs
        self.current_model = None
        
    def start(self, model: str, load_in_4bit: bool):
        if self.process is None or self.process.poll() is not None or self.current_model != model:
            if self.process:
                self.shutdown()
                
            self.process = subprocess.Popen(
                [sys.executable, "-m", "app.modules.llm_subprocess_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace"
            )
            self.jobs_processed = 0
            self.current_model = model
            
            init_payload = {
                "model_name": model,
                "load_in_4bit": load_in_4bit
            }
            self.process.stdin.write(json.dumps(init_payload) + "\n")
            self.process.stdin.flush()
            
    def analyze(self, model: str, system_prompt: str, user_content: str, temperature: float, load_in_4bit: bool) -> dict:
        with self.lock:
            self.start(model, load_in_4bit)
            
            payload = {
                "system_prompt": system_prompt,
                "user_content": user_content,
                "temperature": temperature,
                "max_new_tokens": settings.local_llm_max_new_tokens
            }
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
            
            output = ""
            while True:
                line = self.process.stdout.readline()
                if not line:
                    self.process = None
                    raise LocalLLMConnectionError("Local LLM worker subprocess died unexpectedly.")
                
                if "__LLM_JSON_START__" not in line and "__LLM_JSON_END__" not in line and line.strip():
                    logger.info(f"[LLM Worker STDOUT] {line.strip()}")
                    
                output += line
                if "__LLM_JSON_END__" in output:
                    break
                    
            if "__LLM_JSON_START__" not in output or "__LLM_JSON_END__" not in output:
                raise LocalLLMGenerationError("Local LLM response did not contain expected JSON markers.")
                
            json_str = output.split("__LLM_JSON_START__")[1].split("__LLM_JSON_END__")[0].strip()
            
            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, AttributeError):
                pass

            fixed = _fix_json_string_escaping(json_str)
            try:
                return json.loads(fixed)
            except (json.JSONDecodeError, AttributeError):
                pass
                
            first_brace = fixed.find('{')
            if first_brace != -1:
                depth = 0
                in_string = False
                escape_next = False
                for i in range(first_brace, len(fixed)):
                    c = fixed[i]
                    if escape_next:
                        escape_next = False
                        continue
                    if c == '\\':
                        escape_next = True
                        continue
                    if c == '"':
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = fixed[first_brace:i + 1]
                            try:
                                return json.loads(candidate)
                            except (json.JSONDecodeError, ValueError):
                                break

            raise LocalLLMGenerationError("Local LLM response wasn't valid JSON.")

    def shutdown_if_needed(self):
        with self.lock:
            self.jobs_processed += 1
            if self.jobs_processed >= self.max_jobs:
                self.shutdown()
                
    def shutdown(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
            self.process = None

_global_llm_worker = PersistentLLMWorker()

async def _call_local_llm_structured(
    model: str, system_prompt: str, user_content: str, schema: dict, temperature: float, load_in_4bit: bool
) -> dict:
    def _run():
        return _global_llm_worker.analyze(model, system_prompt, user_content, temperature, load_in_4bit)
    return await asyncio.to_thread(_run)


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
    fast_mode: bool = False,
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

    if fast_mode:
        logger.info("Module D executing in Fast Mode (Skipping LLM)")
    else:
        logger.info(f"Module D executing local code generation with model: {code_model} (4-bit={load_in_4bit})")

    files: dict[str, str] = {}
    css_blocks: list[str] = []
    any_3d = False

    for layout in layouts:
        code_tree = _build_code_ready_tree(layout.root, layout.state_name)
        component_name = _pascal_case(layout.state_name)

        # Build transition comment for this state
        frame = layout.source_frame
        transition_comment = ""
        if frame.cursor_position:
            cx, cy = frame.cursor_position
            transition_comment = (
                f"{{/* Transition: {frame.inferred_action} near ({cx:.0f}, {cy:.0f}) "
                f"at t={frame.timestamp_sec:.1f}s"
            )
            if frame.ssim_delta_from_previous is not None:
                transition_comment += f", SSIM delta={frame.ssim_delta_from_previous:.4f}"
            transition_comment += " */}\n"

        if fast_mode:
            component_jsx, styles_css = _render_tree_to_jsx_and_css(code_tree, component_name)
        else:
            try:
                component_jsx, styles_css = await _generate_component_and_styles(
                    code_model, component_name, code_tree, temperature, max_retries, strict_validation, load_in_4bit
                )
                if component_jsx.lstrip().startswith('{') or "import React" not in component_jsx:
                    logger.warning(f"{component_name}: LLM output was raw JSON string or invalid; falling back to deterministic Python renderer.")
                    component_jsx, styles_css = _render_tree_to_jsx_and_css(code_tree, component_name)
            except Exception as e:
                logger.warning(f"{component_name}: LLM generation failed ({e}); falling back to deterministic Python renderer.")
                component_jsx, styles_css = _render_tree_to_jsx_and_css(code_tree, component_name)

        # Inject transition comment into component
        if transition_comment and "export default function" in component_jsx:
            component_jsx = component_jsx.replace(
                "export default function",
                f"{transition_comment}export default function",
            )

        files[f"{component_name}.jsx"] = component_jsx
        css_blocks.append(styles_css)

        if layout.is_3d_scene:
            any_3d = True
            if fast_mode:
                scene3d_jsx = _render_tree_to_scene3d_jsx(code_tree, component_name)
            else:
                try:
                    scene3d_jsx = await _generate_scene3d(
                        code_model, component_name, code_tree, temperature, max_retries, strict_validation, load_in_4bit
                    )
                    if scene3d_jsx.lstrip().startswith('{') or "import React" not in scene3d_jsx:
                        logger.warning(f"{component_name}: 3D LLM output was raw JSON string or invalid; falling back to deterministic Python renderer.")
                        scene3d_jsx = _render_tree_to_scene3d_jsx(code_tree, component_name)
                except Exception as e:
                    logger.warning(f"{component_name}: 3D LLM generation failed ({e}); falling back to deterministic Python renderer.")
                    scene3d_jsx = _render_tree_to_scene3d_jsx(code_tree, component_name)
            files[f"Scene3D_{component_name}.jsx"] = scene3d_jsx

        logger.info(f"Module D: generated {component_name} ({len(code_tree['children'])} top-level child node(s))")

    _global_llm_worker.shutdown_if_needed()

    # Build shared color palette from all CSS blocks
    color_to_var, updated_css_blocks = _extract_color_palette(css_blocks)
    if color_to_var:
        logger.info(f"Module D: extracted shared palette with {len(set(color_to_var.values()))} color variables")

    files["styles.css"] = "\n\n".join(updated_css_blocks)
    files["App.jsx"] = _build_index_scaffold(layouts)
    files["package.json"] = _build_package_json(any_3d)
    files["index.html"] = _build_index_html()
    files["main.jsx"] = _build_main_jsx()
    files["vite.config.js"] = _build_vite_config()
    files["README.md"] = _build_readme()
    files["transitions.json"] = _build_transitions_json(layouts)

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

