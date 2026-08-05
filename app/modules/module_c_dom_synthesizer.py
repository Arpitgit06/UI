"""
Module C: The "DOM" Synthesizer (CPU-bound)

Pure mathematical processing: takes Module B's FLAT list of
DetectedElement (bounding boxes, text, colors, per-element depth) and
reconstructs a parent-child DOMNode tree by spatial containment, then
persists it as `{state_name}_layout.json` -- the exact contract Module D
consumes.

Containment algorithm:

  - "A is contained in B" uses an overlap RATIO
    (intersection_area(A, B) / area(A) >= dom_containment_threshold),
    not strict corner containment. Real detections are noisy -- a
    button's box and the text inside it rarely align to the pixel --
    so requiring literal full containment would miss real parent-child
    relationships over a few pixels of detector imprecision.

  - Each element's PARENT is the smallest-area element that contains it
    (the tightest enclosing box), not just any containing box. A text
    label sitting inside a button which sits inside a card should
    attach to the button directly, not skip straight to the card.

  - A parent candidate must have STRICTLY greater area than the child.
    This one rule also makes the algorithm cycle-proof by construction:
    a cycle would require two elements each with area strictly greater
    than the other, which is impossible for real numbers. Elements with
    no valid containing element become direct children of a synthesized
    root/page node sized to the actual source image dimensions.

  - Near-duplicate boxes (similar element and near-identical size) are
    NOT nested into each other under this rule -- they end up as
    siblings. This is a deliberate simplicity choice: arbitrarily
    picking one of two near-identical boxes as the "parent" would be a
    coin flip, not a meaningful structural decision.

  - Within a parent, children are sorted by z_index ascending (farthest
    first, nearest last) to match typical HTML/CSS painting order,
    where later-in-DOM elements render on top absent explicit
    stacking rules.

Deliberately NOT attempted here: distinguishing a genuine 3D scene from
a flat 2D UI (LayoutState.is_3d_scene is left at its False default).
Module B's z_index values are already min-max normalized to 0-100 per
state, which stretches even a tiny real depth range to fill the full
scale -- so the normalized values alone can't tell "meaningfully
layered" apart from "flat, but normalization amplified noise." Making
this determination reliably would need Module B to also pass through
some pre-normalization statistic (e.g. the raw depth range or
variance); that's a small, targeted addition to make later rather than
a heuristic guess bolted on here.
"""
import asyncio
import json
from pathlib import Path
from typing import Optional

from PIL import Image

from app.config import settings
from app.models.schemas import BoundingBox, DetectedElement, DOMNode, KeyStateFrame, LayoutState
from app.utils.logger import get_logger

logger = get_logger("omniui.module_c")

# Best-effort element_type -> HTML tag mapping, covering common Rico/UI-dataset
# taxonomy classes plus Module B's own "text" (OCR) label. Falls back to "div"
# for anything unrecognized -- including COCO classes like "person"/"car" if
# this runs against Module B's current stock (non-UI-trained) YOLO weights.
_TAG_HINT_MAP = {
    "text": "span",
    "button": "button",
    "text_button": "button",
    "input": "input",
    "checkbox": "input",
    "radio_button": "input",
    "on/off_switch": "input",
    "switch": "input",
    "slider": "input",
    "icon": "i",
    "image": "img",
    "image_view": "img",
    "background_image": "div",
    "navbar": "nav",
    "toolbar": "nav",
    "button_bar": "nav",
    "list_item": "li",
    "list": "ul",
    "card": "div",
    "modal": "div",
    "drawer": "div",
    "web_view": "iframe",
    "map_view": "div",
    "video": "video",
    "advertisement": "div",
    "date_picker": "input",
    "number_stepper": "input",
    "pager_indicator": "div",
}
_DEFAULT_TAG_HINT = "div"


def _tag_hint_for(element_type: str) -> str:
    normalized = element_type.strip().lower().replace(" ", "_")
    return _TAG_HINT_MAP.get(normalized, _DEFAULT_TAG_HINT)


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------


def _area(bbox: BoundingBox) -> float:
    return max(0.0, bbox.width) * max(0.0, bbox.height)


def _intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _containment_ratio(inner: BoundingBox, outer: BoundingBox) -> float:
    """Fraction of `inner`'s area that overlaps `outer`. 1.0 = fully contained."""
    inner_area = _area(inner)
    if inner_area <= 0:
        return 0.0
    return _intersection_area(inner, outer) / inner_area


def _assign_parents(elements: list[DetectedElement], containment_threshold: float) -> dict[str, Optional[str]]:
    """
    For each element, finds its tightest (smallest-area) valid containing
    element among the others. Returns element_id -> parent element_id,
    or None if nothing qualifies (a direct child of the root/page node).
    O(n^2) pairwise checks -- fine for the tens-to-low-hundreds of
    elements a single UI screenshot produces.
    """
    areas = {el.element_id: _area(el.bbox) for el in elements}
    parent_of: dict[str, Optional[str]] = {}

    for el in elements:
        best_parent_id: Optional[str] = None
        best_parent_area = float("inf")
        for other in elements:
            if other.element_id == el.element_id:
                continue
            if areas[other.element_id] <= areas[el.element_id]:
                continue  # a parent must be strictly larger (also what keeps this cycle-proof)
            if _containment_ratio(el.bbox, other.bbox) < containment_threshold:
                continue
            if areas[other.element_id] < best_parent_area:
                best_parent_id = other.element_id
                best_parent_area = areas[other.element_id]
        parent_of[el.element_id] = best_parent_id

    return parent_of


def _root_bbox_from_image(image_path: str) -> Optional[BoundingBox]:
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return BoundingBox(x=0, y=0, width=float(width), height=float(height))
    except Exception:
        logger.warning(f"Could not read image dimensions from {image_path}; falling back to detection bounds.")
        return None


def _union_bbox(elements: list[DetectedElement]) -> BoundingBox:
    if not elements:
        return BoundingBox(x=0, y=0, width=0, height=0)
    min_x = min(el.bbox.x for el in elements)
    min_y = min(el.bbox.y for el in elements)
    max_x = max(el.bbox.x + el.bbox.width for el in elements)
    max_y = max(el.bbox.y + el.bbox.height for el in elements)
    return BoundingBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)


def _detected_element_to_dom_node(el: DetectedElement) -> DOMNode:
    return DOMNode(
        node_id=el.element_id,
        tag_hint=_tag_hint_for(el.element_type),
        element_type=el.element_type,
        bbox=el.bbox,
        text_content=el.text_content,
        hex_colors=el.hex_colors,
        confidence=el.confidence,
    )


def _build_tree(elements: list[DetectedElement], parent_of: dict[str, Optional[str]], root: DOMNode) -> None:
    nodes: dict[str, DOMNode] = {root.node_id: root}
    for el in elements:
        nodes[el.element_id] = _detected_element_to_dom_node(el)

    children_map: dict[str, list[DOMNode]] = {node_id: [] for node_id in nodes}
    for el in elements:
        parent_id = parent_of[el.element_id] or root.node_id
        children_map[parent_id].append(nodes[el.element_id])

    def _attach(node: DOMNode) -> None:
        kids = children_map.get(node.node_id, [])
        # paint order: farthest first, nearest last (higher z_index renders on top)
        kids.sort(key=lambda n: n.bbox.z_index if n.bbox.z_index is not None else 0.0)
        node.children = kids
        for kid in kids:
            _attach(kid)

    _attach(root)


def _deduplicate(elements: list[DetectedElement]) -> list[DetectedElement]:
    """
    Defensive: Module B's id scheme (kind + monotonic index) shouldn't
    ever produce collisions, but a duplicate element_id would silently
    corrupt the tree (two elements claiming the same node), so it's
    cheap to guard against.
    """
    seen: set[str] = set()
    unique: list[DetectedElement] = []
    for el in elements:
        if el.element_id in seen:
            logger.warning(f"Duplicate element_id {el.element_id!r} in detections; keeping first occurrence.")
            continue
        seen.add(el.element_id)
        unique.append(el)
    return unique


# ---------------------------------------------------------------------------
# Local Module Entry Points (Called exclusively by local pipeline.py)
# ---------------------------------------------------------------------------


def _detect_3d_scene(
    detections: list[DetectedElement],
    raw_depth_variance: Optional[float] = None,
) -> bool:
    """
    Detects whether a scene likely requires 3D Three.js / R3F reconstruction by checking:
    1. Element taxonomy labels from Vision LLM (canvas, 3d_view, mesh, webgl)
    2. Significant z-index spread across detected elements (>= 65 on normalized 0-100 scale)
    3. Raw pre-normalization depth variance (if available from Module B)
    """
    if not detections:
        return False
    
    # Check Vision LLM-assigned element types for 3D indicators
    for el in detections:
        if el.element_type:
            et = el.element_type.lower()
            if any(k in et for k in ("3d", "canvas", "webgl", "mesh", "model", "scene", "viewport", "orbit")):
                return True
    
    # Check raw depth variance (pre-normalization) — most reliable signal
    if raw_depth_variance is not None and raw_depth_variance > 500.0:
        return True
    
    # Check normalized z-index spread (less reliable due to normalization stretching)
    z_vals = [el.bbox.z_index for el in detections if el.bbox.z_index is not None]
    if len(z_vals) >= 3 and (max(z_vals) - min(z_vals)) >= 65.0:
        return True
    
    return False


def _synthesize_dom_sync(
    state: KeyStateFrame,
    detections: list[DetectedElement],
    containment_threshold: Optional[float] = None,
    is_3d_scene: Optional[bool] = None,
    vision_verifier_approved: bool = False,
    raw_depth_variance: Optional[float] = None,
) -> LayoutState:
    containment_threshold = (
        settings.dom_containment_threshold if containment_threshold is None else containment_threshold
    )
    detections = _deduplicate(detections)

    root_bbox = _root_bbox_from_image(state.image_path) or _union_bbox(detections)
    root = DOMNode(node_id="root", tag_hint="div", element_type="page", bbox=root_bbox)

    if detections:
        parent_of = _assign_parents(detections, containment_threshold)
        _build_tree(detections, parent_of, root)

    if is_3d_scene is None:
        is_3d_scene = _detect_3d_scene(detections, raw_depth_variance)

    logger.info(f"Module C: {state.image_path} -> {len(detections)} element(s) nested under root (is_3d={is_3d_scene})")

    return LayoutState(
        state_name=f"state_{state.frame_index}",
        source_frame=state,
        root=root,
        is_3d_scene=is_3d_scene,
        vision_verifier_approved=vision_verifier_approved,
        raw_depth_variance=raw_depth_variance,
    )


def write_layout_json(layout_state: LayoutState, output_dir: Path) -> Path:
    """Persists a LayoutState to `{state_name}_layout.json`. The one I/O touch point in this module."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{layout_state.state_name}_layout.json"
    json_path.write_text(layout_state.model_dump_json(indent=2))
    return json_path


async def synthesize_dom(
    state: KeyStateFrame,
    detections: list[DetectedElement],
    output_dir: Optional[Path] = None,
    containment_threshold: Optional[float] = None,
    is_3d_scene: Optional[bool] = None,
    vision_verifier_approved: bool = False,
    raw_depth_variance: Optional[float] = None,
) -> LayoutState:
    """
    Builds the tree in a worker thread (consistent with the rest of the
    pipeline's handling of blocking calls) and, if `output_dir` is given,
    persists it as `{state_name}_layout.json` in that directory.
    """

    def _run() -> LayoutState:
        layout_state = _synthesize_dom_sync(
            state, detections, containment_threshold, is_3d_scene,
            vision_verifier_approved, raw_depth_variance,
        )
        if output_dir is not None:
            write_layout_json(layout_state, output_dir)
        return layout_state

    return await asyncio.to_thread(_run)

