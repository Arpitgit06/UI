"""Build semantically valid containment trees with measured presentation attributes. Scene depth is not CSS stacking order."""
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
    "link": "a",
    "radio": "input",
    "canvas": "div",
    "header": "header",
    "footer": "footer",
    "sidebar": "aside",
    "container": "div",
    "dropdown": "select",
    "tab": "button",
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
            if _tag_hint_for(other.element_type) not in {"div","nav","aside","header","footer","ul","li","button"}:
                continue
            if _tag_hint_for(other.element_type) == "button" and _tag_hint_for(el.element_type) in {"button","input","select","a"}:
                continue
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
        styles=el.styles,
        asset_path=el.asset_path,
        attributes={**({"type": {"checkbox":"checkbox", "radio":"radio", "radio_button":"radio",
                    "slider":"range", "date_picker":"date", "switch":"checkbox",
                    "on/off_switch":"checkbox", "number_stepper":"number"}.get(el.element_type,"text")}
                    if _tag_hint_for(el.element_type) == "input" else {}), **el.attributes},
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
        duplicate = False
        for previous in unique:
            intersection = _intersection_area(el.bbox, previous.bbox)
            union = _area(el.bbox)+_area(previous.bbox)-intersection
            if union and intersection/union > .85 and el.element_type == previous.element_type and el.text_content == previous.text_content:
                duplicate = True
                break
        if not duplicate:
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
    # Depth range, normalized z-index and generic canvas labels are not evidence
    # of editable geometry. Explicit semantics are an optional scene hint only.
    return any(el.element_type.lower() in {"3d_view", "webgl", "mesh", "3d_scene"}
               for el in detections)


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
    try:
        from app.modules.visual_styles import measure_styles
        with Image.open(state.image_path) as image:
            root.styles = measure_styles(image, (0,0,image.width,image.height),None,"page")
            from app.modules.visual_styles import control_rectangles
            rectangles=control_rectangles(image)
            refreshed=[]
            for element in detections:
                box=element.bbox
                if element.element_type in {'button','input','dropdown','link','tab'}:
                    choices=[]
                    for x,y,w,h in rectangles:
                        intersection=max(0,min(x+w,box.x+box.width)-max(x,box.x))*max(0,min(y+h,box.y+box.height)-max(y,box.y))
                        if intersection/max(box.width*box.height,1)>.65 and w>=box.width*.9 and 15<h<180 and w*h<image.width*image.height*.15:
                            choices.append((w*h,BoundingBox(x=x,y=y,width=w,height=h)))
                    if choices:box=min(choices,key=lambda item:item[0])[1]
                presentation=measure_styles(image,(box.x,box.y,box.width,box.height),element.text_content,element.element_type)
                if element.text_content and element.attributes.get('data-text-bbox'):
                    try:
                        text_box=json.loads(element.attributes['data-text-bbox'])
                        measured_text=measure_styles(image,text_box,element.text_content,'text')
                        presentation.update({k:v for k,v in measured_text.items() if k in {'font-size','color','font-family'}})
                    except (ValueError,TypeError):pass
                refreshed.append(element.model_copy(update={'bbox':box,'styles':presentation}))
            detections=refreshed
            from app.modules.visual_groups import augment_visual_groups
            detections=augment_visual_groups(image,detections)
    except OSError:
        pass

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
    json_path.write_text(layout_state.model_dump_json(indent=2), encoding="utf-8")
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

