"""
Pydantic contracts shared across all four pipeline modules.

    Module A produces KeyStateFrame.
    Module B consumes KeyStateFrame, produces DetectedElement.
    Module C consumes DetectedElement, produces LayoutState (a DOMNode tree).
    Module D consumes LayoutState, produces source files.

Keeping these as explicit, validated models (rather than raw dicts)
means each module can be implemented and unit-tested independently
against a fixed interface, and FastAPI gets free request/response
validation and OpenAPI docs for the Job model.
"""
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    PARSING_VIDEO = "parsing_video"
    DETECTING_ELEMENTS = "detecting_elements"
    SYNTHESIZING_DOM = "synthesizing_dom"
    GENERATING_CODE = "generating_code"
    PACKAGING = "packaging"
    COMPLETE = "complete"
    FAILED = "failed"


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float = Field(ge=0)
    height: float = Field(ge=0)
    z_index: Optional[float] = None  # populated by Depth-Anything-V2 in Module B


class DetectedElement(BaseModel):
    element_id: str
    element_type: str  # e.g. button, input, modal, navbar, text, image, container
    bbox: BoundingBox
    text_content: Optional[str] = None
    hex_colors: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class KeyStateFrame(BaseModel):
    """Output of Module A: one 'crucial' UI state extracted from the video."""

    frame_index: int
    timestamp_sec: float
    image_path: str
    ssim_delta_from_previous: Optional[float] = None
    cursor_position: Optional[tuple[float, float]] = None
    inferred_action: Literal["hover", "click", "drag", "none"] = "none"


class DOMNode(BaseModel):
    """One node in Module C's synthesized parent-child tree."""

    node_id: str
    tag_hint: str  # div, button, input, span, img, canvas (3D)...
    element_type: Optional[str] = None  # Module B's original label (e.g. "button", "text"); None for the synthesized root
    bbox: BoundingBox
    text_content: Optional[str] = None
    hex_colors: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    children: list["DOMNode"] = Field(default_factory=list)


DOMNode.model_rebuild()  # resolves the self-reference in `children`


class LayoutState(BaseModel):
    """The full layout_state.json contract handed to Module D."""

    state_name: str
    source_frame: KeyStateFrame
    root: DOMNode
    is_3d_scene: bool = False


class Job(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime
    updated_at: datetime
    source_video_path: str
    output_zip_path: Optional[str] = None
    error_message: Optional[str] = None
    current_stage_detail: Optional[str] = None
    key_states_detected: int = 0
    progress_percent: float = 0.0
    enable_3d: bool = False
    fast_mode: bool = False

