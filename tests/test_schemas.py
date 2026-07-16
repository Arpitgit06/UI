import pytest
from pydantic import ValidationError

from app.models.schemas import BoundingBox, DOMNode


def test_bounding_box_rejects_negative_dimensions():
    with pytest.raises(ValidationError):
        BoundingBox(x=0, y=0, width=-5, height=10)


def test_dom_node_nests_children_and_round_trips_json():
    child = DOMNode(
        node_id="child",
        tag_hint="button",
        bbox=BoundingBox(x=10, y=10, width=50, height=20),
    )
    root = DOMNode(
        node_id="root",
        tag_hint="div",
        bbox=BoundingBox(x=0, y=0, width=100, height=100),
        children=[child],
    )

    assert root.children[0].node_id == "child"
    # This is the exact payload shape Module D will consume as layout_state.json.
    rebuilt = DOMNode.model_validate_json(root.model_dump_json())
    assert rebuilt.children[0].tag_hint == "button"
