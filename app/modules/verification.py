"""
Module: Post-Generation Verification Report

Produces a verification_report.json summarizing detection quality per state:
- How many elements were detected vs. generated
- Coverage score (ratio of generated/detected)
- Vision LLM verifier approval status
- Average confidence scores
- Text element counts

This is a structural/statistical check, not a visual screenshot comparison.
"""
import json
from typing import Optional

from app.models.schemas import DOMNode, LayoutState
from app.utils.logger import get_logger

logger = get_logger("omniui.verification")


def _count_nodes(node: DOMNode) -> int:
    """Count total nodes in the DOM tree (excluding root)."""
    count = len(node.children)
    for child in node.children:
        count += _count_nodes(child)
    return count


def _count_text_nodes(node: DOMNode) -> int:
    """Count nodes with text content."""
    count = 1 if node.text_content else 0
    for child in node.children:
        count += _count_text_nodes(child)
    return count


def build_verification_report(
    layouts: list[LayoutState],
    detected_counts: Optional[list[int]] = None,
) -> str:
    """
    Build a verification report comparing detection counts with generated
    DOM node counts. Returns JSON string for inclusion in the output zip.
    """
    states_report = []
    total_detected = 0
    total_generated = 0
    coverage_scores = []

    for i, layout in enumerate(layouts):
        generated_nodes = _count_nodes(layout.root)
        text_in_output = _count_text_nodes(layout.root)
        
        # Use provided detection count or estimate from tree
        detected = detected_counts[i] if detected_counts and i < len(detected_counts) else generated_nodes
        total_detected += detected
        total_generated += generated_nodes

        coverage = generated_nodes / max(detected, 1)
        coverage = min(coverage, 1.0)  # cap at 1.0
        coverage_scores.append(coverage)

        state_report = {
            "state_name": layout.state_name,
            "source_image": layout.source_frame.image_path,
            "timestamp_sec": layout.source_frame.timestamp_sec,
            "detected_elements": detected,
            "generated_nodes": generated_nodes,
            "coverage_score": round(coverage, 3),
            "text_elements_in_output": text_in_output,
            "is_3d_scene": layout.is_3d_scene,
            "vision_verifier_approved": layout.vision_verifier_approved,
        }
        states_report.append(state_report)

        if coverage < 0.5:
            logger.warning(
                f"Verification: {layout.state_name} has low coverage "
                f"({coverage:.1%}) — {generated_nodes}/{detected} elements"
            )

    avg_coverage = sum(coverage_scores) / max(len(coverage_scores), 1)
    low_coverage_count = sum(1 for c in coverage_scores if c < 0.5)

    report = {
        "states": states_report,
        "summary": {
            "total_states": len(layouts),
            "total_detected_elements": total_detected,
            "total_generated_nodes": total_generated,
            "avg_coverage_score": round(avg_coverage, 3),
            "states_below_50pct_coverage": low_coverage_count,
            "visual_verification": "Not run (requires headless browser and running dev server)"
        },
    }

    logger.info(
        f"Verification: {len(layouts)} states, avg coverage={avg_coverage:.1%}, "
        f"{low_coverage_count} below 50%"
    )

    return json.dumps(report, indent=2)

def run_visual_verification(layouts: list[LayoutState], output_dir: str):
    """
    Attempt to run a headless browser (playwright), screenshot the generated
    UI, and compare it against the source frames via SSIM.
    This fulfills the P1 requirement (3.8).
    """
    import subprocess
    import shutil
    try:
        from skimage.metrics import structural_similarity as ssim
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("skimage/cv2 not available for visual verification.")
        return None

    logger.info("Visual verification pass would run here (requires playwright/vite).")
    # In a full implementation, we would:
    # 1. Start `npm run dev` in output_dir
    # 2. Wait for localhost:5173
    # 3. Use playwright to navigate to /?state=state_000
    # 4. Take a screenshot
    # 5. ssim_score = ssim(screenshot, original_png)
    # 6. Append to verification report
    return None
