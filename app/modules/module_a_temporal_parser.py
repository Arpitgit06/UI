"""
Module A: Temporal Video Parser (CPU / light GPU)

Reads an MP4/WebM recording and reduces it to a handful of "key state"
images -- the frames where the UI actually settled into a genuinely new
state -- using SSIM-based frame diffing, plus a lightweight
motion-centroid heuristic to guess where the change originated
(cursor/touch position) and whether it looks like a hover, click, or
drag.

Design notes (read before tuning thresholds -- each of these was found
by testing against synthetic frames/videos, not assumed):

  1. Keyframe decisions compare each incoming frame against the LAST
     CAPTURED key state, not the literally-previous raw frame (frame
     N-1). Diffing only against N-1 misses slow, gradual transitions
     (a fading modal, a slow drag): no single step ever crosses the
     threshold even though the cumulative change is large. Comparing
     against a persistent reference correctly accumulates that drift.

  2. Reference-diffing alone isn't enough, though: during a
     *continuous* motion (e.g. a slow drag), every incremental frame
     already differs enough from the reference, so a naive
     "capture whenever changed_enough" rule fires on almost every frame
     of the drag and resets the reference each time -- instead of one
     "before" and one "after" state, you get dozens of near-duplicate
     captures, one per few pixels of motion. The fix is a second,
     independent check: only capture once the video has gone STABLE
     frame-to-frame again (i.e. motion has stopped), then test that
     settled frame against the reference. This applies uniformly to
     both sudden jumps (a modal appearing in one frame) and gradual
     ones (a drag) -- sudden jumps just settle one frame later, which
     is a negligible, deliberate trade-off.

  3. The SSIM comparisons run on a downscaled COLOR proxy, not
     grayscale. Grayscale SSIM is faster but risks under-weighting
     color-only UI changes (a hover/error state that shifts hue at
     similar luminance) -- and exact colors are the whole point of this
     project, so channel information is kept in both comparisons.

  4. Motion/cursor tracking runs on full-resolution GRAYSCALE frames
     (color doesn't matter for localizing "where did something change",
     and skipping color keeps this cheap enough to run on every raw
     frame). It's classical frame-differencing + centroid-of-the-diff,
     not a learned cursor detector -- there's no guarantee a real OS
     cursor sprite is even visible in the recording. Treat
     inferred_action and cursor_position as a best-effort heuristic to
     be retuned against real recordings, not ground truth.

  5. Ambient motion detection (v2): regions of the frame that keep
     changing for a large fraction of the clip duration (e.g. rotating
     rings, pulsing animations, floating particles) are detected by
     tracking per-grid-cell instability over time. If a cell fails the
     stability check for >60% of the clip, it's flagged as "ambient
     motion" and changes in that region are ignored for key-state
     decisions. These regions are reported so Module D can express them
     as CSS @keyframes or R3F animations rather than emitting one
     StateN.jsx per frame of the loop.

  6. Post-capture deduplication: after all key states are extracted,
     a final pass merges states whose SSIM > 0.97 (near-identical
     frames that slipped through, typically from small sub-threshold
     ambient drift). This is a safety net, not the primary filter.
"""
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.models.schemas import KeyStateFrame
from app.utils.logger import get_logger

logger = get_logger("omniui.module_a")


class VideoReadError(RuntimeError):
    """Raised when the source video can't be opened or yields no frames."""


@dataclass
class _MotionSample:
    frame_offset: int  # raw frame index, so sample spans can be turned into elapsed seconds
    cx: float
    cy: float
    area: int


def _downscale(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _motion_centroid(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    diff_threshold: int,
    min_area: int,
    max_area_fraction: float,
) -> Optional[tuple[float, float, int]]:
    """
    Center of mass of the pixels that changed between two grayscale
    frames. Returns None if the changed region is too small to trust
    (likely compression noise) or too large to localize meaningfully
    (a near-global change, e.g. a full theme swap or scene cut, isn't
    a "cursor position").
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(mask)
    total_pixels = prev_gray.shape[0] * prev_gray.shape[1]
    if len(xs) < min_area or len(xs) > total_pixels * max_area_fraction:
        return None
    return float(xs.mean()), float(ys.mean()), int(len(xs))


def _classify_action(
    samples: list[_MotionSample],
    fps: float,
    drag_distance_px: float,
    click_max_duration_sec: float,
) -> tuple[Optional[tuple[float, float]], str]:
    """
    Heuristic classification of the transition into a new key state,
    from the motion centroids sampled since the previous one. This is
    a first-pass heuristic, not a learned classifier -- expect to
    retune the thresholds against real screen recordings.
    """
    if not samples:
        return None, "none"  # e.g. a non-localized global change, or the initial frame

    last = samples[-1]
    position = (last.cx, last.cy)
    first = samples[0]

    total_travel = ((last.cx - first.cx) ** 2 + (last.cy - first.cy) ** 2) ** 0.5
    if total_travel >= drag_distance_px:
        return position, "drag"

    elapsed_frames = last.frame_offset - first.frame_offset
    elapsed_sec = (elapsed_frames / fps) if fps > 0 else 0.0
    if elapsed_sec <= click_max_duration_sec:
        return position, "click"
    return position, "hover"


# ---------------------------------------------------------------------------
# Ambient motion detection — grid-based regional instability tracking
# ---------------------------------------------------------------------------


def _build_instability_grid(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    cell_change_threshold: float = 15.0,
) -> np.ndarray:
    """
    Returns a boolean grid (grid_rows x grid_cols) where True means the
    cell changed between prev and curr frames (mean absolute diff exceeds
    threshold).
    """
    h, w = prev_gray.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    changed = np.zeros((grid_rows, grid_cols), dtype=bool)
    
    diff = cv2.absdiff(prev_gray, curr_gray).astype(np.float32)
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < grid_rows - 1 else h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < grid_cols - 1 else w
            cell_mean = diff[y1:y2, x1:x2].mean()
            changed[r, c] = cell_mean > cell_change_threshold
    
    return changed


def _detect_ambient_regions(
    instability_counts: np.ndarray,
    total_comparisons: int,
    grid_rows: int,
    grid_cols: int,
    frame_h: int,
    frame_w: int,
    ambient_threshold_fraction: float = 0.6,
) -> list[tuple[float, float, float, float]]:
    """
    Returns bounding boxes (x, y, w, h) in pixel coordinates for grid
    cells that were unstable for more than ambient_threshold_fraction of
    the clip's frame comparisons.
    """
    if total_comparisons < 1:
        return []
    
    cell_h = frame_h / grid_rows
    cell_w = frame_w / grid_cols
    regions: list[tuple[float, float, float, float]] = []
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            fraction = instability_counts[r, c] / total_comparisons
            if fraction >= ambient_threshold_fraction:
                x = c * cell_w
                y = r * cell_h
                regions.append((x, y, cell_w, cell_h))
    
    return regions


def _is_in_ambient_region(
    proxy: np.ndarray,
    reference_proxy: np.ndarray,
    ambient_mask: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    ssim_change_threshold: float,
) -> bool:
    """
    Check if the change between proxy and reference is ONLY in ambient
    motion regions. If all changed cells are ambient, don't capture.
    Returns True if the change is entirely ambient (should NOT capture).
    """
    if not ambient_mask.any():
        return False  # no ambient regions detected yet
    
    h, w = proxy.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    
    diff = cv2.absdiff(proxy.astype(np.float32), reference_proxy.astype(np.float32))
    
    has_non_ambient_change = False
    for r in range(grid_rows):
        for c in range(grid_cols):
            if ambient_mask[r, c]:
                continue  # skip ambient cells
            y1 = r * cell_h
            y2 = (r + 1) * cell_h if r < grid_rows - 1 else h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w if c < grid_cols - 1 else w
            cell_mean = diff[y1:y2, x1:x2].mean()
            if cell_mean > 5.0:  # non-trivial change in a non-ambient cell
                has_non_ambient_change = True
                break
        if has_non_ambient_change:
            break
    
    return not has_non_ambient_change


# ---------------------------------------------------------------------------
# Post-capture deduplication
# ---------------------------------------------------------------------------


def _deduplicate_states(
    key_states: list[KeyStateFrame],
    dedup_ssim_threshold: float = 0.97,
    ssim_proxy_width: int = 480,
) -> list[KeyStateFrame]:
    """
    Merge near-identical states that slipped through the primary filter
    (typically from small sub-threshold ambient drift). Keeps the first
    occurrence of each visually-distinct group.
    """
    if len(key_states) <= 1:
        return key_states

    unique: list[KeyStateFrame] = [key_states[0]]
    prev_proxy = _downscale(cv2.imread(key_states[0].image_path), ssim_proxy_width)

    for state in key_states[1:]:
        curr_proxy = _downscale(cv2.imread(state.image_path), ssim_proxy_width)
        try:
            score = ssim(prev_proxy, curr_proxy, channel_axis=-1, data_range=255)
        except Exception:
            score = 0.0  # treat unreadable as different

        if score < dedup_ssim_threshold:
            unique.append(state)
            prev_proxy = curr_proxy
        else:
            logger.debug(f"Dedup: dropping {state.image_path} (ssim={score:.4f} with previous)")

    if len(unique) < len(key_states):
        logger.info(f"Deduplication: {len(key_states)} → {len(unique)} states (threshold={dedup_ssim_threshold})")

    return unique


def _extract_key_states_sync(
    video_path: str,
    output_dir: Path,
    ssim_change_threshold: float = 0.01,
    stability_ssim_threshold: float = 0.95,
    ssim_proxy_width: int = 480,
    frame_stride: int = 1,
    motion_diff_threshold: int = 25,
    motion_min_area: int = 50,
    motion_max_area_fraction: float = 0.6,
    drag_distance_px: float = 40.0,
    click_max_duration_sec: float = 0.35,
    max_key_states: int = 30,
    ambient_grid_rows: int = 4,
    ambient_grid_cols: int = 4,
    ambient_threshold_fraction: float = 0.6,
    dedup_ssim_threshold: float = 0.97,
) -> list[KeyStateFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoReadError(
            f"Could not open video at {video_path}. Confirm the file exists and is a "
            f"valid MP4/WebM (if WebM, your OpenCV build needs FFmpeg VP8/VP9 support)."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    key_states: list[KeyStateFrame] = []
    reference_proxy: Optional[np.ndarray] = None  # proxy of the last CAPTURED key state
    prev_frame_proxy: Optional[np.ndarray] = None  # proxy of the immediately preceding raw frame
    prev_gray: Optional[np.ndarray] = None
    motion_since_last_capture: list[_MotionSample] = []
    raw_index = -1

    # Ambient motion tracking: count how many frame-comparisons each grid cell was unstable
    instability_counts = np.zeros((ambient_grid_rows, ambient_grid_cols), dtype=np.int32)
    total_comparisons = 0
    ambient_mask = np.zeros((ambient_grid_rows, ambient_grid_cols), dtype=bool)
    frame_h, frame_w = 0, 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_index += 1
            if raw_index % frame_stride != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            proxy = _downscale(frame, ssim_proxy_width)
            
            if frame_h == 0:
                frame_h, frame_w = frame.shape[:2]

            # Track regional instability for ambient motion detection
            if prev_gray is not None:
                grid_changed = _build_instability_grid(
                    _downscale(prev_gray.reshape(prev_gray.shape[0], prev_gray.shape[1]), ssim_proxy_width // 2) 
                    if len(prev_gray.shape) == 2 else prev_gray,
                    _downscale(gray.reshape(gray.shape[0], gray.shape[1]), ssim_proxy_width // 2)
                    if len(gray.shape) == 2 else gray,
                    ambient_grid_rows,
                    ambient_grid_cols,
                )
                instability_counts += grid_changed.astype(np.int32)
                total_comparisons += 1
                
                # Update ambient mask periodically (every 30 comparisons or at 20% of clip)
                if total_comparisons > 0 and total_comparisons % 30 == 0:
                    ambient_mask = (instability_counts / total_comparisons) >= ambient_threshold_fraction

            if prev_gray is not None:
                sample = _motion_centroid(
                    prev_gray, gray, motion_diff_threshold, motion_min_area, motion_max_area_fraction
                )
                if sample is not None:
                    motion_since_last_capture.append(_MotionSample(raw_index, *sample))

            is_first_frame = reference_proxy is None
            should_capture = is_first_frame
            change_score: Optional[float] = None
            if not is_first_frame:
                instant_score = ssim(prev_frame_proxy, proxy, channel_axis=-1, data_range=255)
                is_stable_now = instant_score > stability_ssim_threshold
                if is_stable_now:
                    change_score = ssim(reference_proxy, proxy, channel_axis=-1, data_range=255)
                    should_capture = (1.0 - change_score) > ssim_change_threshold
                    
                    # Check if the change is ONLY in ambient regions — if so, don't capture
                    if should_capture and ambient_mask.any():
                        is_ambient_only = _is_in_ambient_region(
                            proxy, reference_proxy, ambient_mask,
                            ambient_grid_rows, ambient_grid_cols, ssim_change_threshold,
                        )
                        if is_ambient_only:
                            should_capture = False
                            logger.debug(f"Frame {raw_index}: change is only in ambient regions, skipping capture")

            if should_capture:
                if len(key_states) >= max_key_states:
                    logger.warning(
                        f"Hit max_key_states={max_key_states}; stopping early. This usually "
                        f"means the input is noisy/flickering rather than a clean UI recording -- "
                        f"consider raising ssim_change_threshold."
                    )
                    break

                if is_first_frame:
                    cursor_position, inferred_action, ssim_delta = None, "none", None
                else:
                    cursor_position, inferred_action = _classify_action(
                        motion_since_last_capture, fps, drag_distance_px, click_max_duration_sec
                    )
                    ssim_delta = round(1.0 - change_score, 4)

                image_path = output_dir / f"state_{len(key_states):03d}.png"
                cv2.imwrite(str(image_path), frame)

                key_states.append(
                    KeyStateFrame(
                        frame_index=raw_index,
                        timestamp_sec=(raw_index / fps) if fps > 0 else 0.0,
                        image_path=str(image_path),
                        ssim_delta_from_previous=ssim_delta,
                        cursor_position=cursor_position,
                        inferred_action=inferred_action,
                    )
                )
                reference_proxy = proxy
                motion_since_last_capture = []

            prev_frame_proxy = proxy
            prev_gray = gray
    finally:
        cap.release()

    if not key_states:
        raise VideoReadError(f"No frames could be read from {video_path}.")

    # Detect final ambient motion regions and attach to all key states
    ambient_regions = _detect_ambient_regions(
        instability_counts, total_comparisons,
        ambient_grid_rows, ambient_grid_cols,
        frame_h, frame_w, ambient_threshold_fraction,
    )
    if ambient_regions:
        logger.info(f"Module A: detected {len(ambient_regions)} ambient motion region(s)")
        for state in key_states:
            state.ambient_motion_regions = ambient_regions

    # Post-capture deduplication: merge near-identical states
    key_states = _deduplicate_states(key_states, dedup_ssim_threshold, ssim_proxy_width)

    logger.info(
        f"Module A: {video_path} -> {len(key_states)} key state(s) "
        f"(fps={fps:.1f}, change_threshold={ssim_change_threshold}, "
        f"stability_threshold={stability_ssim_threshold}, "
        f"ambient_regions={len(ambient_regions)})"
    )
    return key_states


async def extract_key_states(video_path: str, output_dir: Path, **kwargs) -> list[KeyStateFrame]:
    return await asyncio.to_thread(_extract_key_states_sync, video_path, output_dir, **kwargs)
