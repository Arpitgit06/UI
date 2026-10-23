"""Scan complete recordings, select stable states and bounded motion samples, and persist an explicit full-duration timeline."""
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
    video_path: str, output_dir: Path, ssim_change_threshold: float = 0.01,
    stability_ssim_threshold: float = 0.95, ssim_proxy_width: int = 640,
    frame_stride: int = 1, motion_diff_threshold: int = 25,
    motion_min_area: int = 50, motion_max_area_fraction: float = 0.6,
    drag_distance_px: float = 40.0, click_max_duration_sec: float = 0.35,
    max_key_states: int = 30, ambient_grid_rows: int = 8,
    ambient_grid_cols: int = 8, ambient_threshold_fraction: float = 0.6,
    dedup_ssim_threshold: float = 0.995, stability_duration_sec: float = 0.12,
    motion_sample_interval_sec: float = 1.5,
) -> list[KeyStateFrame]:
    """Scan the entire clip, select bounded candidates, then decode selected frames.

    Pixel-local motion plus a dwell window prevents a small moving control from
    being mistaken for a stable screen. Continuous artwork still gets periodic
    samples and an end frame. The capture budget never terminates decoding.
    """
    import json
    import math
    from app.config import settings

    if frame_stride < 1 or max_key_states < 2:
        raise ValueError("frame_stride must be >= 1 and max_key_states >= 2")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoReadError(f"Could not open video at {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise VideoReadError("Video has no valid frame rate")
    candidates = []
    previous = reference = None
    samples = []
    stable_count = 0
    required_stable = max(1, int(np.ceil(stability_duration_sec * fps / frame_stride)))
    raw_index = -1
    last_candidate_index = 0
    frame_h = frame_w = 0
    instability = np.zeros((ambient_grid_rows, ambient_grid_cols), dtype=np.int32)
    comparisons = 0
    last_proxy = None

    def changed(a, b):
        delta = np.max(cv2.absdiff(a, b), axis=2)
        # Local changes survive even if global SSIM barely moves (tooltips/text).
        local = np.count_nonzero(delta > motion_diff_threshold) / delta.size
        score = float(ssim(a, b, channel_axis=-1, data_range=255))
        return ((1.0 - score) > ssim_change_threshold or local > 0.001), score

    def add(index, proxy, reason):
        nonlocal reference, samples, last_candidate_index
        different, score = (True, 1.0) if reference is None else changed(reference, proxy)
        if not different:
            return
        cursor, action = _classify_action(samples, fps, drag_distance_px, click_max_duration_sec)
        candidates.append({"frame_index": index, "timestamp_sec": index / fps,
                           "cursor_position": cursor, "inferred_action": action,
                           "action_confidence": 0.25 if cursor else 0.0,
                           "ssim_delta_from_previous": None if reference is None else round(1-score, 5),
                           "capture_reason": reason})
        reference = proxy.copy()
        last_candidate_index = index
        samples = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_index += 1
            if raw_index / fps > settings.max_video_seconds:
                raise VideoReadError(f"Video exceeds {settings.max_video_seconds} second limit")
            frame_h, frame_w = frame.shape[:2]
            if frame_h * frame_w > settings.max_video_pixels:
                raise VideoReadError("Video resolution exceeds configured pixel limit")
            if raw_index % frame_stride:
                continue
            proxy = _downscale(frame, ssim_proxy_width)
            last_proxy = proxy
            gray = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)
            if previous is None:
                add(raw_index, proxy, "initial")
            else:
                previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
                instability += _build_instability_grid(previous_gray, gray, ambient_grid_rows, ambient_grid_cols)
                comparisons += 1
                delta = np.max(cv2.absdiff(previous, proxy), axis=2)
                local_motion = np.count_nonzero(delta > motion_diff_threshold) / delta.size
                instant = ssim(previous, proxy, channel_axis=-1, data_range=255)
                stable = local_motion < 0.0003 and instant >= stability_ssim_threshold
                stable_count = stable_count + 1 if stable else 0
                sample = _motion_centroid(previous_gray, gray, motion_diff_threshold,
                                          max(3, int(motion_min_area * (proxy.shape[1]/frame_w)**2)),
                                          motion_max_area_fraction)
                if sample:
                    scale = frame_w / proxy.shape[1]
                    samples.append(_MotionSample(raw_index, sample[0]*scale, sample[1]*scale, sample[2]))
                    samples = samples[-int(fps * 10):]
                if stable_count >= required_stable:
                    add(raw_index, proxy, "stable")
                elif (raw_index - last_candidate_index) / fps >= motion_sample_interval_sec:
                    add(raw_index, proxy, "motion_sample")
            previous = proxy
        if raw_index < 0:
            raise VideoReadError(f"No frames could be read from {video_path}")
        if last_proxy is not None:
            add(raw_index - raw_index % frame_stride, last_proxy, "final")
    finally:
        cap.release()

    candidate_count = len(candidates)
    if candidate_count > max_key_states:
        indices = np.linspace(0, candidate_count-1, max_key_states, dtype=int)
        selected = [candidates[i] for i in indices]
    else:
        selected = candidates
    ambient_regions = _detect_ambient_regions(instability, comparisons,
        ambient_grid_rows, ambient_grid_cols, frame_h, frame_w, ambient_threshold_fraction)
    # Sequential second decode avoids inaccurate seeking in inter-frame codecs.
    wanted = {item["frame_index"]: item for item in selected}
    key_states = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        index = -1
        while wanted:
            ok, frame = cap.read()
            if not ok:
                raise VideoReadError("Video could not be decoded consistently on second pass")
            index += 1
            if index not in wanted:
                continue
            item = wanted.pop(index)
            image_path = output_dir / f"state_{index:06d}.png"
            if not cv2.imwrite(str(image_path), frame):
                raise VideoReadError(f"Unable to write {image_path}")
            key_states.append(KeyStateFrame(**item, image_path=str(image_path),
                                             ambient_motion_regions=ambient_regions))
    finally:
        cap.release()
    manifest = {"fps": fps, "frames_scanned": raw_index+1, "duration_sec": (raw_index+1)/fps,
                "analyzed_through_sec": raw_index/fps, "full_video_scanned": True,
                "candidate_count": candidate_count, "selected_count": len(selected),
                "omitted_candidates": candidate_count-len(selected),
                "states": [state.model_dump() for state in key_states]}
    (output_dir / "timeline.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Module A: scanned %s frames; selected %s/%s candidates across %.2fs",
                raw_index+1, len(selected), candidate_count, (raw_index+1)/fps)
    return key_states


async def extract_key_states(video_path: str, output_dir: Path, **kwargs) -> list[KeyStateFrame]:
    return await asyncio.to_thread(_extract_key_states_sync, video_path, output_dir, **kwargs)
