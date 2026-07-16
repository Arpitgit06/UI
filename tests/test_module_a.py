"""
Tests for Module A (Temporal Video Parser). Builds small synthetic
videos with cv2.VideoWriter so these exercise the real SSIM
keyframe-capture and motion-centroid logic end-to-end, not mocks.

The exact frame indices and classifications asserted below were
confirmed against this file's real implementation (not just a
prototype) before being written -- see the parameter choices in
module_a_temporal_parser.py's module docstring for why two thresholds
(change vs. stability) are both needed.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.modules.module_a_temporal_parser import VideoReadError, _extract_key_states_sync

W, H, FPS = 320, 180, 30


def _make_frame(bg=(30, 30, 30)):
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:] = bg
    return frame


def _draw_square(frame, cx, cy, size, color):
    f = frame.copy()
    half = size // 2
    cv2.rectangle(f, (cx - half, cy - half), (cx + half, cy + half), color, -1)
    return f


def _write_video(path, frame_sequence, fps=FPS):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for frame in frame_sequence:
        writer.write(frame)
    writer.release()


def test_two_sudden_state_changes_produce_three_key_states(tmp_path):
    base = _make_frame()
    state_b = _draw_square(base, 60, 40, 50, (0, 200, 0))
    state_c = _draw_square(base, 260, 140, 50, (0, 200, 0))

    sequence = [base] * 30 + [state_b] * 30 + [state_c] * 30
    video_path = tmp_path / "clip.mp4"
    _write_video(video_path, sequence)

    result = _extract_key_states_sync(str(video_path), tmp_path / "key_states")

    assert len(result) == 3
    # captures land one frame after each jump (30, 60) because of the
    # stability debounce -- see the module docstring, point 2.
    assert [ks.frame_index for ks in result] == [0, 31, 61]
    assert result[0].ssim_delta_from_previous is None
    assert result[1].ssim_delta_from_previous > 0.02
    for ks in result:
        assert Path(ks.image_path).exists()


def test_continuous_drag_collapses_to_start_and_settled_state(tmp_path):
    base = _make_frame()
    positions = [(40 + i * 8, 90) for i in range(20)]
    moving_frames = [_draw_square(base, x, y, 50, (255, 255, 255)) for x, y in positions]
    settled = moving_frames[-1]

    # 10 static, 20 sliding steps, 10 settled -- a naive "diff vs N-1"
    # or "capture whenever changed" approach would over-capture every
    # incremental step of the slide; this asserts it doesn't.
    sequence = [base] * 10 + moving_frames + [settled] * 10
    video_path = tmp_path / "drag.mp4"
    _write_video(video_path, sequence)

    result = _extract_key_states_sync(str(video_path), tmp_path / "key_states")

    assert len(result) == 2, f"expected exactly a start and a settled-after-drag state, got {result}"
    assert result[1].inferred_action == "drag"
    assert result[1].cursor_position is not None


def test_fully_static_video_produces_a_single_key_state(tmp_path):
    video_path = tmp_path / "static.mp4"
    _write_video(video_path, [_make_frame()] * 20)

    result = _extract_key_states_sync(str(video_path), tmp_path / "key_states")

    assert len(result) == 1
    assert result[0].frame_index == 0
    assert result[0].inferred_action == "none"


def test_invalid_video_path_raises_video_read_error(tmp_path):
    with pytest.raises(VideoReadError):
        _extract_key_states_sync(str(tmp_path / "does_not_exist.mp4"), tmp_path / "key_states")


def test_max_key_states_caps_rapidly_changing_input(tmp_path):
    # 9 distinct, clearly-different square positions across a grid,
    # each held for 2 frames so it reads as "stable" -- without a cap
    # this produces well over a dozen captures (one per position jump).
    base = _make_frame()
    grid_positions = [(x, y) for x in (60, 160, 260) for y in (40, 90, 140)] * 2
    frames = []
    for x, y in grid_positions:
        frames.extend([_draw_square(base, x, y, 50, (0, 200, 0))] * 2)

    video_path = tmp_path / "rapid_changes.mp4"
    _write_video(video_path, frames)

    result = _extract_key_states_sync(str(video_path), tmp_path / "key_states", max_key_states=5)

    assert len(result) <= 5
