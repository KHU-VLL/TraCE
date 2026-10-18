"""Draw a trajectory-concept medoid on a frame of its source video.

A medoid is stored as L per-step displacements ``d_r = p_{r+1} - p_r`` on the flow
grid, seeded at pixel ``(x, y)`` of the segment that starts at flow index
``start_frame`` (flow index r maps frame r to frame r + 1). Its positions are
``p_0 = (x, y)`` and ``p_{r+1} = p_r + d_r`` (L + 1 points).

Example::

    from src.discovery.trajectory.concepts import load_bundle
    from src.discovery.trajectory.visualize import draw_medoid, read_video_frames
    trajs, info = load_bundle("checkpoints/trajectory_concepts/ntu_mutual")
    meta = info["medoids"][k]
    frames = read_video_frames("videos/" + meta["sample_id"] + ".avi")
    image = draw_medoid(frames, trajs[k], meta)   # RGB uint8 [H, W, 3]
"""
from __future__ import annotations

import cv2
import numpy as np


def medoid_points(trajectory: np.ndarray, x: float, y: float, scale_xy: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """``[L+1, 2]`` absolute (x, y) positions, scaled from the flow grid to the target image."""
    steps = np.asarray(trajectory, dtype=np.float32).reshape(-1, 2)
    pts = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(steps, axis=0)], axis=0)
    pts += np.asarray([x, y], dtype=np.float32)
    return pts * np.asarray(scale_xy, dtype=np.float32)


def draw_medoid(
    frames,
    trajectory: np.ndarray,
    metadata: dict,
    frame: str = "start",
    thickness: int = 2,
    zoom: float = 1.0,
) -> np.ndarray:
    """Overlay one medoid polyline on a frame of its source video.

    Args:
        frames: sequence of RGB frames ``[H, W, 3]`` (uint8) of the medoid's source video,
            decoded at the original frame rate (any resolution; coordinates are rescaled).
        trajectory: ``[L, 2]`` medoid displacements (flow-grid pixels).
        metadata: dict with ``start_frame``, ``x``, ``y`` and optionally ``grid_hw``
            (flow-grid size; defaults to the frame size).
        frame: ``"start"`` draws on the segment start frame, ``"end"`` on the last frame of the segment.
        zoom: optional output upscaling factor (useful for small videos such as KTH).
    Returns:
        RGB uint8 image with the polyline (blue -> red over time), a green start dot and an end arrow.
    """
    length = int(np.asarray(trajectory).reshape(-1, 2).shape[0])
    start = int(metadata["start_frame"])
    idx = start if frame == "start" else start + length
    idx = int(np.clip(idx, 0, len(frames) - 1))
    image = np.ascontiguousarray(np.asarray(frames[idx], dtype=np.uint8).copy())
    h, w = image.shape[:2]
    grid_h, grid_w = metadata.get("grid_hw") or (h, w)
    if zoom != 1.0:
        image = cv2.resize(image, (int(round(w * zoom)), int(round(h * zoom))), interpolation=cv2.INTER_LINEAR)
    sx, sy = image.shape[1] / grid_w, image.shape[0] / grid_h
    pts = medoid_points(trajectory, metadata["x"], metadata["y"], (sx, sy))
    ipts = np.round(pts).astype(np.int32)

    n = len(ipts) - 1
    for r in range(n):
        t = r / max(n - 1, 1)
        color = (int(255 * t), 0, int(255 * (1 - t)))  # RGB: blue -> red
        cv2.line(image, tuple(map(int, ipts[r])), tuple(map(int, ipts[r + 1])), color, thickness, cv2.LINE_AA)
    if n >= 1:
        cv2.arrowedLine(image, tuple(map(int, ipts[-2])), tuple(map(int, ipts[-1])), (255, 0, 0),
                        thickness, cv2.LINE_AA, tipLength=0.5)
    cv2.circle(image, tuple(map(int, ipts[0])), thickness + 2, (0, 200, 0), -1, cv2.LINE_AA)
    return image


def read_video_frames(path: str) -> list[np.ndarray]:
    """Decode all frames of a video as RGB arrays (convenience wrapper around OpenCV)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
