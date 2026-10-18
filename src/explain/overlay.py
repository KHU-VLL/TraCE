"""Animated trajectory-concept overlays in the style of the paper figures.

For a trajectory concept, the medoid trajectory of its cluster is drawn as a
pink "glow" arrow on the input clip. The arrow starts at the Grad-CAM peak of
the concept (where in the clip the concept responds) and grows frame by frame
from the peak frame on. Grad-CAM only places the drawing; it is not part of
the explanation scores.

Typical use (see tools/explain_ar.py and tools/explain_vqa.py, ``--overlay``)::

    t, x, y = find_cam_peak(cam)                       # cam: [T, H, W] in [0, 1]
    frames = animate_trajectory(frames, medoid, (x, y), t)
    grid = build_grid(frames_orig, {"[T] trajectory_051": frames}, title=...)
    save_animation(grid, "overlay.gif")
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PINK = (255, 105, 180)          # RGB, arrow and start marker
BACKGROUND = (38, 38, 38)       # RGB, grid background
TEXT = (230, 230, 230)


def find_cam_peak(cam: np.ndarray, temporal_window: int = 3, spatial_window: int = 1) -> Tuple[int, int, int]:
    """(t, x, y) of the maximum of a [T, H, W] CAM after an optional box filter."""
    tw, sw = max(1, int(temporal_window)), max(1, int(spatial_window))
    if tw > 1 or sw > 1:
        from scipy.ndimage import uniform_filter
        cam = uniform_filter(cam, size=(tw, sw, sw), mode="reflect")
    t, y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return int(t), int(x), int(y)


def trajectory_points(displacement: np.ndarray, anchor: Tuple[float, float], extent: float = 40.0) -> np.ndarray:
    """[L, 2] step displacements -> [L+1, 2] pixel positions starting at ``anchor``.

    The path is rescaled so that the larger side of its bounding box is
    ``extent`` pixels (direction and shape are kept, absolute speed is not).
    """
    cum = np.concatenate([np.zeros((1, 2), np.float32), np.cumsum(np.asarray(displacement, np.float32), 0)], 0)
    span = float(max(np.ptp(cum[:, 0]), np.ptp(cum[:, 1])))
    scale = extent / span if span > 1e-6 else 1.0
    return cum * scale + np.asarray(anchor, np.float32)[None]


def draw_glow_trajectory(frame: np.ndarray, points: np.ndarray, n_show: int, thickness: float = 1.0,
                         color: Tuple[int, int, int] = PINK) -> np.ndarray:
    """Draw the first ``n_show`` steps of ``points`` on an RGB frame: start ring,
    shadowed path with a white core, white arrow head with a colored outline."""
    s = float(thickness)
    main = (color[2], color[1], color[0])                      # BGR
    dark = tuple(max(0, c - 50) for c in main)
    white = (255, 255, 255)
    img = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    start = (int(points[0][0]), int(points[0][1]))
    r_out, r_in = max(3, round(8 * s)), max(2, round(5 * s))
    cv2.circle(img, start, r_out, main, -1, cv2.LINE_AA)
    cv2.circle(img, start, r_in, white, -1, cv2.LINE_AA)
    cv2.circle(img, start, r_out, main, max(1, round(2 * s)), cv2.LINE_AA)
    n_show = min(int(n_show), len(points) - 1)
    if n_show >= 1:
        pts = np.array([[np.clip(int(p[0]), 3, W - 4), np.clip(int(p[1]), 3, H - 4)] for p in points[:n_show + 1]],
                       np.int32)
        cv2.polylines(img, [pts + 1], False, dark, max(2, round(6 * s)), cv2.LINE_AA)
        cv2.polylines(img, [pts], False, main, max(2, round(6 * s)), cv2.LINE_AA)
        cv2.polylines(img, [pts], False, white, max(1, round(2 * s)), cv2.LINE_AA)
        far = pts[-min(5, len(pts))].astype(float)
        d = pts[-1].astype(float) - far
        n = float(np.linalg.norm(d))
        d = d / n if n >= 1 else np.array([1.0, 0.0])
        perp = np.array([-d[1], d[0]])
        a = 14.0 * s
        tip = pts[-1] + d * 4
        head = np.array([tip, tip - d * a + perp * a * 0.4, tip - d * a - perp * a * 0.4])
        head = np.stack([np.clip(head[:, 0], 3, W - 4), np.clip(head[:, 1], 3, H - 4)], 1).astype(np.int32)
        cv2.fillPoly(img, [head], white, cv2.LINE_AA)
        cv2.polylines(img, [head], True, main, max(2, round(3 * s)), cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def animate_trajectory(frames: np.ndarray, displacement: np.ndarray, anchor: Tuple[float, float],
                       peak_frame: int, extent: float = 40.0, speed: float = 2.0,
                       thickness: float = 1.0) -> np.ndarray:
    """Overlay a growing trajectory on [T, H, W, 3] RGB frames.

    Nothing is drawn before ``peak_frame``; from there the path grows and is
    complete after ``(T - peak_frame) / speed`` frames.
    """
    pts = trajectory_points(displacement, anchor, extent)
    T, steps = len(frames), len(pts) - 1
    remaining = max(1, T - peak_frame)
    out = []
    for t in range(T):
        if t < peak_frame:
            out.append(frames[t])
            continue
        progress = min(1.0, (t - peak_frame + 1) / remaining * speed)
        out.append(draw_glow_trajectory(frames[t], pts, max(1, round(progress * steps)), thickness))
    return np.stack(out)


def _put(img, text, org, scale=0.45, color=TEXT, bold=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color[::-1], bold, cv2.LINE_AA)


def build_grid(original: np.ndarray, rows: Dict[str, Optional[np.ndarray]], title: str = "",
               cell: int = 224, label_w: int = 150, gap: int = 4) -> np.ndarray:
    """[T, H, W, 3] original + {row label: [T, H, W, 3] overlay or None} -> [T, Hg, Wg, 3] grid.

    Rows are concepts; columns are "Original" and "Trajectory" (empty for
    concepts without a trajectory drawing, e.g. object / scene concepts).
    """
    T = len(original)
    rs = lambda a: cv2.resize(a, (cell, cell), interpolation=cv2.INTER_AREA)
    head = 48 if title else 26
    Hg = head + len(rows) * (cell + gap)
    Wg = label_w + 2 * cell + 3 * gap
    out = np.empty((T, Hg, Wg, 3), np.uint8)
    for t in range(T):
        g = np.full((Hg, Wg, 3), BACKGROUND, np.uint8)
        if title:
            _put(g, title, (6, 18), 0.5, bold=1)
        for c, name in enumerate(["Original", "Trajectory"]):
            _put(g, name, (label_w + gap + c * (cell + gap) + cell // 2 - 35, head - 8), 0.45)
        for r, (label, ov) in enumerate(rows.items()):
            y = head + r * (cell + gap)
            for k, line in enumerate(label.split("\n")):
                _put(g, line, (6, y + cell // 2 + 16 * k), 0.42)
            g[y:y + cell, label_w + gap:label_w + gap + cell] = rs(original[t])
            if ov is not None:
                x = label_w + 2 * gap + cell
                g[y:y + cell, x:x + cell] = rs(ov[t])
        out[t] = g
    return out


def save_animation(frames: np.ndarray, path: str, fps: int = 4) -> str:
    """Save [T, H, W, 3] RGB frames as .gif (PIL) or .mp4 (OpenCV, mp4v)."""
    if path.endswith(".gif"):
        from PIL import Image
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
    else:
        h, w = frames.shape[1:3]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
    return path


def interpolate_cam(cam: np.ndarray, n: int) -> np.ndarray:
    """Linearly interpolate a [T, H, W] CAM to ``n`` frames (display only)."""
    src = np.linspace(0, 1, len(cam))
    dst = np.linspace(0, 1, n)
    i = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, len(cam) - 2) if len(cam) > 1 else np.zeros(n, int)
    if len(cam) == 1:
        return np.repeat(cam, n, 0)
    w = ((dst - src[i]) / (src[i + 1] - src[i]))[:, None, None]
    return (1 - w) * cam[i] + w * cam[i + 1]


def temporal_resample(frames: np.ndarray, n: int) -> np.ndarray:
    """Pick ``n`` evenly spaced frames (used to show short model clips as smoother videos)."""
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return frames[idx]


def concept_rows(labels: Sequence[str], overlays: List[Optional[np.ndarray]]) -> Dict[str, Optional[np.ndarray]]:
    return {lab: ov for lab, ov in zip(labels, overlays)}


def concept_overlay_grid(frames: np.ndarray, items: Sequence[tuple], cams: Dict[int, np.ndarray],
                         medoids: Optional[np.ndarray], vocab, title: str = "", extent: float = 40.0,
                         speed: float = 2.0, temporal_window: int = 3, hold: int = 2,
                         max_start: Optional[float] = None) -> np.ndarray:
    """Paper-style grid for the top concepts of one prediction.

    frames: [T, H, W, 3] RGB clip shown to the model; items: (concept_id, group,
    name, contribution) tuples (e.g. ``model.top_concepts(...)``); cams:
    concept_id -> [T, H, W] Grad-CAM of that concept on the same frames;
    medoids: [Mt, L, 2] trajectory medoids indexed by the concept's
    ``source_index`` (cluster id), or None. The last frame is repeated
    ``hold`` times so that the completed trajectories stay visible.
    ``max_start`` (fraction of the clip, display only) makes a drawing start no
    later than that point, e.g. 0.25 for short model clips whose CAM peak is
    quantized to few frames.
    """
    tag = {"trajectory": "T", "object": "O", "scene": "S"}
    labels, overlays = [], []
    for cid, group, name, value in items:
        labels.append(f"[{tag.get(group, group)}] {name}\ncontrib {value:+.3f}")
        if group == "trajectory" and medoids is not None and cid in cams:
            t, x, y = find_cam_peak(cams[cid], temporal_window=temporal_window)
            if max_start is not None:
                t = min(t, int(max_start * len(frames)))
            k = vocab[cid].source_index
            overlays.append(animate_trajectory(frames, medoids[k], (x, y), t, extent, speed))
        else:
            overlays.append(None)
    grid = build_grid(frames, concept_rows(labels, overlays), title)
    return np.concatenate([grid, np.repeat(grid[-1:], hold, 0)]) if hold > 0 else grid
