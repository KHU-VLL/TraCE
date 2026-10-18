"""Per-video trajectory filtering (foreground gating + top-rho% motion).

This reproduces ``prepare_sample`` of the research code for the setting used in
all final runs: segment-based stacks, foreground gating on the mask of each
segment's *start* frame, then a per-video percentile threshold on the motion
magnitude computed over the foreground trajectories only, and ``flatten``
features (the raw ``[L, 2]`` displacement sequence, no normalisation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .masks import count_mask_frames, load_mask
from .stacks import compute_segment_start_indices, load_trajectory_stack


@dataclass
class PreparedSample:
    sample_id: str
    input_path: Path
    raw_shape: tuple[int, ...]       # (K, L, H, W, 2)
    features: np.ndarray             # [N, 2L] float32
    coords: np.ndarray               # [N, 3] int32 (segment_id, y, x)
    kept_indices: np.ndarray         # [N] int32 flat index into K*H*W
    start_frames: np.ndarray         # [N] int32 start flow index of the segment
    extra_info: dict = field(default_factory=dict)


def flat_indices_to_coords(flat_indices: np.ndarray, height: int, width: int) -> np.ndarray:
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    hw = height * width
    rem = flat_indices % hw
    return np.stack([flat_indices // hw, rem // width, rem % width], axis=1).astype(np.int32)


def compute_path_lengths_from_stacks(stacks: np.ndarray, diff_dtype: str = "stack") -> np.ndarray:
    """Motion magnitude used for the top-rho% filter, flattened over ``[K, H, W]``.

    NOTE: identical to the research code, the magnitude is the summed norm of the
    differences of consecutive stored steps, sum_r ||s_{r+1} - s_r||. Since the
    stack stores displacements, this is the summed change of displacement
    (an acceleration-like quantity), not sum_r ||Delta p_r|| as in Eq. (17).

    ``diff_dtype="stack"`` takes the differences in the stack dtype (float16) and
    is what the NTU / UCF-101 / TOMATO / STAR runs and all val labelling used.
    ``diff_dtype="float32"`` up-casts first, as the earlier code version used for
    the Chiral SSv2 and KTH clustering runs. The two differ only by float16
    rounding, which can flip trajectories that sit exactly at the threshold.
    """
    num_segments, length, height, width, _ = stacks.shape
    out = np.empty(num_segments * height * width, dtype=np.float32)
    if length < 2:
        out.fill(0.0)
        return out
    stride = height * width
    for k in range(num_segments):
        segment = stacks[k] if diff_dtype == "stack" else np.asarray(stacks[k], dtype=np.float32)
        step_vectors = np.diff(segment, axis=0)
        step_lengths = np.sqrt((step_vectors.astype(np.float32, copy=False) ** 2).sum(axis=-1))
        out[k * stride:(k + 1) * stride] = step_lengths.sum(axis=0, dtype=np.float32).reshape(-1)
    return out


def resolve_num_flow_frames(
    sample_id: str,
    num_flow_frames: int | None = None,
    flow_root: Path | None = None,
    mask_root: Path | None = None,
) -> int:
    """Number of flow fields T of a video, needed to recompute the segment start frames.

    Priority: explicit value > ``<flow_root>/[<subdir>/]<sample_id>.npy`` > (#masks - 1).
    """
    if num_flow_frames is not None:
        return int(num_flow_frames)
    if flow_root is not None and flow_root.is_dir():
        candidates = [flow_root / f"{sample_id}.npy"]
        candidates += [d / f"{sample_id}.npy" for d in sorted(flow_root.iterdir()) if d.is_dir()]
        existing = [p for p in candidates if p.exists()]
        if len(existing) > 1:
            raise RuntimeError(f"Multiple flow files for {sample_id}: {existing}")
        if existing:
            return int(np.load(existing[0], mmap_mode="r").shape[0])
    if mask_root is not None:
        # one mask per video frame, one flow field per pair of frames
        return max(count_mask_frames(mask_root, sample_id) - 1, 1)
    raise FileNotFoundError(f"Cannot determine the number of flow frames for {sample_id}")


def prepare_sample(
    input_path: Path,
    mask_root: Path,
    motion_percentile: float = 95.0,
    flow_root: Path | None = None,
    num_flow_frames: int | None = None,
    path_length_dtype: str = "stack",
) -> PreparedSample:
    """Filter the dense trajectories of one video and build clustering features."""
    input_path = Path(input_path)
    stacks = load_trajectory_stack(input_path)
    sample_id = input_path.stem
    raw_shape = tuple(int(v) for v in stacks.shape)
    num_segments, length, height, width, _ = raw_shape
    total = num_segments * height * width

    t = resolve_num_flow_frames(sample_id, num_flow_frames, flow_root, mask_root)
    segment_start_frames = compute_segment_start_indices(t, length, num_segments).astype(np.int32)
    extra_info: dict = {"num_flow_frames": int(t)}

    # Stage 1: foreground gating with the mask of each segment's start frame.
    selected = np.zeros((num_segments, height, width), dtype=bool)
    for frame_idx in np.unique(segment_start_frames).tolist():
        mask = load_mask(mask_root, sample_id, frame_idx, expected_hw=(height, width))
        selected[np.flatnonzero(segment_start_frames == frame_idx)] = mask[None] == 1
    n_fg = int(selected.sum())
    extra_info.update({
        "num_selected_after_saliency": n_fg,
        "num_filtered_out_by_saliency": int(total - n_fg),
        "saliency_all_empty": n_fg == 0,
    })

    # Stage 2: keep trajectories whose magnitude >= per-video percentile (over foreground only).
    if n_fg > 0:
        path_lengths = compute_path_lengths_from_stacks(stacks, path_length_dtype).reshape(num_segments, height, width)
        fg_lengths = path_lengths[selected]
        threshold = float(np.percentile(fg_lengths, motion_percentile))
        motion_mask = np.zeros_like(selected)
        motion_mask[selected] = fg_lengths >= threshold
        selected &= motion_mask
        extra_info["motion_filter"] = {
            "threshold_source": f"percentile_{motion_percentile:g}",
            "threshold_value": threshold,
            "num_selected_after_motion": int(selected.sum()),
            "path_length_min": float(fg_lengths.min()),
            "path_length_median": float(np.median(fg_lengths)),
            "path_length_mean": float(fg_lengths.mean()),
            "path_length_max": float(fg_lengths.max()),
        }
    else:
        extra_info["motion_filter"] = {"threshold_source": "skipped_empty_input"}

    kept_indices = np.flatnonzero(selected.reshape(-1)).astype(np.int32)
    coords = flat_indices_to_coords(kept_indices, height, width)
    trajectories = np.asarray(stacks[coords[:, 0], :, coords[:, 1], coords[:, 2], :], dtype=np.float32)
    features = trajectories.reshape(len(kept_indices), length * 2)
    return PreparedSample(
        sample_id=sample_id,
        input_path=input_path,
        raw_shape=raw_shape,
        features=features,
        coords=coords,
        kept_indices=kept_indices,
        start_frames=segment_start_frames[coords[:, 0]].astype(np.int32),
        extra_info=extra_info,
    )
