"""Dense optical-flow trajectories ("trajectory stacks").

For every video the flow sequence ``[T, H, W, 2]`` (T = #frames - 1) is cut into
``num_segments`` windows of ``length`` flow steps centred on equal-duration
segments. Inside each window every pixel is used as a seed (grid stride 1) and
traced by bilinear sampling of the flow at the current sub-pixel position.

The stored value at step ``r`` is the *sampled flow* (the displacement
``p_{r+1} - p_r``), so one trajectory is a ``[length, 2]`` array of
displacements, and a stack is ``[num_segments, length, H, W, 2]``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pad_flow_to_length(flow: np.ndarray, length: int) -> tuple[np.ndarray, int]:
    """Zero-pad a short flow sequence (T < length) along time."""
    num_frames = int(flow.shape[0])
    if num_frames >= length:
        return flow, 0
    if num_frames <= 0:
        raise ValueError("Cannot pad an empty flow sequence.")
    pad_count = length - num_frames
    pad = np.zeros((pad_count, *flow.shape[1:]), dtype=flow.dtype)
    return np.concatenate([flow, pad], axis=0), pad_count


def compute_segment_start_indices(num_frames: int, length: int, num_segments: int) -> np.ndarray:
    """Start flow index of each of the ``num_segments`` windows.

    ``num_frames`` is the number of flow fields T. The window of segment k starts
    at ``round(center_k - length / 2)`` clipped to ``[0, max(T, length) - length]``.
    """
    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive, got {num_segments}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    effective_num_frames = max(int(num_frames), int(length))
    segment_edges = np.linspace(0.0, float(effective_num_frames), num_segments + 1)
    segment_centers = 0.5 * (segment_edges[:-1] + segment_edges[1:])
    start_indices = np.rint(segment_centers - (length / 2.0)).astype(np.int32)
    return np.clip(start_indices, 0, effective_num_frames - length)


@torch.no_grad()
def build_trajectory_stacks(
    flow: np.ndarray,
    length: int,
    num_segments: int,
    device: torch.device | str = "cpu",
    compute_dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Trace dense trajectories for all segments at once.

    Args:
        flow: ``[T, H, W, 2]`` forward flow (x, y) in pixels.
    Returns:
        float32 array ``[num_segments, length, H, W, 2]`` of per-step displacements.
    """
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow with shape [T, H, W, 2], got {flow.shape}")

    flow, _ = pad_flow_to_length(flow, length)
    num_frames, height, width, _ = flow.shape
    start_indices = compute_segment_start_indices(num_frames, length, num_segments).astype(np.int64)
    k = int(start_indices.shape[0])

    flow_t = torch.from_numpy(flow).to(device=device, dtype=compute_dtype).permute(0, 3, 1, 2).contiguous()
    start_t = torch.from_numpy(start_indices).to(device=device)

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=compute_dtype),
        torch.arange(width, device=device, dtype=compute_dtype),
        indexing="ij",
    )
    pos_x = xx.unsqueeze(0).expand(k, height, width).contiguous()
    pos_y = yy.unsqueeze(0).expand(k, height, width).contiguous()

    inv_w = 2.0 / max(width - 1, 1)
    inv_h = 2.0 / max(height - 1, 1)

    outputs = torch.empty((k, length, height, width, 2), device=device, dtype=compute_dtype)
    for step in range(length):
        flow_step = flow_t.index_select(0, start_t + step)  # [K, 2, H, W]
        grid = torch.stack([pos_x * inv_w - 1.0, pos_y * inv_h - 1.0], dim=-1)  # [K, H, W, 2]
        sampled = F.grid_sample(flow_step, grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.permute(0, 2, 3, 1)  # [K, H, W, 2]
        outputs[:, step] = sampled
        if step < length - 1:
            pos_x = pos_x + sampled[..., 0]
            pos_y = pos_y + sampled[..., 1]

    return outputs.to(dtype=torch.float32).cpu().numpy()


def load_trajectory_stack(path) -> np.ndarray:
    """Load a ``[K, L, H, W, 2]`` stack saved as ``.npy`` (memory-mapped) or ``.npz`` (key ``trajectory``)."""
    path = str(path)
    if path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
    elif path.endswith(".npz"):
        with np.load(path) as obj:
            arr = obj["trajectory"]
    else:
        raise ValueError(f"Unsupported trajectory file: {path}")
    if arr.ndim != 5 or arr.shape[-1] != 2:
        raise ValueError(f"Expected [K,L,H,W,2], got {arr.shape} in {path}")
    return arr
