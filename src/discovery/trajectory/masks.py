"""Foreground (saliency) mask I/O.

Layout: ``<mask_root>/<sample_id>/frame_XXXXXX.{npy,npz}`` with one binary mask per
*video frame* (index 0 = first frame). A nested layout
``<mask_root>/<subdir>/<sample_id>/`` is also accepted (UCF-101 class folders,
TOMATO ``human/object/simulated``).

``.npy``: a 2D uint8 0/1 array. ``.npz``: either a bit-packed mask
``{"packed": np.packbits(mask.ravel()), "shape": (H, W)}`` or a single array.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MASK_SUFFIXES = (".npy", ".npz")
_SAMPLE_DIR_CACHE: dict[tuple[str, str], Path] = {}


def resolve_mask_sample_dir(mask_root: Path, sample_id: str) -> Path:
    key = (str(mask_root), sample_id)
    if key in _SAMPLE_DIR_CACHE:
        return _SAMPLE_DIR_CACHE[key]
    direct = mask_root / sample_id
    if direct.is_dir():
        _SAMPLE_DIR_CACHE[key] = direct
        return direct
    existing = [sub / sample_id for sub in sorted(mask_root.iterdir()) if sub.is_dir() and (sub / sample_id).is_dir()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one mask directory for {sample_id} under {mask_root} "
            f"(directly or one level below), found {len(existing)}"
        )
    _SAMPLE_DIR_CACHE[key] = existing[0]
    return existing[0]


def count_mask_frames(mask_root: Path, sample_id: str) -> int:
    """Number of video frames that have a mask (distinct ``frame_*`` stems)."""
    sample_dir = resolve_mask_sample_dir(mask_root, sample_id)
    stems = {p.stem for p in sample_dir.iterdir() if p.name.startswith("frame_") and p.suffix in MASK_SUFFIXES}
    if not stems:
        raise FileNotFoundError(f"No frame_* masks in {sample_dir}")
    return len(stems)


def read_mask_file(mask_path: Path) -> np.ndarray:
    if mask_path.suffix != ".npz":
        return np.asarray(np.load(mask_path))
    with np.load(mask_path) as obj:
        if "packed" in obj:
            shape = tuple(int(v) for v in obj["shape"])
            bits = np.unpackbits(np.asarray(obj["packed"], dtype=np.uint8))
            return bits[: int(np.prod(shape))].reshape(shape)
        keys = list(obj.keys())
        if len(keys) != 1:
            raise KeyError(f"Expected 'packed' or a single array in {mask_path}, got {keys}")
        return np.asarray(obj[keys[0]])


def save_mask_packed(mask: np.ndarray, path: Path) -> None:
    """Save a binary mask in the bit-packed ``.npz`` format."""
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    np.savez_compressed(path, packed=np.packbits(mask.reshape(-1)), shape=np.asarray(mask.shape, dtype=np.int64))


def resize_mask_nearest(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h_in, w_in = mask.shape
    h_out, w_out = target_hw
    if (h_in, w_in) == (h_out, w_out):
        return mask
    ys = np.minimum((np.arange(h_out) * h_in / h_out).astype(np.int64), h_in - 1)
    xs = np.minimum((np.arange(w_out) * w_in / w_out).astype(np.int64), w_in - 1)
    return mask[ys[:, None], xs[None, :]]


def load_mask(mask_root: Path, sample_id: str, frame_idx: int, expected_hw: tuple[int, int]) -> np.ndarray:
    """Mask of video frame ``frame_idx``, nearest-resized to the trajectory grid ``expected_hw``."""
    sample_dir = resolve_mask_sample_dir(mask_root, sample_id)
    stem = f"frame_{frame_idx:06d}"
    mask_path = next((sample_dir / f"{stem}{s}" for s in MASK_SUFFIXES if (sample_dir / f"{stem}{s}").exists()), None)
    if mask_path is None:
        raise FileNotFoundError(f"Mask not found: {sample_dir / stem}{{.npy,.npz}}")
    mask = np.squeeze(read_mask_file(mask_path))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask at {mask_path}, got {mask.shape}")
    return resize_mask_nearest(mask, expected_hw)
