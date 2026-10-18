"""VideoMAE-B/16 feature extractor for action recognition.

The feature is ``forward_features`` of the fine-tuned VideoMAE ViT-B/16:
mean-pooled tokens followed by ``fc_norm`` (768-d). Clips have 16 frames at
224x224. Preprocessing reproduces the validation loaders used to build the
released feature caches:

* resize the shorter side to 224 (OpenCV bilinear), center crop 224x224,
  scale to [0, 1], ImageNet mean/std normalization;
* frame sampling ``"segment"`` (NTU RGB+D Mutual, Chiral SSv2, KTH): split the
  video into 16 equal segments and take the middle frame of each;
* frame sampling ``"dense"`` (UCF-101): 16 frames with stride 4 from a window
  placed at the middle of 5 evenly spaced candidate end points.

The model code is vendored from VideoMAE (see third_party/videomae/NOTICE).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional

import cv2
import numpy as np
import torch

from ..third_party.videomae.modeling_finetune import vmae_vit_base_patch16_224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_videomae(ckpt_path: str, device: str = "cuda", num_frames: int = 16) -> torch.nn.Module:
    """Load a fine-tuned VideoMAE-B/16 checkpoint (keys under 'model' or 'module')."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = None
    for key in ("model", "module"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    if state is None:
        state = ckpt
    clean = OrderedDict()
    for k, v in state.items():
        if k.startswith("backbone."):
            k = k[9:]
        elif k.startswith("encoder."):
            k = k[8:]
        clean[k] = v
    num_classes = clean["head.weight"].shape[0] if "head.weight" in clean else 0
    model = vmae_vit_base_patch16_224(num_classes=num_classes, all_frames=num_frames, tubelet_size=2,
                                      use_mean_pooling=True, init_scale=0.001)
    msg = model.load_state_dict(clean, strict=False)
    bad = [k for k in msg.missing_keys if not k.startswith("head.")]
    if bad:
        raise RuntimeError(f"VideoMAE checkpoint is missing weights, e.g. {bad[:5]}")
    model.load_report = {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}
    return model.to(device).eval()


# ------------------------------------------------------------- frame sampling
def segment_indices(n: int, num_frames: int = 16) -> List[int]:
    """Middle frame of each of ``num_frames`` equal segments (frame 0 for videos shorter than that)."""
    avg = n // num_frames
    if avg > 0:
        return [int(i * avg + avg // 2) for i in range(num_frames)]
    return [0] * num_frames


def dense_indices(n: int, clip_len: int = 16, rate: int = 4, end_point: int = 2) -> List[int]:
    """Dense sampling of the UCF-101 loader (``num_segment=1``)."""
    converted = clip_len * rate
    seg_len = n
    if seg_len <= converted:
        idx = np.linspace(0, seg_len, num=seg_len // rate)
        idx = np.concatenate((idx, np.ones(clip_len - seg_len // rate) * seg_len))
        idx = np.clip(idx, 0, seg_len - 1).astype(np.int64)
    else:
        points = np.linspace(converted, seg_len, 5, endpoint=True)
        end_idx = int(points[end_point])
        str_idx = end_idx - converted
        idx = np.linspace(str_idx, end_idx, num=clip_len)
        idx = np.clip(idx, str_idx, end_idx - 1).astype(np.int64)
    return idx.tolist()


def read_clip(path: str, sampling: str = "segment", num_frames: int = 16, rate: int = 4) -> np.ndarray:
    """Decode the sampled frames (RGB uint8, [T, H, W, 3]) with decord."""
    from decord import VideoReader, cpu
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    n = len(vr)
    if sampling == "segment":
        idx = segment_indices(n, num_frames)
    elif sampling == "dense":
        idx = dense_indices(n, num_frames, rate)
    else:
        raise ValueError(f"unknown sampling {sampling!r}")
    vr.seek(0)
    return vr.get_batch(idx).asnumpy()


def preprocess(frames: np.ndarray, size: int = 224) -> torch.Tensor:
    """[T, H, W, 3] uint8 RGB -> normalized [3, T, size, size] float tensor."""
    h, w = frames.shape[1:3]
    if not ((w <= h and w == size) or (h <= w and h == size)):
        if w < h:
            nw, nh = size, int(size * h / w)
        else:
            nh, nw = size, int(size * w / h)
        frames = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR) for f in frames])
    h, w = frames.shape[1:3]
    y1, x1 = int(round((h - size) / 2.0)), int(round((w - size) / 2.0))
    frames = frames[:, y1:y1 + size, x1:x1 + size]
    x = torch.from_numpy(frames.transpose(3, 0, 1, 2).astype(np.float64)).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None, None]
    return (x - mean) / std


def load_clip(path: str, sampling: str = "segment") -> torch.Tensor:
    return preprocess(read_clip(path, sampling))


@torch.no_grad()
def encode(model: torch.nn.Module, clips: torch.Tensor) -> torch.Tensor:
    """clips [B, 3, 16, 224, 224] -> features [B, 768]."""
    dev = next(model.parameters()).device
    return model.forward_features(clips.to(dev)).float().cpu()


def iv2_frames_from_clip(clip: torch.Tensor, num_frames: int = 4) -> torch.Tensor:
    """InternVideo2 input built from a preprocessed VideoMAE clip, as used for the
    AR object/scene pseudo-labels: ``linspace(0, T-1, 4)`` frames of the same
    16-frame clip. [3, T, H, W] -> [1, 4, 3, H, W]."""
    idx = torch.linspace(0, clip.shape[1] - 1, num_frames).long()
    return clip[:, idx].permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def find_backbone(checkpoint_file: str, root: str) -> str:
    """Locate a VideoMAE checkpoint by file name: our fine-tuned backbones are in
    ``checkpoints/backbones/``, official VideoMAE weights in ``pretrained/``."""
    import os
    for d in ("checkpoints/backbones", "pretrained"):
        p = os.path.join(root, d, checkpoint_file)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{checkpoint_file} not found in {root}/checkpoints/backbones or {root}/pretrained")
