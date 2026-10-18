"""Grad-CAM for single concepts (visualization only).

Grad-CAM shows *where* in the clip a concept activation comes from. It is not
used to compute contributions or interventions; those are defined on the
concept activations themselves (``TraCEAR.explain`` / ``TraCEVQA.explain``).

* AR (VideoMAE-B/16): backpropagates the standardized activation ``z_m`` of
  concept m. Default layer ``blocks[9].norm1`` (paper App. D.1 names
  ``blocks[9].norm``; earlier research figures also used ``blocks[-1].norm1``).
* VQA (InternVideo2-1B): backpropagates the raw video-side activation
  ``z_m = Wv_m x``. Default layer ``vision_encoder.blocks[38].norm1``.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    """Resolve paths like ``blocks[9].norm1``."""
    cur = root
    for part in path.split("."):
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(\[-?\d+\])?", part)
        if m is None:
            raise ValueError(f"unsupported module path segment {part!r}")
        cur = getattr(cur, m.group(1))
        if m.group(2):
            cur = cur[int(m.group(2)[1:-1])]
    return cur


class _Hook:
    def __init__(self, module: nn.Module):
        self.act = self.grad = None
        self.h1 = module.register_forward_hook(lambda m, i, o: setattr(self, "act", o.detach()))
        self.h2 = module.register_full_backward_hook(lambda m, gi, go: setattr(self, "grad", go[0].detach()))

    def remove(self):
        self.h1.remove()
        self.h2.remove()


def _tokens_to_3d(tokens: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
    B, N, C = tokens.shape
    if N == T * H * W + 1:          # drop CLS token (InternVideo2)
        tokens = tokens[:, 1:]
    elif N != T * H * W:
        raise ValueError(f"unexpected token count {N} for grid {T}x{H}x{W}")
    return tokens.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)


def _cam(act, grad, grid, out_size) -> torch.Tensor:
    a, g = _tokens_to_3d(act.float(), *grid), _tokens_to_3d(grad.float(), *grid)
    cam = (g.mean(dim=(2, 3, 4), keepdim=True) * a).sum(1, keepdim=True)
    cam = F.interpolate(F.relu(cam), size=out_size, mode="trilinear", align_corners=False)[0, 0]
    return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).cpu()


def videomae_concept_cam(backbone: nn.Module, ar_model, clip: torch.Tensor, concept_id: int,
                         target_layer: str = "blocks[9].norm1") -> torch.Tensor:
    """clip [3, 16, 224, 224] (normalized) -> CAM [16, 224, 224] in [0, 1]."""
    dev = next(backbone.parameters()).device
    hook = _Hook(resolve_module(backbone, target_layer))
    try:
        backbone.zero_grad(set_to_none=True)
        x = clip.unsqueeze(0).to(dev).requires_grad_(True)
        z = ar_model.to(dev).concept_activations(backbone.forward_features(x))
        z[0, concept_id].backward()
        T, S = clip.shape[1], clip.shape[2]
        p = backbone.patch_embed.patch_size[0]
        return _cam(hook.act, hook.grad, (T // backbone.patch_embed.tubelet_size, S // p, S // p), (T, S, S))
    finally:
        hook.remove()


def iv2_concept_cam(iv2, vqa_model, frames: torch.Tensor, concept_id: int,
                    target_layer: str = "vision_encoder.blocks[38].norm1") -> torch.Tensor:
    """frames [1, 4, 3, 224, 224] (normalized) -> CAM [4, 224, 224] in [0, 1]."""
    for blk in iv2.vision_encoder.blocks:   # activation checkpointing would fire hooks twice
        blk.with_cp = False
    hook = _Hook(resolve_module(iv2, target_layer))
    try:
        iv2.zero_grad(set_to_none=True)
        x = frames.to(iv2.device, iv2.dtype).requires_grad_(True)
        z = vqa_model.to(iv2.device).video_activations(iv2.vision_features_with_grad(x))
        z[0, concept_id].backward()
        T, S = frames.shape[1], frames.shape[-1]
        return _cam(hook.act, hook.grad, (T, S // 14, S // 14), (T, S, S))
    finally:
        hook.remove()


def overlay(frame_rgb_uint8, cam_frame, alpha: float = 0.45):
    """Blend a [H, W] CAM in [0, 1] onto an RGB uint8 frame (same color map as the research figures)."""
    import numpy as np
    c = np.asarray(cam_frame, dtype=np.float32)
    heat = np.stack([np.clip(1.5 * c, 0, 1), np.clip(1.5 * (1 - np.abs(c - 0.5) * 2), 0, 1),
                     np.clip(1.5 * (1 - c), 0, 1)], -1)
    out = frame_rgb_uint8.astype(np.float32) / 255.0 * (1 - alpha) + heat * alpha
    return np.clip(out * 255, 0, 255).astype(np.uint8)
