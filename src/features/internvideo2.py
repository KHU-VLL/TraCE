"""InternVideo2-Stage2-1B (224px, 4 frames) video and text encoders.

Used for (i) the VQA video/text features ``x``/``u`` and (ii) the object/scene
pseudo-labels (video-text cosine with "A video of {concept}").

Both outputs are the L2-normalized 512-d projections of the Stage-2 model:
``vision_proj(pooled vision tokens)`` and ``text_proj(CLS token)``.

Weights (not redistributed here):
  * ``InternVideo2-stage2_1b-224p-f4.pt``  (OpenGVLab, InternVideo2 Stage-2 1B)
  * ``bert-large-uncased`` directory (Hugging Face; tokenizer + config, the
    Stage-2 checkpoint overrides the BERT weights it uses)

The model code is vendored from the OpenGVLab InternVideo2 multi-modality /
MC-QA demo code (see trace_xai/third_party/internvideo2/NOTICE).
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Sequence

import cv2
import numpy as np
import torch
from torch import nn

from ..third_party.internvideo2.bert.tokenization_bert import BertTokenizer
from ..third_party.internvideo2.bert.xbert import BertConfig, BertForMaskedLM
from ..third_party.internvideo2.internvideo2 import pretrain_internvideo2_1b_patch14_224
from ..third_party.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new

_BERT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "third_party", "internvideo2",
                            "config_bert_large.json")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


class InternVideo2Stage2(nn.Module):
    """Inference-only subset of the InternVideo2 Stage-2 model (same parameter names)."""

    def __init__(self, bert_path: str, use_flash_attn: bool = False):
        super().__init__()
        args = SimpleNamespace(use_flash_attn=use_flash_attn, use_fused_rmsnorm=use_flash_attn,
                               use_fused_mlp=use_flash_attn)
        self.vision_encoder = pretrain_internvideo2_1b_patch14_224(args)
        cfg = BertConfig.from_json_file(os.path.abspath(_BERT_CONFIG))
        cfg.encoder_width = 1408
        cfg.gradient_checkpointing = False
        cfg.fusion_layer = 19
        self.text_encoder = BertForMaskedLM.from_pretrained(bert_path, config=cfg, local_files_only=True)
        self.vision_proj = nn.Linear(768, 512)
        self.text_proj = nn.Linear(1024, 512)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def dtype(self):
        return self.vision_encoder.patch_embed.proj.weight.dtype

    @property
    def device(self):
        return self.vision_proj.weight.device

    @torch.no_grad()
    def get_vid_feat(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [B, T, C, H, W] (normalized) -> [B, 512] L2-normalized."""
        image = frames.permute(0, 2, 1, 3, 4).to(self.dtype)
        _, pooled, _, _ = self.vision_encoder(image, None, frames.shape[1] == 1)
        v = self.vision_proj(pooled)
        v = v / v.norm(dim=-1, keepdim=True)
        return v.reshape(v.shape[0], -1).float()

    def vision_features_with_grad(self, frames: torch.Tensor) -> torch.Tensor:
        """Same as get_vid_feat but differentiable w.r.t. intermediate activations (Grad-CAM)."""
        image = frames.permute(0, 2, 1, 3, 4).to(self.dtype)
        _, pooled, _, _ = self.vision_encoder(image, None, frames.shape[1] == 1)
        v = self.vision_proj(pooled)
        return (v / v.norm(dim=-1, keepdim=True)).reshape(v.shape[0], -1).float()

    @torch.no_grad()
    def get_txt_feat(self, texts: Sequence[str] | str) -> torch.Tensor:
        """texts -> [N, 512] L2-normalized (max length 40 tokens, as in the original code)."""
        if isinstance(texts, str):
            texts = [texts]
        tok = self.tokenizer(list(texts), padding="max_length", truncation=True, max_length=40,
                             return_tensors="pt").to(self.device)
        out = self.text_encoder.bert(tok.input_ids, attention_mask=tok.attention_mask,
                                     return_dict=True, mode="text")
        t = self.text_proj(out.last_hidden_state[:, 0])
        return (t / t.norm(dim=-1, keepdim=True)).float()


def load_internvideo2(ckpt_path: str, bert_path: str, device: str = "cuda",
                      dtype: torch.dtype = torch.float32, use_flash_attn: bool = False) -> InternVideo2Stage2:
    """Build the model and load the Stage-2 checkpoint (4-frame positional embedding)."""
    model = InternVideo2Stage2(bert_path, use_flash_attn=use_flash_attn)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("module", ckpt)) if isinstance(ckpt, dict) else ckpt
    interpolate_pos_embed_internvideo2_new(state, model.vision_encoder, 4)
    msg = model.load_state_dict(state, strict=False)
    missing = [k for k in msg.missing_keys if not k.startswith("text_encoder.cls.")]
    needed = ("vision_encoder.patch_embed", "vision_encoder.blocks.0.", "vision_proj", "text_proj",
              "text_encoder.bert.encoder.layer.0.")
    bad = [k for k in missing if k.startswith(needed)]
    if bad:
        raise RuntimeError(f"checkpoint is missing required weights, e.g. {bad[:5]}")
    model.load_report = {"missing": missing, "unexpected": list(msg.unexpected_keys)}
    return model.to(device=device, dtype=dtype).eval()


# ---------------------------------------------------------------- video input
def read_frames_cv2(path: str) -> List[np.ndarray]:
    """All frames of a video as BGR uint8 arrays (OpenCV decoding, as in the original pipeline)."""
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def frames_to_tensor(frames: List[np.ndarray], num_frames: int = 4, size: int = 224) -> torch.Tensor:
    """Uniform-stride sampling used for all VQA features.

    ``step = len(frames) // num_frames`` and the first ``num_frames`` frames of
    ``frames[::step]`` are kept (i.e. the first frame of each segment). Frames
    are converted to RGB and resized to ``size x size`` without cropping, then
    ImageNet-normalized. Returns [1, T, C, H, W].
    """
    if len(frames) < num_frames:
        raise ValueError(f"video has {len(frames)} frames < {num_frames}")
    step = len(frames) // num_frames
    sel = frames[::step][:num_frames]
    sel = [cv2.resize(f[:, :, ::-1], (size, size)) for f in sel]
    tube = np.stack([(f / 255.0 - IMAGENET_MEAN) / IMAGENET_STD for f in sel])[None]  # 1,T,H,W,C
    return torch.from_numpy(tube.transpose(0, 1, 4, 2, 3).copy()).float()


@torch.no_grad()
def encode_video_file(model: InternVideo2Stage2, path: str, num_frames: int = 4) -> torch.Tensor:
    x = frames_to_tensor(read_frames_cv2(path), num_frames).to(model.device, model.dtype)
    return model.get_vid_feat(x)[0]


@torch.no_grad()
def encode_concepts(model: InternVideo2Stage2, words: Sequence[str], prompt: str = "A video of {}",
                    batch_size: int = 1) -> torch.Tensor:
    """Concept text embeddings used for object/scene pseudo-labels, [C, 512].

    The research code encoded one prompt per call; ``batch_size=1`` keeps that.
    """
    feats = [model.get_txt_feat([prompt.format(w) for w in words[i:i + batch_size]])
             for i in range(0, len(words), batch_size)]
    return torch.cat(feats).float()
