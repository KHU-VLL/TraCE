"""Feature-cache I/O for multiple-choice VQA.

A split directory contains::

    video_features.pt   {"videos": [str], "features": Tensor[N_video, D]}
    text_features.pt    {"texts":  [str], "features": Tensor[N_text, D]}
    sample_index.json   {"samples": [{"video", "video_idx", "option_text_idxs", "answer_idx"}, ...]}

Each sample stores its candidate answers as indices into ``text_features``
in the original candidate order; ``answer_idx`` indexes that list.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class MCQASplit:
    videos: List[str]            # per sample
    video_feats: torch.Tensor    # [N, D] per sample (videos repeated if several questions)
    option_feats: torch.Tensor   # [N, J, D]
    option_mask: torch.Tensor    # [N, J] bool (False = padding)
    options: List[List[str]]     # per sample candidate strings
    answers: torch.Tensor        # [N]

    def __len__(self):
        return len(self.videos)


def load_split(split_dir: str) -> MCQASplit:
    v = torch.load(os.path.join(split_dir, "video_features.pt"), map_location="cpu", weights_only=True)
    t = torch.load(os.path.join(split_dir, "text_features.pt"), map_location="cpu", weights_only=True)
    with open(os.path.join(split_dir, "sample_index.json")) as f:
        samples = json.load(f)["samples"]
    vf, tf = v["features"].float(), t["features"].float()
    J = max(len(s["option_text_idxs"]) for s in samples)
    N, D = len(samples), tf.shape[1]
    opt = torch.zeros(N, J, D)
    mask = torch.zeros(N, J, dtype=torch.bool)
    for i, s in enumerate(samples):
        idx = s["option_text_idxs"]
        opt[i, :len(idx)] = tf[idx]
        mask[i, :len(idx)] = True
    return MCQASplit(
        videos=[s["video"] for s in samples],
        video_feats=vf[[s["video_idx"] for s in samples]],
        option_feats=opt, option_mask=mask,
        options=[[t["texts"][j] for j in s["option_text_idxs"]] for s in samples],
        answers=torch.tensor([s["answer_idx"] for s in samples], dtype=torch.long),
    )
