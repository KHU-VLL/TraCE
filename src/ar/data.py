"""Annotation and feature-cache I/O for action recognition.

Annotation csv: one ``<relative video path>,<class index>`` row per video, no
header. ``class_list.txt`` lists class names, one per line, in index order.

Feature cache: ``torch.save({"videos": [str], "features": Tensor[N, D]})``.
Rows follow the annotation csv order. A bare ``Tensor[N, D]`` (the format of
the original research cache) is also accepted and is then assumed to follow
the csv order.
"""
from __future__ import annotations

import csv
from typing import List, Tuple

import torch


def read_annotation(csv_path: str) -> Tuple[List[str], List[int]]:
    videos, labels = [], []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            videos.append(row[0])
            labels.append(int(row[1]))
    return videos, labels


def read_class_list(path: str) -> List[str]:
    with open(path) as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def load_features(path: str, videos: List[str] | None = None) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, torch.Tensor):
        feats = obj
        if videos is not None and feats.shape[0] != len(videos):
            raise ValueError(f"{path}: {feats.shape[0]} rows but annotation has {len(videos)} videos")
        return feats.float()
    feats, names = obj["features"].float(), list(obj["videos"])
    if videos is None or names == list(videos):
        return feats
    pos = {v: i for i, v in enumerate(names)}
    missing = [v for v in videos if v not in pos]
    if missing:
        raise KeyError(f"{path}: {len(missing)} annotated videos have no feature, e.g. {missing[:3]}")
    return feats[[pos[v] for v in videos]]


def save_features(path: str, videos: List[str], feats: torch.Tensor) -> None:
    torch.save({"videos": list(videos), "features": feats.cpu()}, path)
