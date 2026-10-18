"""Soft trajectory-concept labels.

Train videos: ``c_i = h_i / ||h_i||_1`` where ``h_i[k]`` counts the filtered
trajectories of video i that FINCH put in cluster k (Eq. 18-19), read directly
from a clustering output directory.

Val videos (never clustered): every filtered trajectory is assigned to the
nearest medoid feature (squared L2), then the same normalised histogram is taken.

Label file format (``train.pkl`` / ``val.pkl``; same content as the research
``hard_label_{train,val}.pkl`` files, which hold soft labels despite their name): a pickle list of
``{"video_name": str, "attribute_label": list[float] of length Mt}``.
Videos without any trajectory information get ``[-1] * Mt``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def read_split_ids(path: Path) -> list[str]:
    """Video names of a split, in file order.

    CSV/text: first column (comma or whitespace separated), e.g. ``ApplyEyeMakeup/v_..._c01.avi``.
    JSON (VQA splits): one name per entry (duplicates kept, one row per question):
    ``entry["video"]`` (STAR) or ``entry["key"] + ".mp4"`` (TOMATO).
    """
    path = Path(path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.values() if isinstance(data, dict) else data
        return [str(e["video"]) if "video" in e else f"{e['key']}.mp4" for e in entries]
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line.replace(",", " ").split()[0])
    return names


def load_cluster_assignments(cluster_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(cluster_labels, per-trajectory sample index, sample ids)`` of a clustering run.

    Supports the current layout (``selected_sample_indices.npy`` + ``sample_index_to_id.json``)
    and the older in-memory layout that stored ``selected_sample_ids.npy`` (KTH run).
    """
    labels = np.load(cluster_dir / "cluster_labels.npy")
    if (cluster_dir / "selected_sample_indices.npy").exists():
        sample_idx = np.load(cluster_dir / "selected_sample_indices.npy")
        with open(cluster_dir / "sample_index_to_id.json", "r", encoding="utf-8") as f:
            mapping = json.load(f)
        sample_ids = [mapping[str(i)] for i in range(len(mapping))]
    else:
        ids = np.load(cluster_dir / "selected_sample_ids.npy")
        sample_ids, sample_idx = np.unique(ids, return_inverse=True)
        sample_ids = [str(s) for s in sample_ids]
    if len(labels) != len(sample_idx):
        raise RuntimeError("cluster_labels and sample indices have different lengths")
    return labels.astype(np.int64), np.asarray(sample_idx, dtype=np.int64), sample_ids


def cluster_histograms(labels: np.ndarray, sample_idx: np.ndarray, num_samples: int, num_clusters: int) -> np.ndarray:
    """``[num_samples, num_clusters]`` trajectory counts h_{i,k}."""
    flat = sample_idx * num_clusters + labels
    return np.bincount(flat, minlength=num_samples * num_clusters).reshape(num_samples, num_clusters)


def normalize_rows(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    return counts / counts.sum(axis=1, keepdims=True)


def train_soft_labels(cluster_dir: Path, video_names: list[str]) -> list[dict]:
    """Soft labels for the given train videos (``video_names`` = split CSV first column, in order)."""
    labels, sample_idx, sample_ids = load_cluster_assignments(cluster_dir)
    num_clusters = int(labels.max()) + 1
    hist = cluster_histograms(labels, sample_idx, len(sample_ids), num_clusters)
    row_of = {sid: i for i, sid in enumerate(sample_ids)}
    out = []
    for name in video_names:
        i = row_of.get(Path(name).stem)
        if i is None or hist[i].sum() == 0:
            attr = [-1] * num_clusters
        else:
            attr = (hist[i] / hist[i].sum()).tolist()
        out.append({"video_name": name, "attribute_label": attr})
    return out


def nearest_medoid_soft_label(features: np.ndarray, medoid_features: np.ndarray) -> list[float]:
    """Histogram of nearest-medoid (squared L2) assignments, divided by the trajectory count."""
    num_clusters = medoid_features.shape[0]
    attr = [0.0] * num_clusters
    if features.shape[0] == 0:
        return attr
    diff = features[:, None, :].astype(np.float32) - medoid_features[None, :, :].astype(np.float32)
    nearest = np.argmin(np.einsum("nkd,nkd->nk", diff, diff), axis=1)
    counts = np.bincount(nearest, minlength=num_clusters)
    total = float(len(nearest))
    for k in np.flatnonzero(counts):
        attr[int(k)] = int(counts[k]) / total
    return attr


def save_label_file(entries: list[dict], out_dir: Path, split: str) -> Path:
    """Write ``<out_dir>/<split>.pkl`` (the file read by tools/train_ar.py / train_vqa.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"{split}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(entries, f)
    return pkl_path


def load_label_file(path: Path) -> tuple[list[str], np.ndarray]:
    """Load a label pickle as ``(video_names, [N, Mt] float array)``."""
    with open(path, "rb") as f:
        entries = pickle.load(f)
    names = [e["video_name"] for e in entries]
    return names, np.asarray([e["attribute_label"] for e in entries], dtype=np.float64)
