"""Global FINCH clustering of filtered trajectories and medoid selection.

Reproduces the global / ``partition`` / memmap code path of the research script:
FINCH (pip package ``finch-clust``) with cosine distance, a FAISS IVF-Flat
precomputed first-neighbour rank (``initial_rank``) and FAISS for the 1-NN
searches of the higher levels; one output directory per requested partition level.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


def _faiss_num_gpus() -> int:
    try:
        import faiss
        return int(faiss.get_num_gpus())
    except Exception:
        return 0


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)).astype(np.float32, copy=False)


def build_initial_rank_faiss(
    data: np.ndarray,
    nlist: int = 4096,
    nprobe: int = 32,
    train_size: int = 1_000_000,
    batch_size: int = 250_000,
) -> np.ndarray:
    """Self-excluded approximate cosine 1-NN of every row (FINCH ``initial_rank``), FAISS IVF-Flat."""
    import faiss

    x = l2_normalize(np.ascontiguousarray(np.asarray(data, dtype=np.float32)))
    d = x.shape[1]
    index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, int(nlist), faiss.METRIC_INNER_PRODUCT)
    train_n = min(int(train_size), x.shape[0])
    if train_n == x.shape[0]:
        train_x = x
    else:  # evenly spaced rows, as in the original
        train_x = np.asarray(x[np.linspace(0, x.shape[0] - 1, num=train_n, dtype=np.int64)], dtype=np.float32)
    index.train(train_x)
    index.nprobe = int(nprobe)

    n = x.shape[0]
    for start in range(0, n, batch_size):
        index.add(np.asarray(x[start:start + batch_size], dtype=np.float32))

    if _faiss_num_gpus() > 0:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.shard = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
        if hasattr(index, "nprobe"):
            index.nprobe = int(nprobe)

    initial_rank = np.empty((n,), dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        _, nn2 = index.search(np.asarray(x[start:end], dtype=np.float32), 2)
        rows = np.arange(start, end, dtype=np.int64)
        nn = np.where(nn2[:, 0] == rows, nn2[:, 1], nn2[:, 0]).astype(np.int64)
        nn[nn < 0] = rows[nn < 0]
        initial_rank[start:end] = nn
    return initial_rank


def run_finch_partitions(
    features: np.ndarray,
    partition_levels: list[int],
    faiss_nlist: int = 4096,
    faiss_nprobe: int = 32,
    faiss_train_size: int = 1_000_000,
    verbose: bool = True,
) -> tuple[list[tuple[int, np.ndarray, int]], dict]:
    """Run FINCH (cosine) once and return ``[(level, labels, num_clusters), ...]`` for the requested levels.

    FAISS builds ``initial_rank`` and FINCH is forced onto its FAISS 1-NN branch
    for every level (``ann_threshold=1``, ``faiss_threshold=1``), as in the paper runs.
    """
    from finch import FINCH

    if not faiss_available():
        raise RuntimeError("FAISS is required (pip install faiss-cpu or faiss-gpu).")
    data = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
    initial_rank = build_initial_rank_faiss(data, faiss_nlist, faiss_nprobe, faiss_train_size)
    num_gpus = _faiss_num_gpus()
    c, num_clust, _ = FINCH(
        data,
        initial_rank=initial_rank,
        req_clust=None,
        distance="cosine",
        ann_threshold=1,
        ensure_early_exit=True,
        verbose=verbose,
        faiss_threshold=1,
        faiss_use_gpu=num_gpus > 0,
    )
    info = {"finch_distance": "cosine", "faiss_index_type": "ivf_flat", "faiss_nlist": int(faiss_nlist),
            "faiss_nprobe": int(faiss_nprobe), "faiss_train_size": int(faiss_train_size),
            "faiss_num_gpus": num_gpus}
    info["finch_num_levels"] = int(c.shape[1])
    info["finch_partition_sizes"] = [int(v) for v in num_clust]
    outputs, seen = [], set()
    for level in partition_levels:
        level = min(max(int(level), 0), c.shape[1] - 1)  # clamp like the original
        if level in seen:
            continue
        seen.add(level)
        outputs.append((level, np.asarray(c[:, level], dtype=np.int32), int(num_clust[level])))
    info["requested_partition_nums"] = [o[0] for o in outputs]
    return outputs, info


def compute_medoid_indices(data: np.ndarray, labels: np.ndarray, num_clusters: int) -> np.ndarray:
    """Per cluster, the member closest (squared L2) to the cluster mean.

    NOTE: this is the research-code definition; the paper text describes the member
    with the smallest average distance to all members.
    """
    counts = np.bincount(labels, minlength=num_clusters)
    if np.any(counts == 0):
        raise RuntimeError(f"Cluster {int(np.flatnonzero(counts == 0)[0])} is empty.")
    order = np.argsort(labels, kind="mergesort")
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    medoids = np.empty((num_clusters,), dtype=np.int32)
    for k in range(num_clusters):
        members_idx = order[offsets[k]:offsets[k + 1]]
        members = np.asarray(data[members_idx], dtype=np.float32)
        sq = ((members - members.mean(axis=0, keepdims=True)) ** 2).sum(axis=1)
        medoids[k] = int(members_idx[int(np.argmin(sq))])
    return medoids


def save_global_results(
    save_dir: Path,
    input_paths: list[Path],
    sample_indices: np.ndarray,
    kept_indices: np.ndarray,
    start_frames: np.ndarray,
    coords: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    trajectory_length: int,
    summary: dict,
) -> None:
    """Write one partition level in the research output format."""
    save_dir.mkdir(parents=True, exist_ok=True)
    # Re-index cluster ids to 0..M-1 in sorted order (LabelEncoder in the original).
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int32)
    num_clusters = int(labels.max()) + 1

    medoid_local = compute_medoid_indices(features, labels, num_clusters)
    medoid_features = np.asarray(features[medoid_local], dtype=np.float32)
    medoid_coords = np.asarray(coords[medoid_local], dtype=np.int32)
    medoid_start = np.asarray(start_frames[medoid_local], dtype=np.int32)
    medoid_original = np.asarray(kept_indices[medoid_local], dtype=np.int32)
    medoid_samples = np.asarray(sample_indices[medoid_local], dtype=np.int64)
    # flatten features are the raw [L, 2] displacements, so the medoid trajectory is a reshape
    medoid_trajectories = medoid_features.reshape(num_clusters, trajectory_length, 2)

    np.save(save_dir / "cluster_labels.npy", labels)
    np.save(save_dir / "selected_sample_indices.npy", np.asarray(sample_indices, dtype=np.int32))
    np.save(save_dir / "selected_trajectory_indices.npy", np.asarray(kept_indices, dtype=np.int32))
    np.save(save_dir / "selected_start_frames.npy", np.asarray(start_frames, dtype=np.int32))
    np.save(save_dir / "medoid_local_indices.npy", medoid_local.astype(np.int32))
    np.save(save_dir / "medoid_original_indices.npy", medoid_original)
    np.save(save_dir / "medoid_features.npy", medoid_features)
    np.save(save_dir / "medoid_trajectories.npy", medoid_trajectories)
    np.save(save_dir / "medoid_coords.npy", medoid_coords)
    np.save(save_dir / "medoid_start_frames.npy", medoid_start)
    with open(save_dir / "sample_index_to_id.json", "w", encoding="utf-8") as f:
        json.dump({str(i): p.stem for i, p in enumerate(input_paths)}, f, indent=2)

    records = [
        {
            "cluster_id": k,
            "sample_id": input_paths[int(medoid_samples[k])].stem,
            "global_trajectory_index": int(medoid_local[k]),
            "original_trajectory_index": int(medoid_original[k]),
            "segment_id": int(medoid_coords[k, 0]),
            "start_frame": int(medoid_start[k]),
            "y": int(medoid_coords[k, 1]),
            "x": int(medoid_coords[k, 2]),
        }
        for k in range(num_clusters)
    ]
    with open(save_dir / "medoid_metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    summary = dict(summary)
    summary.update({
        "num_input_samples": len(input_paths),
        "num_selected_trajectories": int(len(labels)),
        "num_medoids": num_clusters,
        "cluster_counts": {str(k): int(v) for k, v in sorted(Counter(labels.tolist()).items())},
    })
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def timestamp() -> str:
    return datetime.now().strftime("%m-%d_%H-%M-%S")
