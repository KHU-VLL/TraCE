"""Release bundle of a trajectory-concept vocabulary (for visualisation).

``trajectory_concepts.npz``
    medoid_trajectories  float32 [Mt, L, 2]  per-step displacements (x, y) in flow-grid pixels
    cluster_ids          int32   [Mt]        trajectory concept k = cluster k (label column k)
    medoid_start_frames  int32   [Mt]        first flow index of the medoid's segment
    medoid_coords        int32   [Mt, 3]     (segment_id, y, x) of the seed pixel on the flow grid
``trajectory_concepts.json``
    dataset, num_concepts (Mt), trajectory_length (L), num_segments (Ns), partition_level,
    concept_order, and ``medoids``: one record per concept with sample_id, segment_id,
    start_frame, y, x, grid_hw (flow-grid H, W of the source video) and num_flow_frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_bundle(cluster_dir: Path, out_dir: Path, dataset: str) -> dict:
    cluster_dir, out_dir = Path(cluster_dir), Path(out_dir)
    trajectories = np.load(cluster_dir / "medoid_trajectories.npy").astype(np.float32)
    with open(cluster_dir / "medoid_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(cluster_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    num_concepts, length, _ = trajectories.shape
    if [m["cluster_id"] for m in metadata] != list(range(num_concepts)):
        raise RuntimeError("medoid_metadata.json is not ordered by cluster id")
    extra = summary.get("extra_info", {})
    level = summary.get("selected_partition", extra.get("selected_partition"))
    per_sample = summary.get("per_sample_filtering", extra.get("per_sample_filtering", {}))
    raw_shapes = summary.get("raw_shapes_by_sample", {})

    records = []
    for m in metadata:
        sid = m["sample_id"]
        shape = raw_shapes.get(sid)
        records.append({
            "concept": int(m["cluster_id"]),
            "sample_id": sid,
            "segment_id": int(m["segment_id"]),
            "start_frame": int(m["start_frame"]),
            "y": int(m["y"]),
            "x": int(m["x"]),
            "grid_hw": [int(shape[2]), int(shape[3])] if shape else None,
            "num_flow_frames": per_sample.get(sid, {}).get("num_flow_frames"),
        })
    num_segments = {int(s[0]) for s in raw_shapes.values()}
    info = {
        "dataset": dataset,
        "source_cluster_dir": cluster_dir.name,
        "num_concepts": int(num_concepts),
        "trajectory_length": int(length),
        "num_segments": num_segments.pop() if len(num_segments) == 1 else sorted(num_segments),
        "partition_level": None if level is None else int(level),
        "num_clustered_videos": int(summary.get("num_input_samples", len(raw_shapes))),
        "num_clustered_trajectories": int(summary.get("num_selected_trajectories", 0)),
        "concept_order": "trajectory concept k = FINCH cluster k = column k of the trajectory soft labels",
        "trajectory_format": "medoid_trajectories[k, r] = (dx, dy) displacement of step r in flow-grid pixels",
        "medoids": records,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "trajectory_concepts.npz",
        medoid_trajectories=trajectories,
        cluster_ids=np.arange(num_concepts, dtype=np.int32),
        medoid_start_frames=np.asarray([r["start_frame"] for r in records], dtype=np.int32),
        medoid_coords=np.asarray([[r["segment_id"], r["y"], r["x"]] for r in records], dtype=np.int32),
    )
    with open(out_dir / "trajectory_concepts.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def load_bundle(bundle_dir: Path) -> tuple[np.ndarray, dict]:
    """Return ``(medoid_trajectories [Mt, L, 2], info dict)``."""
    bundle_dir = Path(bundle_dir)
    with np.load(bundle_dir / "trajectory_concepts.npz") as z:
        trajectories = z["medoid_trajectories"]
    with open(bundle_dir / "trajectory_concepts.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    return trajectories, info
