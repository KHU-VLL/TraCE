"""Filter trajectories of all training videos and cluster them globally with FINCH.

    python tools/discovery/cluster_trajectories.py --config configs/discovery/trajectory_ntu_mutual.yaml \
        --input_dir data/trajectory/ntu_mutual/stacks --mask_root data/trajectory/ntu_mutual/masks \
        --sample_filter_csv annotations/ntu_mutual/train.csv --output_root outputs/trajectory/ntu_mutual

Per video (``prepare_sample``): foreground gating with the mask of each segment's
start frame, then keep the top (100 - motion_percentile)% trajectories by motion
magnitude among the foreground ones; features = flattened ``[L, 2]`` displacements.
All features are written to a disk-backed memmap, FINCH (cosine, FAISS IVF-Flat
initial rank) is run once, and every requested partition level is saved to
``<output_root>/global_partition_partition{level}_{Mt}clusters_{timestamp}/``.
Outputs: cluster_labels.npy, selected_sample_indices.npy, sample_index_to_id.json, medoid_*.npy, medoid_metadata.json, summary.json.
"""
import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.discovery.trajectory.clustering import (  # noqa: E402
    run_finch_partitions,
    save_global_results,
    timestamp,
)
from src.discovery.trajectory.config import parse_args_with_config  # noqa: E402
from src.discovery.trajectory.filtering import prepare_sample  # noqa: E402
from src.discovery.trajectory.labels import read_split_ids  # noqa: E402


def select_inputs(input_dir: Path, filter_files: list[str] | None, strip_suffixes: list[str] | None) -> list[Path]:
    paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix in (".npy", ".npz"))
    if not paths:
        raise FileNotFoundError(f"No trajectory stacks under {input_dir}")
    if not filter_files:
        return paths
    ids = {Path(name).stem for f in filter_files for name in read_split_ids(Path(f))}

    def norm(stem):
        for suf in strip_suffixes or []:
            if suf and stem.endswith(suf):
                return stem[: -len(suf)]
        return stem

    kept = [p for p in paths if p.stem in ids or norm(p.stem) in ids]
    print(f"[sample-filter] kept={len(kept)} dropped={len(paths) - len(kept)}")
    if not kept:
        raise RuntimeError("The sample filter removed all inputs; check that ids match file stems.")
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", help="trajectory stacks from build_trajectories.py")
    ap.add_argument("--mask_root", help="foreground masks from foreground_masks_u2net.py")
    ap.add_argument("--output_root")
    ap.add_argument("--flow_root", default=None, help="optional raw flow dir used to read T (else #masks - 1)")
    ap.add_argument("--sample_filter_csv", nargs="*", default=None,
                    help="split file(s) (CSV or VQA JSON) listing the videos to cluster; default: all stacks")
    ap.add_argument("--sample_filter_strip_suffix", nargs="*", default=None)
    ap.add_argument("--motion_percentile", type=float, default=95.0, help="keep trajectories >= this percentile (rho = 100 - p)")
    ap.add_argument("--path_length_dtype", choices=("stack", "float32"), default="stack",
                    help="dtype of the step differences in the motion magnitude (see docs)")
    ap.add_argument("--partition_levels", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9],
                    help="FINCH partition levels to save (0 = finest)")
    ap.add_argument("--faiss_nlist", type=int, default=4096)
    ap.add_argument("--faiss_nprobe", type=int, default=32)
    ap.add_argument("--faiss_train_size", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=655)
    ap.add_argument("--workers", type=int, default=1, help="processes for per-video filtering")
    ap.add_argument("--keep_memmap", action="store_true")
    args = parse_args_with_config(ap, "clustering", required=("input_dir", "mask_root", "output_root"))

    np.random.seed(args.seed)
    output_root = Path(args.output_root)
    input_paths = select_inputs(Path(args.input_dir), args.sample_filter_csv, args.sample_filter_strip_suffix)
    prep = partial(
        prepare_sample,
        mask_root=Path(args.mask_root),
        motion_percentile=args.motion_percentile,
        flow_root=Path(args.flow_root) if args.flow_root else None,
        path_length_dtype=args.path_length_dtype,
    )

    # 1) filter every video and append its rows to disk-backed arrays
    ts = timestamp()
    mm_dir = output_root / ".memmap_cache" / f"run_{ts}"
    mm_dir.mkdir(parents=True, exist_ok=True)
    fields = {"features": np.float32, "coords": np.int32, "kept_indices": np.int32,
              "start_frames": np.int32, "sample_indices": np.int32}
    handles = {k: open(mm_dir / f"{k}.dat", "wb") for k in fields}
    total, raw_shapes, per_sample = 0, {}, {}
    feature_dim = None
    pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
    results = pool.map(prep, input_paths, chunksize=1) if pool else map(prep, input_paths)
    try:
        for sample_idx, p in enumerate(results):
            feature_dim = p.raw_shape[1] * 2
            n = len(p.kept_indices)
            if n:
                np.ascontiguousarray(p.features, dtype=np.float32).tofile(handles["features"])
                np.ascontiguousarray(p.coords, dtype=np.int32).tofile(handles["coords"])
                np.ascontiguousarray(p.kept_indices, dtype=np.int32).tofile(handles["kept_indices"])
                np.ascontiguousarray(p.start_frames, dtype=np.int32).tofile(handles["start_frames"])
                np.full(n, sample_idx, dtype=np.int32).tofile(handles["sample_indices"])
            total += n
            raw_shapes[p.sample_id] = list(p.raw_shape)
            per_sample[p.sample_id] = p.extra_info
            if (sample_idx + 1) % 500 == 0 or sample_idx + 1 == len(input_paths):
                print(f"[prepare] {sample_idx + 1}/{len(input_paths)} videos, {total} trajectories")
    finally:
        for h in handles.values():
            h.close()
        if pool:
            pool.shutdown()
    if total == 0:
        raise RuntimeError("No trajectories remain after filtering.")

    def mm(name, shape):
        return np.memmap(mm_dir / f"{name}.dat", dtype=fields[name], mode="r", shape=shape)

    features = mm("features", (total, feature_dim))
    coords = mm("coords", (total, 3))
    kept = mm("kept_indices", (total,))
    starts = mm("start_frames", (total,))
    sample_idx = mm("sample_indices", (total,))
    print(f"Global feature matrix: {features.shape}")

    # 2) FINCH once, 3) save every requested partition level
    try:
        runs, finch_info = run_finch_partitions(
            features,
            args.partition_levels,
            faiss_nlist=args.faiss_nlist,
            faiss_nprobe=args.faiss_nprobe,
            faiss_train_size=args.faiss_train_size,
        )
        settings = {k: v for k, v in vars(args).items() if k != "config"}
        for level, labels, num_clusters in runs:
            save_dir = output_root / f"global_partition_partition{level}_{num_clusters}clusters_{ts}"
            summary = {
                "scope": "global",
                "clustering_mode": "partition",
                "feature_type": "flatten",
                "normalized": False,
                "saliency_filter_mode": "start_frame",
                "selected_partition": level,
                "input_paths": [str(p) for p in input_paths],
                "args": settings,
                "finch": finch_info,
                "raw_shapes_by_sample": raw_shapes,
                "per_sample_filtering": per_sample,
            }
            save_global_results(save_dir, input_paths, sample_idx, kept, starts, coords, features, labels,
                                trajectory_length=feature_dim // 2, summary=summary)
            print(f"Saved partition {level} ({num_clusters} clusters) to {save_dir}")
    finally:
        del features, coords, kept, starts, sample_idx
        if not args.keep_memmap:
            shutil.rmtree(mm_dir, ignore_errors=True)
            if not any(mm_dir.parent.iterdir()):
                mm_dir.parent.rmdir()


if __name__ == "__main__":
    main()
