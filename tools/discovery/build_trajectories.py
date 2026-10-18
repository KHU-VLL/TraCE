"""Build dense trajectory stacks from saved optical flow.

    python tools/discovery/build_trajectories.py --config configs/discovery/trajectory_ntu_mutual.yaml \
        --input_dir data/trajectory/ntu_mutual/flow --output_dir data/trajectory/ntu_mutual/stacks

Input: ``<input_dir>/**/<id>.npy`` float ``[T, H, W, 2]`` (from extract_flow_waft.py).
Output: ``<output_dir>/<same relative path>.npy`` of shape ``[Ns, L, H, W, 2]``
(default float16): for each of the Ns segments, every pixel of the segment's first
flow frame is a seed and step r stores the bilinearly sampled flow (= displacement)
at the current position. Existing outputs are skipped.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.discovery.trajectory.config import parse_args_with_config  # noqa: E402
from src.discovery.trajectory.stacks import build_trajectory_stacks  # noqa: E402

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", help="directory of [T,H,W,2] flow .npy files")
    ap.add_argument("--output_dir")
    ap.add_argument("--length", type=int, default=16, help="trajectory length L (flow steps)")
    ap.add_argument("--num_segments", type=int, default=3, help="segments per video Ns")
    ap.add_argument("--compute_dtype", choices=tuple(DTYPES), default="float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    args = parse_args_with_config(ap, "trajectories", required=("input_dir", "output_dir"))

    in_root, out_root = Path(args.input_dir), Path(args.output_dir)
    files = sorted(in_root.rglob("*.npy"))
    files = files[args.shard_index::args.num_shards]
    if not files:
        raise FileNotFoundError(f"No .npy flow files under {in_root}")

    for i, src in enumerate(files, start=1):
        dst = out_root / src.relative_to(in_root)
        if dst.exists():
            continue
        t0 = time.time()
        flow = np.load(src).astype(np.float32)
        stacks = build_trajectory_stacks(
            flow, args.length, args.num_segments, args.device, DTYPES[args.compute_dtype]
        ).astype("float16")
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, stacks)
        pad = f" (zero/edge padded {args.length - flow.shape[0]} steps)" if flow.shape[0] < args.length else ""
        print(f"[{i}/{len(files)}] {src.stem}: flow {flow.shape} -> {stacks.shape}{pad} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
