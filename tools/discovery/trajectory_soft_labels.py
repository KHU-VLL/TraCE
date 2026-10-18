"""Soft trajectory-concept labels (Eq. 18-19) from a clustering output directory.

Train split: per-video histogram of ``cluster_labels.npy`` over the clustered
trajectories, L1-normalised. Rows follow the split CSV order and keep its first
column as ``video_name``.

Val split: the val trajectory stacks are filtered exactly like the clustered ones
(foreground gating + top-rho% motion) and every surviving trajectory is assigned
to the nearest train medoid (``medoid_features.npy``, squared L2). Rows are sorted
by sample id and ``video_name`` is ``<sample_id>.mp4``, as in the released files.

    # train labels only
    python tools/discovery/trajectory_soft_labels.py \
        --cluster_dir outputs/trajectory/ntu_mutual/global_partition_partition7_121clusters_<ts> \
        --train_csv annotations/ntu_mutual/train.csv --out_dir outputs/trajectory/ntu_mutual/labels

    # + val labels
    python tools/discovery/trajectory_soft_labels.py --config configs/discovery/trajectory_ntu_mutual.yaml \
        --cluster_dir ... --train_csv ... --val_csv annotations/ntu_mutual/val.csv \
        --val_stack_dir <val trajectory stacks> --mask_root <foreground masks> --out_dir ...

Output: ``<out_dir>/train.pkl`` and ``<out_dir>/val.pkl`` (read by tools/train_ar.py and
tools/train_vqa.py via --trajectory_labels), each a list of
``{"video_name": str, "attribute_label": list[float]}`` (``[-1]*Mt`` if a video has no trajectories).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.discovery.trajectory.config import parse_args_with_config  # noqa: E402
from src.discovery.trajectory.filtering import prepare_sample  # noqa: E402
from src.discovery.trajectory.labels import (  # noqa: E402
    nearest_medoid_soft_label,
    read_split_ids,
    save_label_file,
    train_soft_labels,
)


def val_soft_labels(args, medoid_features: np.ndarray) -> list[dict]:
    num_clusters = medoid_features.shape[0]
    names = read_split_ids(Path(args.val_csv))
    if args.val_csv.endswith(".json"):  # annotation order, one row per entry
        wanted = [Path(n).stem for n in names]
    else:  # sorted unique ids, as in the released AR val files
        wanted = sorted({Path(n).stem for n in names})
    stack_dir = Path(args.val_stack_dir)
    wanted_set = set(wanted)
    paths = {p.stem: p for p in sorted(stack_dir.rglob("*.npy")) if p.stem in wanted_set}
    print(f"[val] {len(wanted_set)} ids in {args.val_csv}, {len(paths)} trajectory stacks found under {stack_dir}")

    cache: dict[str, list] = {}
    entries = []
    for i, sample_id in enumerate(wanted, start=1):
        if sample_id in cache:
            entries.append({"video_name": f"{sample_id}.mp4", "attribute_label": cache[sample_id]})
            continue
        if sample_id not in paths:
            attr = [-1] * num_clusters
        else:
            try:
                prepared = prepare_sample(
                    paths[sample_id],
                    mask_root=Path(args.mask_root),
                    motion_percentile=args.motion_percentile,
                    flow_root=Path(args.flow_root) if args.flow_root else None,
                    path_length_dtype=args.path_length_dtype,
                )
                attr = nearest_medoid_soft_label(prepared.features, medoid_features)
            except Exception as exc:  # same behaviour as the research code: empty row of zeros
                print(f"[val] {sample_id}: filtering failed ({exc}); writing zeros")
                attr = [0.0] * num_clusters
        cache[sample_id] = attr
        entries.append({"video_name": f"{sample_id}.mp4", "attribute_label": attr})
        if i % 100 == 0 or i == len(wanted):
            print(f"[val] {i}/{len(wanted)}")
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster_dir", help="output directory of cluster_trajectories.py (one partition level)")
    ap.add_argument("--train_csv", help="train split CSV (first column = video file) or VQA JSON")
    ap.add_argument("--out_dir")
    ap.add_argument("--val_csv", default=None, help="val split CSV or VQA JSON (optional)")
    ap.add_argument("--val_stack_dir", default=None, help="trajectory stacks of the val videos")
    ap.add_argument("--mask_root", default=None, help="foreground masks (needed for val)")
    ap.add_argument("--flow_root", default=None, help="optional raw flow dir to read T; otherwise #masks - 1")
    ap.add_argument("--motion_percentile", type=float, default=95.0, help="100 - rho")
    ap.add_argument("--path_length_dtype", choices=("stack", "float32"), default="stack")
    args = parse_args_with_config(ap, "labels", required=("cluster_dir", "out_dir"))

    cluster_dir, out_dir = Path(args.cluster_dir), Path(args.out_dir)
    if args.train_csv:
        train = train_soft_labels(cluster_dir, read_split_ids(Path(args.train_csv)))
        n_missing = sum(e["attribute_label"][0] == -1 for e in train)
        print(f"[train] {len(train)} videos ({n_missing} without trajectories) -> {save_label_file(train, out_dir, 'train')}")
    if args.val_csv:
        if not (args.val_stack_dir and args.mask_root):
            ap.error("--val_csv needs --val_stack_dir and --mask_root")
        medoid_features = np.load(cluster_dir / "medoid_features.npy").astype(np.float32)
        val = val_soft_labels(args, medoid_features)
        print(f"[val] {len(val)} videos -> {save_label_file(val, out_dir, 'val')}")


if __name__ == "__main__":
    main()
