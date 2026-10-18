"""Export the trajectory-concept vocabulary of one clustering run as a small release bundle.

    python tools/discovery/export_trajectory_concepts.py \
        --cluster_dir outputs/trajectory/ntu_mutual/global_partition_partition7_121clusters_<ts> \
        --dataset ntu_mutual --out_dir checkpoints/trajectory_concepts/ntu_mutual

Writes ``trajectory_concepts.npz`` (medoid_trajectories [Mt, L, 2], cluster_ids,
medoid_start_frames, medoid_coords) and ``trajectory_concepts.json`` (Mt, L, Ns,
partition level, per-concept medoid metadata). Concept k = cluster k = label column k.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.discovery.trajectory.concepts import export_bundle  # noqa: E402

DATASETS = ("ntu_mutual", "chiral_ssv2", "ucf101", "kth", "tomato", "star")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster_dir", required=True)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    info = export_bundle(a.cluster_dir, a.out_dir, a.dataset)
    print(f"[{a.dataset}] Mt={info['num_concepts']} L={info['trajectory_length']} Ns={info['num_segments']} "
          f"partition={info['partition_level']} videos={info['num_clustered_videos']} -> {a.out_dir}")


if __name__ == "__main__":
    main()
