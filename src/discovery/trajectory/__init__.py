"""Trajectory-concept discovery (Sec. 3.2.1, App. A.5.1).

Pipeline (CLI scripts in ``tools/discovery/``):
WAFT flow -> dense trajectory stacks -> U^2-Net foreground masks ->
filtering + global FINCH clustering -> soft labels / exported concept bundle.
Pipeline: WAFT flow -> trajectories -> foreground masks -> FINCH clustering -> soft labels.
"""
from .concepts import export_bundle, load_bundle
from .filtering import PreparedSample, prepare_sample
from .labels import load_label_file, train_soft_labels
from .stacks import build_trajectory_stacks, compute_segment_start_indices
from .visualize import draw_medoid, medoid_points

__all__ = [
    "PreparedSample",
    "build_trajectory_stacks",
    "compute_segment_start_indices",
    "draw_medoid",
    "export_bundle",
    "load_bundle",
    "load_label_file",
    "medoid_points",
    "prepare_sample",
    "train_soft_labels",
]
