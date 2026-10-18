"""Plots for concept contributions (matplotlib)."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

GROUP_COLORS = {"trajectory": "#d1495b", "object": "#00798c", "scene": "#edae49"}


def _plt():
    import matplotlib.pyplot as plt  # headless sessions fall back to the Agg backend automatically
    return plt


def plot_contributions(items: Sequence[tuple], title: str = "", path: Optional[str] = None,
                       extra: Optional[dict] = None):
    """Horizontal bar chart of (concept_id, group, name, value) tuples.

    ``extra`` adds labelled bars for additional score terms (e.g. {"bias": b}).
    """
    plt = _plt()
    rows = [(f"[{g}] {n}", v, GROUP_COLORS.get(g, "gray")) for _, g, n, v in items]
    for k, v in (extra or {}).items():
        rows.append((k, v, "#555555"))
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(rows) + 1.2))
    ax.barh(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("contribution to score")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def plot_vqa_matrix(contributions: np.ndarray, concept_labels: List[str], candidates: List[str],
                    title: str = "", path: Optional[str] = None):
    """Concept x candidate matrix of matching contributions (rows: concepts)."""
    plt = _plt()
    c = np.asarray(contributions)
    fig, ax = plt.subplots(figsize=(1.2 * len(candidates) + 3, 0.35 * len(concept_labels) + 1.5))
    im = ax.imshow(c, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels(candidates, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(concept_labels)))
    ax.set_yticklabels(concept_labels, fontsize=8)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c[i, j] != 0:
                ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def plot_removal_curve(results: dict, ks: Sequence[int], path: Optional[str] = None, title: str = ""):
    """Accuracy after removing K concepts for each ranking in ``results`` ({"top": [acc...], ...})."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for name, accs in results.items():
        ax.plot(list(ks), accs, marker="o", label=name)
    ax.set_xlabel("removed concepts K")
    ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=10)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig
