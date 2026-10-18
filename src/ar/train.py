"""Training of the TraCE action-recognition head on cached features.

Ported from the research code that produced the released AR checkpoints.
Three stages, in this order (the RNG stream is shared, as in the original):

1. Trajectory concept layer: bias-free linear map trained with
   ``BCEWithLogitsLoss`` against the soft trajectory labels. Samples whose
   label row is all -1 (no filtered trajectories) are skipped. Validation BCE
   is checked every 500 steps and the best checkpoint is kept.
2. Object / scene concept layers: candidates whose mean top-5 train video-text
   similarity is not above ``clip_cutoff`` are dropped ("activation cutoff");
   a bias-free linear map is trained with the label-free CBM cos^3 loss
   against the InternVideo2 video-text similarities, with early stopping at the
   first validation increase (checked every 500 steps); concepts whose
   validation cos^3 similarity is not above ``interpretability_cutoff`` are
   dropped ("projection cutoff").
3. Sparse classifier: GLM-SAGA elastic net (alpha=0.99, step size 0.05,
   lambda = lam / alpha, single regularization value) on the concatenated
   standardized activations. The weights after ``n_iters`` epochs (or the
   tolerance stop) are kept; the validation split is only logged.

Every group is standardized with the mean/std of its raw scores over the
training set (``torch.std``, unbiased).
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..third_party.glm_saga.elasticnet import IndexedTensorDataset, glm_saga

_MIN_NORM = 1e-8


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cos_similarity_cubed_single_concept(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Label-free CBM similarity, one value per concept (column).

    Columns are centered over the batch, cubed and L2-normalized over the batch.
    """
    target = target.float()
    target = target - target.mean(dim=0, keepdim=True)
    pred = pred - pred.mean(dim=0, keepdim=True)
    target, pred = target ** 3, pred ** 3
    target = target / target.norm(p=2, dim=0, keepdim=True).clamp(min=_MIN_NORM)
    pred = pred / pred.norm(p=2, dim=0, keepdim=True).clamp(min=_MIN_NORM)
    return (target * pred).sum(dim=0)


def _standardization(W: torch.Tensor, train_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        a = train_x.float() @ W.detach().cpu().T
        return a.mean(dim=0), a.std(dim=0)


# --------------------------------------------------------------- trajectory
def train_trajectory_layer(train_x: torch.Tensor, val_x: torch.Tensor, train_y: torch.Tensor,
                           val_y: torch.Tensor, steps: int = 3000, batch_size: int = 50000,
                           lr: float = 1e-3, device: str = "cuda", log=print) -> Dict[str, torch.Tensor]:
    """train_y / val_y: [N, Mt] soft labels; rows equal to -1 mark videos without trajectories."""
    tr = torch.where(train_y.max(dim=1).values != -1)[0]
    va = torch.where(val_y.max(dim=1).values != -1)[0]
    Xtr, Ytr, Xva, Yva = train_x[tr], train_y[tr], val_x[va], val_y[va]
    layer = torch.nn.Linear(Xtr.shape[1], Ytr.shape[1], bias=False).to(device)
    torch.nn.init.xavier_uniform_(layer.weight)
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    crit = torch.nn.BCEWithLogitsLoss()
    indices = list(range(len(Xtr)))
    bs = min(batch_size, len(Xtr))
    best, best_step, best_w = float("inf"), 0, None
    for i in range(steps):
        batch = torch.LongTensor(random.sample(indices, k=bs))
        loss = crit(layer(Xtr[batch].to(device)), Ytr[batch].to(device)).mean()
        loss.backward()
        opt.step()
        if i % 500 == 0 or i == steps - 1:
            with torch.no_grad():
                v = crit(layer(Xva.to(device)), Yva.to(device)).mean()
            if i == 0 or v <= best:
                best, best_step, best_w = v, i, layer.weight.detach().clone()
            log(f"[trajectory] step {i}: train bce {loss.item():.4f} val bce {v.item():.4f}")
        opt.zero_grad()
    log(f"[trajectory] best step {best_step}, val bce {float(best):.4f}")
    W = best_w.cpu()
    mean, std = _standardization(W, train_x)
    return {"W": W, "mean": mean, "std": std, "best_step": best_step, "val_loss": float(best)}


# ------------------------------------------------------------ object / scene
def activation_filter(train_sim: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Keep concepts whose mean top-5 train video-text similarity is above ``cutoff``."""
    return torch.topk(train_sim, dim=0, k=5)[0].mean(dim=0) > cutoff


def train_text_concept_layer(train_x: torch.Tensor, val_x: torch.Tensor, train_sim: torch.Tensor,
                             val_sim: torch.Tensor, interpretability_cutoff: float, steps: int = 3000,
                             batch_size: int = 50000, lr: float = 1e-3, device: str = "cuda",
                             name: str = "object", log=print) -> Dict[str, torch.Tensor]:
    """train_sim / val_sim: [N, C] InternVideo2 video-text cosine pseudo-labels (after the activation cutoff)."""
    layer = torch.nn.Linear(train_x.shape[1], train_sim.shape[1], bias=False).to(device)
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    indices = list(range(len(train_x)))
    bs = min(batch_size, len(train_x))
    best, best_step, best_w = float("inf"), 0, None
    for i in range(steps):
        batch = torch.LongTensor(random.sample(indices, k=bs))
        out = layer(train_x[batch].to(device))
        loss = -cos_similarity_cubed_single_concept(train_sim[batch].to(device), out).mean()
        loss.backward()
        opt.step()
        if i % 500 == 0 or i == steps - 1:
            with torch.no_grad():
                v = -cos_similarity_cubed_single_concept(val_sim.to(device), layer(val_x.to(device))).mean()
            if i == 0:
                best, best_step, best_w = v, i, layer.weight.detach().clone()
            elif v < best:
                best, best_step, best_w = v, i, layer.weight.detach().clone()
            else:
                break
            log(f"[{name}] step {i}: train sim {-loss.item():.4f} val sim {-v.item():.4f}")
        opt.zero_grad()
    layer.load_state_dict({"weight": best_w})
    with torch.no_grad():
        sim = cos_similarity_cubed_single_concept(val_sim.to(device), layer(val_x.to(device)))
    keep = (sim > interpretability_cutoff).cpu()
    W = best_w[keep.to(best_w.device)].cpu()
    log(f"[{name}] best step {best_step}; projection cutoff {interpretability_cutoff}: "
        f"{len(keep)} -> {int(keep.sum())} concepts")
    mean, std = _standardization(W, train_x) if len(W) else (torch.zeros(0), torch.zeros(0))
    return {"W": W, "mean": mean, "std": std, "keep": keep, "best_step": best_step, "val_sim": sim.cpu()}


# ---------------------------------------------------------------- classifier
def train_sparse_classifier(train_z: torch.Tensor, train_y: torch.Tensor, val_z: torch.Tensor,
                            val_y: torch.Tensor, num_classes: int, lam: float = 7e-4, alpha: float = 0.99,
                            step_size: float = 0.05, n_iters: int = 30000, batch_size: int = 128,
                            device: str = "cuda", verbose: int = 500) -> Dict:
    train_loader = DataLoader(IndexedTensorDataset(train_z, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_z, val_y), batch_size=batch_size, shuffle=False)
    linear = torch.nn.Linear(train_z.shape[1], num_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    out = glm_saga(linear, train_loader, step_size, n_iters, alpha, epsilon=1, k=1, val_loader=val_loader,
                   do_zero=False, metadata={"max_reg": {"nongrouped": lam}}, n_ex=len(train_z),
                   n_classes=num_classes, verbose=verbose, family="multinomial")
    p = out["path"][0]
    W, b = p["weight"], p["bias"]
    return {"W": W, "b": b, "lam": float(p["lam"]), "alpha": alpha, "metrics": p["metrics"],
            "nonzero": int((W.abs() > 1e-5).sum())}


def standardize(x: torch.Tensor, W: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x.float() @ W.T - mean) / std


def concept_similarity(video_feats: torch.Tensor, concept_feats: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between L2-normalized InternVideo2 video features and concept text features."""
    v = video_feats.float() / video_feats.float().norm(dim=1, keepdim=True)
    t = concept_feats.float() / concept_feats.float().norm(dim=1, keepdim=True)
    return v @ t.T


def labels_from_pkl(items: List[dict], videos: List[str]) -> torch.Tensor:
    """Align a soft-label pkl (list of {video_name, attribute_label}) to the annotation order.

    Names are matched without file extension (label files may store ``.mp4``
    names for ``.avi`` videos).
    """
    stem = lambda n: os.path.splitext(n)[0]
    lookup = {stem(it["video_name"]): it["attribute_label"] for it in items}
    missing = [v for v in videos if stem(v) not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} videos have no trajectory label, e.g. {missing[:3]}")
    return torch.tensor([lookup[stem(v)] for v in videos], dtype=torch.float32)
