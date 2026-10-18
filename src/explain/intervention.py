"""Top-K vs Random-K concept removal.

Protocol (AR, as used for the paper's Table 4):

* Contributions are computed for the *predicted* class of the unedited model,
  ``c_m = z_m * W_AR[y_hat, m]``.
* Only concepts of one group (default: trajectory) are candidates for removal.
* Top-K removes the K concepts with the largest signed contribution.
  Random-K removes K concepts drawn uniformly from the same group with a
  per-sample permutation (``torch.randperm``, CPU generator seeded with ``seed``).
* Removal sets the standardized activation to 0 (training-set mean score).
* A *flip* is a change of the arg-max prediction relative to the unedited model.

VQA uses the same ranking and sampling rules on the video-side activations
of the selected group; contributions are those of the predicted answer
(zero outside the top-K matching set). Removal sets the raw activation to 0,
then ReLU, normalization and top-K matching are recomputed.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch


def rankings(contrib: torch.Tensor, pool: torch.Tensor, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Per-sample orderings of the concepts in ``pool`` (global ids).

    contrib: [N, M]. Returns {"top": [N, P], "bottom": [N, P], "random": [N, P]}.
    """
    pool = pool.cpu()
    cp = contrib[:, pool].cpu()
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    rand = torch.stack([torch.randperm(cp.shape[1], generator=g) for _ in range(cp.shape[0])])
    return {"top": pool[torch.argsort(cp, dim=1, descending=True)],
            "bottom": pool[torch.argsort(cp, dim=1, descending=False)],
            "random": pool[rand]}


def remove_first_k(act: torch.Tensor, ranking: torch.Tensor, k: int, value: float = 0.0) -> torch.Tensor:
    """Set the first k concepts of each sample's ranking to ``value``."""
    out = act.clone()
    if k <= 0:
        return out
    idx = ranking[:, :k].to(act.device)
    rows = torch.arange(act.shape[0], device=act.device).unsqueeze(1).expand_as(idx)
    out[rows, idx] = value
    return out


@torch.no_grad()
def ar_removal(model, feats: torch.Tensor, labels: torch.Tensor, ks: Iterable[int] = (5, 10),
               group: str = "trajectory", seed: int = 0) -> Dict[str, dict]:
    """Top-K / Random-K / Bottom-K removal for a ``TraCEAR`` model on cached features."""
    z = model.concept_activations(feats)
    base = model.classify(z)
    yhat = base.argmax(1)
    contrib = model.contributions(z, yhat)
    pool = torch.tensor(model.vocab.ids(group))
    R = rankings(contrib, pool, seed)
    out = {"base_acc": float((yhat == labels).float().mean()), "n": int(len(labels)), "group": group,
           "pool_size": int(len(pool))}
    for k in ks:
        for name, rk in R.items():
            pred = model.classify(remove_first_k(z, rk, k)).argmax(1)
            out[f"{name}-{k}"] = {"flip_rate": float((pred != yhat).float().mean()),
                                  "acc": float((pred == labels).float().mean())}
    return out


@torch.no_grad()
def vqa_removal(model, video_feats: torch.Tensor, option_feats: torch.Tensor, answers: torch.Tensor,
                mask: Optional[torch.Tensor] = None, ks: Iterable[int] = (5, 10),
                group: str = "trajectory", seed: int = 0,
                batch_size: int = 1024) -> Dict[str, dict]:
    """Top-K / Random-K / Bottom-K removal for a ``TraCEVQA`` model on cached features."""
    N = video_feats.shape[0]
    z = model.video_activations(video_feats)
    base_s, contrib_pred = [], []
    for i in range(0, N, batch_size):
        r = model.text_relevance(option_feats[i:i + batch_size])
        m = None if mask is None else mask[i:i + batch_size]
        s, c, _ = model.matching(z[i:i + batch_size], r, mask=m)
        p = s.argmax(1)
        contrib_pred.append(c[torch.arange(len(p)), p])
        base_s.append(s)
    base = torch.cat(base_s)
    yhat = base.argmax(1)
    contrib = torch.cat(contrib_pred)
    pool = torch.tensor(model.vocab.ids(group))
    R = rankings(contrib, pool, seed)
    out = {"base_acc": float((yhat == answers).float().mean()), "n": int(N), "group": group,
           "pool_size": int(len(pool))}
    for k in ks:
        for name, rk in R.items():
            z2 = remove_first_k(z, rk, k)
            preds = []
            for i in range(0, N, batch_size):
                r = model.text_relevance(option_feats[i:i + batch_size])
                m = None if mask is None else mask[i:i + batch_size]
                s, _, _ = model.matching(z2[i:i + batch_size], r, mask=m)
                preds.append(s.argmax(1))
            pred = torch.cat(preds)
            out[f"{name}-{k}"] = {"flip_rate": float((pred != yhat).float().mean()),
                                  "acc": float((pred == answers).float().mean())}
    return out
