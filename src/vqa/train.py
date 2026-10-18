"""Training of the TraCE VQA head on cached InternVideo2 features.

Ported from the research code that produced the released VQA checkpoints.
Each concept group is trained as its own block with its own matching scalar
``gamma_g`` (``log_tau``); at inference the block scores are summed (Eq. 5).

Trajectory block
  stage 1  video-side layer only: BCE against the soft trajectory labels
           (same routine as the AR trajectory layer, ``steps`` updates on
           video-level features, best validation BCE every 500 steps);
  stage 2  ``epochs`` epochs over questions, updating Wv, Wt and gamma with
           ``lambda_qa * L_QA + w * L_align``. L_QA is the cross-entropy over
           the matching scores (top-K, ReLU-normalized); L_align is InfoNCE
           between the raw video activation and the raw relevance vectors of
           all candidates (cosine / ``tau``, correct answer = positive); ``w``
           ramps linearly from 0 to 1 over the first ``align_warmup_steps``
           updates. The epoch with the best validation accuracy is kept.

Object / scene blocks (one block per group, trained independently)
  ``steps`` updates of ``(1 - lambda_qa) * (L_v + L_t) + lambda_qa * L_QA``:
  L_v = -cos(z, s)^3 per sample against the InternVideo2 video-text
  similarities s ("A video of {concept}"), L_t = -cos(r, d) per candidate
  against all-mpnet-base-v2 option-concept similarities d. Every 200 steps
  the loss on 64 random validation questions is computed and the best state
  is kept.

The validation split used for model selection is the evaluation split of
each benchmark, as in the research code.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..ar.train import train_trajectory_layer
from .data import MCQASplit


def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


@dataclass
class QASamples:
    """Questions in the order used by the research code (sorted by "<video>__<index>")."""
    video: torch.Tensor       # [N, D]
    options: torch.Tensor     # [N, J, D]
    mask: torch.Tensor        # [N, J]
    answer: torch.Tensor      # [N]
    option_text: List[List[str]]
    video_names: List[str]


def research_order(split: MCQASplit) -> QASamples:
    keys = [f"{os.path.basename(v)}__{i}" for i, v in enumerate(split.videos)]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    return QASamples(split.video_feats[order], split.option_feats[order], split.option_mask[order],
                     split.answers[order], [split.options[i] for i in order], [split.videos[i] for i in order])


def init_linear(in_dim: int, out_dim: int, device) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim, bias=False)
    nn.init.xavier_uniform_(layer.weight)
    return layer.to(device)


def nec_scores(zv: torch.Tensor, zt: torch.Tensor, mask: Optional[torch.Tensor], k: int,
               log_tau: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Single-group matching score (training form of Eq. 5)."""
    zv, zt = F.relu(zv), F.relu(zt)
    contrib = zt * zv[:, None, :]
    s = torch.topk(contrib, k=min(k, contrib.shape[-1]), dim=-1).values.sum(-1)
    s = s / (zv.norm(dim=-1, keepdim=True) * zt.norm(dim=-1)).clamp_min(eps)
    s = s * torch.exp(log_tau)
    return s if mask is None else s.masked_fill(~mask, -1e9)


def infonce_align(zv: torch.Tensor, zt: torch.Tensor, ans: torch.Tensor, mask: torch.Tensor,
                  tau: float = 0.07, eps: float = 1e-6) -> torch.Tensor:
    s = zt / (zt.norm(dim=-1, keepdim=True) + eps)
    t = zv / (zv.norm(dim=-1, keepdim=True) + eps)
    logits = (s * t[:, None, :]).sum(-1) / tau
    return F.cross_entropy(logits.masked_fill(~mask, -1e9), ans)


# ---------------------------------------------------------------- trajectory
def train_trajectory_block(train: QASamples, val: QASamples, video_train: Dict[str, torch.Tensor],
                           video_val: Dict[str, torch.Tensor], labels_train: Dict[str, List[float]],
                           labels_val: Dict[str, List[float]], label_order_train: List[str],
                           label_order_val: List[str], stage1_steps: int = 5000, stage1_batch: int = 256,
                           epochs: int = 100, batch_size: int = 256, lr: float = 1e-3, k: int = 5,
                           lambda_qa: float = 0.2, tau: float = 0.07, align_warmup_steps: int = 50,
                           device: str = "cuda", log: Callable = print) -> Dict[str, torch.Tensor]:
    D = train.video.shape[1]
    C = len(next(iter(labels_train.values())))
    gI, gT = init_linear(D, C, device), init_linear(D, C, device)

    # stage 1: video-level BCE, in the order of the label files
    def stack(order, feats, labels):
        X = torch.stack([feats[_stem(n)] for n in order])
        Y = torch.tensor([labels[_stem(n)] for n in order], dtype=torch.float32)
        return X, Y
    Xtr, Ytr = stack(label_order_train, video_train, labels_train)
    Xva, Yva = stack(label_order_val, video_val, labels_val)
    s1 = train_trajectory_layer(Xtr, Xva, Ytr, Yva, stage1_steps, stage1_batch, lr, device, log)
    with torch.no_grad():
        gI.weight.copy_(s1["W"].to(device))

    # stage 2
    log_tau = nn.Parameter(torch.tensor(0.0, device=device))
    opt = torch.optim.Adam(list(gI.parameters()) + list(gT.parameters()) + [log_tau], lr=lr)
    n = len(train.answer)
    best, state, step = -1.0, None, 0
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            v, t = train.video[b].to(device), train.options[b].to(device)
            m, y = train.mask[b].to(device), train.answer[b].to(device)
            zv, zt = gI(v), gT(t)
            L_qa = F.cross_entropy(nec_scores(zv, zt, m, k, log_tau), y)
            w = min(max(step / float(align_warmup_steps), 0.0), 1.0) if align_warmup_steps > 0 else 1.0
            loss = lambda_qa * L_qa + w * infonce_align(zv, zt, y, m, tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        acc = evaluate_block(gI.weight, gT.weight, log_tau, val, k, device)
        log(f"[trajectory] epoch {ep}: val acc {acc:.4f}")
        if acc > best:
            best = acc
            state = {"Wv": gI.weight.detach().cpu().clone(), "Wt": gT.weight.detach().cpu().clone(),
                     "log_tau": log_tau.detach().cpu().clone(), "epoch": ep}
    log(f"[trajectory] best epoch {state['epoch']} val acc {best:.4f}")
    state["val_acc"] = best
    return state


@torch.no_grad()
def evaluate_block(Wv, Wt, log_tau, data: QASamples, k: int, device, batch_size: int = 1024) -> float:
    correct = 0
    for i in range(0, len(data.answer), batch_size):
        zv = data.video[i:i + batch_size].to(device) @ Wv.T
        zt = data.options[i:i + batch_size].to(device) @ Wt.T
        s = nec_scores(zv, zt, data.mask[i:i + batch_size].to(device), k, log_tau)
        correct += (s.argmax(1).cpu() == data.answer[i:i + batch_size]).sum().item()
    return correct / len(data.answer)


# ------------------------------------------------------------- object / scene
class MPNetTeacher:
    """Option-concept similarity with sentence-transformers/all-mpnet-base-v2 (mean pooling)."""

    def __init__(self, concepts: List[str], device, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.enc = AutoModel.from_pretrained(model_name).to(device).eval()
        self.concepts = self.encode(concepts)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            inp = self.tok(texts[i:i + batch_size], padding=True, truncation=True, return_tensors="pt").to(self.device)
            h = self.enc(**inp).last_hidden_state
            m = inp["attention_mask"].unsqueeze(-1)
            out.append(F.normalize((h * m).sum(1) / m.sum(1).clamp_min(1), dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def score(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts) @ self.concepts.T


def _cos(a, b, eps=1e-8):
    return ((a / (a.norm(dim=1, keepdim=True) + eps)) * (b / (b.norm(dim=1, keepdim=True) + eps))).sum(1)


def train_text_block(train: QASamples, val: QASamples, concepts: List[str], concept_feats: torch.Tensor,
                     steps: int = 5000, batch_size: int = 256, lr: float = 1e-3, k: int = 5,
                     lambda_qa: float = 0.2, use_half: bool = True, device: str = "cuda",
                     name: str = "object", teacher: Optional[MPNetTeacher] = None,
                     log: Callable = print) -> Dict[str, torch.Tensor]:
    D, C = train.video.shape[1], len(concepts)
    teacher = teacher or MPNetTeacher(concepts, device)
    uniq = sorted({o for d in (train, val) for opts in d.option_text for o in opts})
    cache = teacher.score(uniq).cpu()
    row = {o: i for i, o in enumerate(uniq)}
    Wv, Wt = init_linear(D, C, device), init_linear(D, C, device)
    log_tau = nn.Parameter(torch.tensor(0.0, device=device))
    opt = torch.optim.Adam(list(Wv.parameters()) + list(Wt.parameters()) + [log_tau], lr=lr)
    cf = concept_feats.float().to(device)
    amp = use_half and str(device).startswith("cuda")

    def losses(data: QASamples, idx: List[int]):
        v = data.video[idx].to(device)
        opts = [data.option_text[i] for i in idx]
        nopt = [len(o) for o in opts]
        t_flat = torch.cat([data.options[i, :n] for i, n in zip(idx, nopt)]).to(device)
        t_tgt = cache[[row[o] for os_ in opts for o in os_]].to(device)
        y = data.answer[idx].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            t_pred = Wt(t_flat)
            L_t = -_cos(t_tgt, t_pred).mean()
            v_pred = Wv(v)
            L_v = -(_cos(v @ cf.T, v_pred) ** 3).mean()
            qa, off = [], 0
            for si, ni in enumerate(nopt):
                s = nec_scores(v_pred[si:si + 1], t_pred[off:off + ni].unsqueeze(0), None, k, log_tau)
                qa.append(F.cross_entropy(s, y[si:si + 1]))
                off += ni
            L_qa = torch.stack(qa).mean()
            loss = (1.0 - lambda_qa) * (L_v + L_t) + lambda_qa * L_qa
        return loss, (L_v, L_t, L_qa)

    indices = list(range(len(train.answer)))
    bs = min(batch_size, len(indices))
    best, best_state = float("inf"), None
    for step in range(steps):
        loss, parts = losses(train, random.sample(indices, k=bs))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                vidx = random.sample(range(len(val.answer)), k=min(64, len(val.answer)))
                vloss, vparts = losses(val, vidx)
            log(f"[{name}] step {step}: train Lv {parts[0].item():.4f} Lt {parts[1].item():.4f} "
                f"Lqa {parts[2].item():.4f} | val total {vloss.item():.4f}")
            if step == 0 or vloss.item() < best:
                best = vloss.item()
                best_state = {"Wv": Wv.weight.detach().cpu().clone(), "Wt": Wt.weight.detach().cpu().clone(),
                              "log_tau": log_tau.detach().cpu().clone(), "step": step}
    log(f"[{name}] best step {best_state['step']} val total {best:.4f}")
    return best_state
