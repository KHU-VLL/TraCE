"""TraCE video question answering head: video concept layer, text concept
projector and sparse concept matching (paper Eq. 5).

For a video feature ``x`` and the feature ``u_j`` of answer candidate ``j``::

    z_g   = Wv_g x                      video-side concept activation (no bias, no sigmoid)
    r_gj  = Wt_g u_j                    question/answer-side concept relevance
    p_gj  = ReLU(z_g) * ReLU(r_gj) / max(||ReLU(z_g)|| * ||ReLU(r_gj)||, eps)
    s_j   = sum_g exp(gamma_g) * sum(top-K_g entries of p_gj)

``x`` and ``u_j`` are the 512-d L2-normalized InternVideo2-s2-1B projections
(``vision_proj`` / ``text_proj``). ``K_g = 5`` and ``eps = 1e-8``. There is no
bias or any other additive score term, so the per-concept contributions defined
below sum exactly to ``s_j``.

Contribution of concept m (group g) to candidate j is
``exp(gamma_g) * p_gj[m]`` when m is among the top-K_g entries for that
candidate and 0 otherwise.

Interventions act on the raw video-side activation ``z`` (before ReLU and
normalization); ReLU, normalization and top-K are recomputed after the edit,
which is what the model computes for a video whose concept took the new value.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from ..concepts import ConceptVocab

FORMAT = "trace-vqa-v1"


@dataclass
class VQAExplanation:
    scores: torch.Tensor           # [J]
    pred: int
    target: int
    contributions: torch.Tensor    # [J, M] per-candidate concept contributions
    group_scores: Dict[str, torch.Tensor]  # group -> [J]
    video_activations: torch.Tensor  # [M] raw z
    text_relevance: torch.Tensor     # [J, M] raw r
    candidates: Optional[List[str]] = None


class TraCEVQA(torch.nn.Module):
    def __init__(self, vocab: ConceptVocab, Wv: Dict[str, torch.Tensor], Wt: Dict[str, torch.Tensor],
                 log_tau: Dict[str, torch.Tensor], topk: int = 5, eps: float = 1e-8,
                 meta: Optional[dict] = None):
        super().__init__()
        self.vocab = vocab
        self.groups = list(vocab.groups)
        self.topk = int(topk)
        self.eps = float(eps)
        self.meta = meta or {}
        sizes = vocab.group_sizes()
        for g in self.groups:
            if Wv[g].shape[0] != sizes[g] or Wt[g].shape[0] != sizes[g]:
                raise ValueError(f"{g}: weight rows do not match vocabulary size {sizes[g]}")
            self.register_buffer(f"Wv_{g}", Wv[g].float())
            self.register_buffer(f"Wt_{g}", Wt[g].float())
            self.register_buffer(f"log_tau_{g}", torch.as_tensor(log_tau[g]).float().reshape(()))

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, bundle_dir: str, map_location: Union[str, torch.device] = "cpu") -> "TraCEVQA":
        state = torch.load(os.path.join(bundle_dir, "model.pt"), map_location=map_location,
                           weights_only=True)
        if state.get("format") != FORMAT:
            raise ValueError(f"{bundle_dir}: unexpected checkpoint format {state.get('format')!r}")
        vocab = ConceptVocab.load(os.path.join(bundle_dir, "concepts.json"))
        meta = {}
        if os.path.exists(os.path.join(bundle_dir, "meta.json")):
            with open(os.path.join(bundle_dir, "meta.json")) as f:
                meta = json.load(f)
        g = vocab.groups
        return cls(vocab, {k: state[f"Wv.{k}"] for k in g}, {k: state[f"Wt.{k}"] for k in g},
                   {k: state[f"log_tau.{k}"] for k in g}, topk=int(state["topk"]),
                   eps=float(state["eps"]), meta=meta)

    def save(self, bundle_dir: str) -> None:
        os.makedirs(bundle_dir, exist_ok=True)
        state = {"format": FORMAT, "topk": self.topk, "eps": self.eps}
        for g in self.groups:
            state[f"Wv.{g}"] = getattr(self, f"Wv_{g}").cpu()
            state[f"Wt.{g}"] = getattr(self, f"Wt_{g}").cpu()
            state[f"log_tau.{g}"] = getattr(self, f"log_tau_{g}").cpu()
        torch.save(state, os.path.join(bundle_dir, "model.pt"))
        self.vocab.save(os.path.join(bundle_dir, "concepts.json"))
        with open(os.path.join(bundle_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    def group_weight(self, g: str) -> torch.Tensor:
        """exp(gamma_g)."""
        return torch.exp(getattr(self, f"log_tau_{g}"))

    # ------------------------------------------------------------- forward
    def video_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Raw video-side concept activations z, [N, M]."""
        x = x.float()
        return torch.cat([x @ getattr(self, f"Wv_{g}").T for g in self.groups], dim=-1)

    def text_relevance(self, u: torch.Tensor) -> torch.Tensor:
        """Raw answer-side concept relevance r, [..., M] for u [..., D]."""
        u = u.float()
        return torch.cat([u @ getattr(self, f"Wt_{g}").T for g in self.groups], dim=-1)

    def matching(self, z: torch.Tensor, r: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """Score candidates from activations.

        z: [N, M] raw video activations, r: [N, J, M] raw candidate relevance.
        Returns (scores [N, J], contributions [N, J, M], group_scores {g: [N, J]}).
        """
        sl = self.vocab.slices()
        scores = torch.zeros(r.shape[:-1], device=r.device)
        contribs, group_scores = [], {}
        for g in self.groups:
            zv = F.relu(z[:, sl[g]])
            zt = F.relu(r[..., sl[g]])
            nv = zv.norm(dim=-1, keepdim=True)
            nt = zt.norm(dim=-1)
            p = zt * zv[:, None, :] / (nv * nt).clamp_min(self.eps).unsqueeze(-1)
            p = p * self.group_weight(g)
            k = min(self.topk, p.shape[-1])
            top_v, top_i = torch.topk(p, k=k, dim=-1)
            c = torch.zeros_like(p).scatter_(-1, top_i, top_v)
            contribs.append(c)
            group_scores[g] = top_v.sum(-1)
            scores = scores + group_scores[g]
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        return scores, torch.cat(contribs, dim=-1), group_scores

    def forward(self, x: torch.Tensor, u: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [N, D] video features, u: [N, J, D] candidate features -> scores [N, J]."""
        return self.matching(self.video_activations(x), self.text_relevance(u), mask=mask)[0]

    @torch.no_grad()
    def predict(self, x: torch.Tensor, u: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(x, u, mask).argmax(-1)

    # ------------------------------------------------------- explanations
    @torch.no_grad()
    def explain(self, x: torch.Tensor, u: torch.Tensor, target: Optional[int] = None,
                candidates: Optional[Sequence[str]] = None) -> VQAExplanation:
        """Explain one question: x [D], u [J, D] (J answer candidates)."""
        z = self.video_activations(x.reshape(1, -1))
        r = self.text_relevance(u.reshape(1, *u.shape[-2:]))
        s, c, gs = self.matching(z, r)
        pred = int(s[0].argmax())
        return VQAExplanation(scores=s[0].cpu(), pred=pred, target=pred if target is None else int(target),
                              contributions=c[0].cpu(), group_scores={g: v[0].cpu() for g, v in gs.items()},
                              video_activations=z[0].cpu(), text_relevance=r[0].cpu(),
                              candidates=list(candidates) if candidates is not None else None)

    def top_concepts(self, contributions: torch.Tensor, k: int = 5, group: Optional[str] = None) -> list:
        """Top-k concepts of a [M] contribution vector (non-zero entries only)."""
        ids = list(range(len(self.vocab))) if group is None else self.vocab.ids(group)
        vals = contributions[ids]
        order = torch.argsort(vals, descending=True)[:k]
        return [(ids[int(i)], self.vocab[ids[int(i)]].group, self.vocab[ids[int(i)]].name, float(vals[int(i)]))
                for i in order if float(vals[int(i)]) != 0.0]

    # -------------------------------------------------------- intervention
    @staticmethod
    def intervene(z: torch.Tensor, concept_ids: Iterable[int], value: float = 0.0) -> torch.Tensor:
        """Set raw video-side activations z (before ReLU / normalization) of the given concepts.

        value=0.0 removes the concept (ReLU(0) = 0, so it can no longer enter any top-K).
        """
        ids = torch.as_tensor(list(concept_ids), dtype=torch.long, device=z.device)
        out = z.clone()
        out[..., ids] = value
        return out

    @torch.no_grad()
    def score_with_intervention(self, x: torch.Tensor, u: torch.Tensor, concept_ids: Iterable[int],
                                value: float = 0.0, mask: Optional[torch.Tensor] = None):
        """Scores before and after editing video-side concepts. Returns (before, after) [N, J]."""
        z = self.video_activations(x)
        r = self.text_relevance(u)
        before = self.matching(z, r, mask=mask)[0]
        after = self.matching(self.intervene(z, concept_ids, value), r, mask=mask)[0]
        return before, after
