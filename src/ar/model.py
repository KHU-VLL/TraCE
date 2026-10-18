"""TraCE action recognition head: concept layer + sparse linear classifier.

Forward pass (identical to the released checkpoints)::

    a_g = W_g x                       raw concept score, g in (trajectory, object, scene)
    z_g = (a_g - mu_g) / sigma_g      standardization with training-set statistics
    z   = [z_trajectory; z_object; z_scene]
    logits = W_AR z + b_AR

``x`` is the 768-d VideoMAE-B/16 feature (mean-pooled tokens after ``fc_norm``).
No sigmoid is applied at inference: the trajectory branch is trained with a
sigmoid inside the BCE loss, but the classifier consumes the standardized raw
score.

Contribution of concept m to class k is ``z_m * W_AR[k, m]`` and
``sum_m contribution + b_AR[k] == logits[k]`` holds exactly.

Interventions act on the standardized activation ``z`` (the classifier
input). Removing a concept sets ``z_m = 0``, i.e. replaces the raw score by its
training-set mean.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Union

import torch

from ..concepts import ConceptVocab

FORMAT = "trace-ar-v1"


@dataclass
class ARExplanation:
    logits: torch.Tensor          # [K]
    pred: int
    target: int                   # class the contributions refer to
    contributions: torch.Tensor   # [M] z_m * W[target, m]
    activations: torch.Tensor     # [M] standardized z
    bias: float                   # b[target]


class TraCEAR(torch.nn.Module):
    def __init__(self, vocab: ConceptVocab, classes: Sequence[str],
                 W_c: Dict[str, torch.Tensor], mean: Dict[str, torch.Tensor],
                 std: Dict[str, torch.Tensor], W_g: torch.Tensor, b_g: torch.Tensor,
                 meta: Optional[dict] = None):
        super().__init__()
        self.vocab = vocab
        self.classes = list(classes)
        self.meta = meta or {}
        self.groups = list(vocab.groups)
        for g in self.groups:
            n = vocab.group_sizes()[g]
            if W_c[g].shape[0] != n:
                raise ValueError(f"{g}: W_c has {W_c[g].shape[0]} rows, vocabulary has {n}")
            self.register_buffer(f"W_{g}", W_c[g].float())
            self.register_buffer(f"mean_{g}", mean[g].float().reshape(1, -1))
            self.register_buffer(f"std_{g}", std[g].float().reshape(1, -1))
        if W_g.shape != (len(self.classes), len(vocab)):
            raise ValueError(f"W_g shape {tuple(W_g.shape)} != ({len(self.classes)}, {len(vocab)})")
        self.register_buffer("W_g", W_g.float())
        self.register_buffer("b_g", b_g.float().reshape(-1))

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, bundle_dir: str, map_location: Union[str, torch.device] = "cpu") -> "TraCEAR":
        state = torch.load(os.path.join(bundle_dir, "model.pt"), map_location=map_location,
                           weights_only=True)
        if state.get("format") != FORMAT:
            raise ValueError(f"{bundle_dir}: unexpected checkpoint format {state.get('format')!r}")
        vocab = ConceptVocab.load(os.path.join(bundle_dir, "concepts.json"))
        with open(os.path.join(bundle_dir, "classes.txt")) as f:
            classes = [l.rstrip("\n") for l in f if l.strip()]
        meta = {}
        if os.path.exists(os.path.join(bundle_dir, "meta.json")):
            with open(os.path.join(bundle_dir, "meta.json")) as f:
                meta = json.load(f)
        g = vocab.groups
        return cls(vocab, classes,
                   {k: state[f"W_c.{k}"] for k in g}, {k: state[f"mean.{k}"] for k in g},
                   {k: state[f"std.{k}"] for k in g}, state["W_g"], state["b_g"], meta)

    def save(self, bundle_dir: str) -> None:
        os.makedirs(bundle_dir, exist_ok=True)
        state = {"format": FORMAT, "W_g": self.W_g.cpu(), "b_g": self.b_g.cpu()}
        for g in self.groups:
            state[f"W_c.{g}"] = getattr(self, f"W_{g}").cpu()
            state[f"mean.{g}"] = getattr(self, f"mean_{g}").cpu().reshape(-1)
            state[f"std.{g}"] = getattr(self, f"std_{g}").cpu().reshape(-1)
        torch.save(state, os.path.join(bundle_dir, "model.pt"))
        self.vocab.save(os.path.join(bundle_dir, "concepts.json"))
        with open(os.path.join(bundle_dir, "classes.txt"), "w") as f:
            f.write("\n".join(self.classes) + "\n")
        with open(os.path.join(bundle_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    # ------------------------------------------------------------- forward
    def raw_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Raw concept scores ``a = W x`` (before standardization), [N, M]."""
        x = x.float()
        return torch.cat([x @ getattr(self, f"W_{g}").T for g in self.groups], dim=1)

    def concept_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Standardized concept activations ``z`` fed to the classifier, [N, M]."""
        x = x.float()
        zs = [(x @ getattr(self, f"W_{g}").T - getattr(self, f"mean_{g}")) / getattr(self, f"std_{g}")
              for g in self.groups]
        return torch.cat(zs, dim=1)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_g.T + self.b_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.concept_activations(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)

    # ------------------------------------------------------- explanations
    def contributions(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``z[i, m] * W_g[target[i], m]`` for every sample, [N, M]."""
        return z * self.W_g[target]

    @torch.no_grad()
    def explain(self, x: torch.Tensor, target: Optional[int] = None) -> ARExplanation:
        """Explain a single feature vector ``x`` ([D] or [1, D]).

        ``target`` defaults to the predicted class.
        """
        z = self.concept_activations(x.reshape(1, -1))
        logits = self.classify(z)[0]
        pred = int(logits.argmax())
        t = pred if target is None else int(target)
        contrib = self.contributions(z, torch.tensor([t], device=z.device))[0]
        return ARExplanation(logits=logits.cpu(), pred=pred, target=t, contributions=contrib.cpu(),
                             activations=z[0].cpu(), bias=float(self.b_g[t]))

    def top_concepts(self, contributions: torch.Tensor, k: int = 5,
                     group: Optional[str] = None) -> list:
        """Top-k concepts by signed contribution, optionally within one group."""
        ids = list(range(len(self.vocab))) if group is None else self.vocab.ids(group)
        vals = contributions[ids]
        order = torch.argsort(vals, descending=True)[:k]
        return [(ids[int(i)], self.vocab[ids[int(i)]].group, self.vocab[ids[int(i)]].name,
                 float(vals[int(i)])) for i in order]

    # -------------------------------------------------------- intervention
    @staticmethod
    def intervene(z: torch.Tensor, concept_ids: Iterable[int], value: float = 0.0) -> torch.Tensor:
        """Copy of the standardized activations with ``z[:, ids] = value``.

        value=0.0 removes the concept (its score becomes the training-set mean).
        """
        ids = torch.as_tensor(list(concept_ids), dtype=torch.long, device=z.device)
        out = z.clone()
        out[:, ids] = value
        return out
