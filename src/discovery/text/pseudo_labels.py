"""Object / scene pseudo-labels from InternVideo2 video-text similarity (paper Eq. 20).

c_i^g = E_T(C_g) E_V(V_i): raw cosine similarity between the video embedding
and the text embedding of ``"A video of {concept}"`` (no softmax, no rescaling).
Both embeddings are the 512-d InternVideo2 projections; compute them with
``src.features.internvideo2`` (``encode_video_file`` / ``encode_concepts``).

The same matrix also drives the AR activation filter: a concept is kept when the
mean of its top-5 similarities over the training videos exceeds ``clip_cutoff``.
"""
from typing import Sequence

import torch
import torch.nn.functional as F

CONCEPT_PROMPT = "A video of {}"


def concept_prompts(concepts: Sequence[str], template: str = CONCEPT_PROMPT):
    return [template.format(c) for c in concepts]


def concept_pseudo_labels(video_feats: torch.Tensor, text_feats: torch.Tensor,
                          normalize: bool = True) -> torch.Tensor:
    """[N, D] video features x [M, D] concept text features -> [N, M] cosine similarities.

    The research code L2-normalizes both sides before the product (features
    saved by the extractor are already unit norm, so this is a no-op for them).
    """
    v, t = video_feats.float(), text_feats.float()
    if normalize:
        v, t = F.normalize(v, dim=-1), F.normalize(t, dim=-1)
    return v @ t.T


def activation_scores(pseudo_labels: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Per-concept mean of the top-k similarities over videos (AR activation filter score)."""
    k = min(k, pseudo_labels.shape[0])
    return pseudo_labels.topk(k, dim=0).values.mean(dim=0)


def activation_filter(pseudo_labels: torch.Tensor, cutoff: float, k: int = 5) -> torch.Tensor:
    """Boolean mask of concepts kept by the activation filter (``score > cutoff``)."""
    return activation_scores(pseudo_labels, k) > cutoff
