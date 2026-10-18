"""Length and redundancy filters for object / scene candidates (LF-CBM style).

(i)  length filter:      drop candidates longer than ``max_len`` (30) characters.
(ii) redundancy filter:  pairwise similarity ``(MPNet + 3 * CLIP ViT-B/16) / 4``;
     for every pair with similarity >= ``sim_cutoff`` (neither member deleted
     yet) delete the one with the lower summed similarity to all candidates.

The activation filter (top-5 mean InternVideo2 video-text similarity >
``clip_cutoff``) and the projection filter (``interpretability_cutoff`` on
validation projections) depend on video features and a trained concept layer,
so they are applied inside action-recognition training (src/ar/train.py), not here.
"""
from typing import List, Optional, Sequence

import numpy as np
import torch


def remove_too_long(concepts: Sequence[str], max_len: int = 30) -> List[str]:
    return [c for c in concepts if len(c) <= max_len]


@torch.no_grad()
def mpnet_similarity(concepts: Sequence[str], model_name: str = "sentence-transformers/all-mpnet-base-v2",
                     device: str = "cpu") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    feats = model.encode(list(concepts), convert_to_numpy=True)  # all-mpnet-base-v2 output is L2-normalized
    return feats @ feats.T


@torch.no_grad()
def clip_similarity(concepts: Sequence[str], model_name: str = "openai/clip-vit-base-patch16",
                    device: str = "cpu", batch_size: int = 500) -> np.ndarray:
    from transformers import CLIPModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device).eval()
    feats = []
    for i in range(0, len(concepts), batch_size):
        batch = tok(list(concepts[i:i + batch_size]), padding=True, truncation=True, return_tensors="pt").to(device)
        feats.append(model.get_text_features(**batch).float())
    feats = torch.cat(feats)
    feats = feats / feats.norm(dim=1, keepdim=True)
    return (feats @ feats.T).cpu().numpy()


def combined_similarity(concepts: Sequence[str], device: str = "cpu",
                        mpnet_name: str = "sentence-transformers/all-mpnet-base-v2",
                        clip_name: str = "openai/clip-vit-base-patch16") -> np.ndarray:
    """(MPNet + 3 * CLIP) / 4, as in LF-CBM (CLIP is up-weighted because MPNet has higher variance)."""
    return (mpnet_similarity(concepts, mpnet_name, device) + 3 * clip_similarity(concepts, clip_name, device)) / 4


def remove_redundant(concepts: Sequence[str], sim: np.ndarray, sim_cutoff: float = 0.9,
                     verbose: bool = False) -> List[str]:
    """LF-CBM ``filter_too_similar`` on a precomputed similarity matrix.

    Pairs are visited in row-major (i, j) order exactly like the original
    double loop; the result depends on the input order, so pass a sorted list.
    """
    row_sum = sim.sum(axis=1)
    deleted = set()
    for i, j in np.argwhere(sim >= sim_cutoff):
        if i == j or i in deleted or j in deleted:
            continue
        drop = i if row_sum[i] < row_sum[j] else j
        deleted.add(int(drop))
        if verbose:
            print(f"{concepts[i]} - {concepts[j]}, sim:{sim[i, j]:.4f} - deleting {concepts[drop]}")
    return [c for k, c in enumerate(concepts) if k not in deleted]


def filter_candidates(concepts: Sequence[str], max_len: int = 30, sim_cutoff: float = 0.9,
                      device: str = "cpu", sim: Optional[np.ndarray] = None, verbose: bool = False,
                      **encoder_kwargs) -> List[str]:
    """Length filter followed by redundancy filter."""
    kept = remove_too_long(concepts, max_len)
    if sim is None:
        sim = combined_similarity(kept, device=device, **encoder_kwargs)
    return remove_redundant(kept, sim, sim_cutoff, verbose=verbose)
