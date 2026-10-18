"""Unit tests that need no data: score decomposition, interventions, save/load.

    python -m pytest tests -q        (or: python tests/test_core.py)

If release bundles are present under checkpoints/, the same checks also run on them.
"""
import glob
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.ar import TraCEAR  # noqa: E402
from src.concepts import ConceptVocab, trajectory_names  # noqa: E402
from src.explain.intervention import ar_removal, vqa_removal  # noqa: E402
from src.vqa import TraCEVQA  # noqa: E402

torch.manual_seed(0)
SIZES = {"trajectory": 7, "object": 5, "scene": 3}


def toy_vocab():
    return ConceptVocab.from_groups({"trajectory": trajectory_names(7), "object": [f"o{i}" for i in range(5)],
                                     "scene": [f"s{i}" for i in range(3)]})


def toy_ar(D=16, K=4):
    g = SIZES
    return TraCEAR(toy_vocab(), [f"c{i}" for i in range(K)], {k: torch.randn(n, D) for k, n in g.items()},
                   {k: torch.randn(n) for k, n in g.items()}, {k: torch.rand(n) + 0.5 for k, n in g.items()},
                   torch.randn(K, sum(g.values())), torch.randn(K))


def toy_vqa(D=16):
    g = SIZES
    return TraCEVQA(toy_vocab(), {k: torch.randn(n, D) for k, n in g.items()},
                    {k: torch.randn(n, D) for k, n in g.items()}, {k: torch.randn(()) for k in g}, topk=2)


def check_ar(model, x):
    z = model.concept_activations(x)
    logits = model.classify(z)
    for t in range(len(model.classes)):
        c = model.contributions(z, torch.full((len(x),), t))
        assert torch.allclose(c.sum(1) + model.b_g[t], logits[:, t], atol=1e-4)
    e = model.explain(x[0])
    assert abs(float(e.contributions.sum()) + e.bias - float(e.logits[e.pred])) < 1e-4
    # intervention touches only the requested columns and changes the logits accordingly
    ids = [0, len(model.vocab) - 1]
    z2 = model.intervene(z, ids, 0.0)
    mask = torch.ones(z.shape[1], dtype=torch.bool)
    mask[ids] = False
    assert torch.equal(z2[:, mask], z[:, mask]) and torch.all(z2[:, ids] == 0)
    delta = model.classify(z2) - logits
    expect = -(z[:, ids] @ model.W_g[:, ids].T)
    assert torch.allclose(delta, expect, atol=1e-4)


def check_vqa(model, x, u):
    z, r = model.video_activations(x), model.text_relevance(u)
    s, c, gs = model.matching(z, r)
    assert torch.allclose(c.sum(-1), s, atol=1e-4)                      # no bias term
    assert torch.allclose(sum(gs.values()), s, atol=1e-4)
    for g, sl in model.vocab.slices().items():                         # at most K non-zero per group
        assert int((c[..., sl] > 0).sum(-1).max()) <= model.topk
    ids = model.vocab.ids("trajectory")[:2]
    z2 = model.intervene(z, ids, 0.0)
    _, c2, _ = model.matching(z2, r)
    assert torch.all(c2[..., ids] == 0)                                # removed concepts cannot contribute
    before, after = model.score_with_intervention(x, u, ids)
    assert torch.allclose(before, s, atol=1e-5)
    assert torch.allclose(after, model.matching(z2, r)[0], atol=1e-5)


def test_ar_toy():
    m = toy_ar()
    x = torch.randn(9, 16)
    check_ar(m, x)
    with tempfile.TemporaryDirectory() as d:
        m.save(d)
        m2 = TraCEAR.load(d)
        assert torch.allclose(m(x), m2(x)) and m2.vocab.names() == m.vocab.names()
    r = ar_removal(m, x, torch.randint(0, 4, (9,)), ks=[2])
    assert 0 <= r["top-2"]["flip_rate"] <= 1


def test_vqa_toy():
    m = toy_vqa()
    x, u = torch.randn(6, 16), torch.randn(6, 4, 16)
    check_vqa(m, x, u)
    mask = torch.ones(6, 4, dtype=torch.bool)
    mask[0, 3] = False
    assert m.forward(x, u, mask)[0, 3] < -1e8
    with tempfile.TemporaryDirectory() as d:
        m.save(d)
        m2 = TraCEVQA.load(d)
        assert torch.allclose(m(x, u), m2(x, u))
    r = vqa_removal(m, x, u, torch.zeros(6, dtype=torch.long), mask, ks=[1])
    assert "random-1" in r


def test_release_bundles():
    for d in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "ar", "*"))):
        m = TraCEAR.load(d)
        D = m.W_trajectory.shape[1]
        assert m.vocab.group_sizes() == m.meta.get("group_sizes", m.vocab.group_sizes())
        assert m.vocab.groups == ["trajectory", "object", "scene"]
        check_ar(m, torch.randn(4, D))
    for d in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "vqa", "*"))):
        m = TraCEVQA.load(d)
        D = m.Wv_trajectory.shape[1]
        check_vqa(m, torch.nn.functional.normalize(torch.randn(3, D), dim=-1),
                  torch.nn.functional.normalize(torch.randn(3, 5, D), dim=-1))


if __name__ == "__main__":
    test_ar_toy()
    test_vqa_toy()
    test_release_bundles()
    print("all tests passed")
