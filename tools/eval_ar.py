"""Evaluate a TraCE AR bundle on cached VideoMAE features.

    python tools/eval_ar.py --bundle checkpoints/ar/ntu_mutual \
        --features features/ar/ntu_mutual/val_videomae.pt \
        --annotation data/annotations/ntu_mutual/val.csv [--removal]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ar import TraCEAR  # noqa: E402
from src.ar.data import load_features, read_annotation  # noqa: E402
from src.explain.intervention import ar_removal  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--removal", action="store_true", help="also run Top-K / Random-K trajectory removal")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="", help="optional json output path")
    a = ap.parse_args()

    model = TraCEAR.load(a.bundle).to(a.device)
    videos, labels = read_annotation(a.annotation)
    x = load_features(a.features, videos).to(a.device)
    y = torch.tensor(labels, device=a.device)
    pred = model.predict(x)
    res = {"n": len(y), "top1": float((pred == y).float().mean())}
    print(f"[AR] {a.bundle}: top1={res['top1'] * 100:.2f}% (n={res['n']})")
    if a.removal:
        res["removal"] = ar_removal(model, x, y, ks=a.ks, seed=a.seed)
        for k, v in res["removal"].items():
            if isinstance(v, dict):
                print(f"  {k:>10}: flip={v['flip_rate'] * 100:5.1f}%  acc={v['acc'] * 100:5.1f}%")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
