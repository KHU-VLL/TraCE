"""Evaluate a TraCE VQA bundle on a cached InternVideo2 feature split.

    python tools/eval_vqa.py --bundle checkpoints/vqa/ntu_mutual \
        --split_dir features/vqa/ntu_mutual/val [--removal]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explain.intervention import vqa_removal  # noqa: E402
from src.vqa import TraCEVQA  # noqa: E402
from src.vqa.data import load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--removal", action="store_true", help="also run Top-K / Random-K trajectory removal")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    model = TraCEVQA.load(a.bundle).to(a.device)
    d = load_split(a.split_dir)
    x, u, m, y = (t.to(a.device) for t in (d.video_feats, d.option_feats, d.option_mask, d.answers))
    preds = torch.cat([model.predict(x[i:i + a.batch_size], u[i:i + a.batch_size], m[i:i + a.batch_size])
                       for i in range(0, len(y), a.batch_size)])
    res = {"n": len(y), "top1": float((preds == y).float().mean())}
    print(f"[VQA] {a.bundle}: top1={res['top1'] * 100:.2f}% (n={res['n']})")
    if a.removal:
        res["removal"] = vqa_removal(model, x, u, y, m, ks=a.ks, seed=a.seed, batch_size=a.batch_size)
        for k, v in res["removal"].items():
            if isinstance(v, dict):
                print(f"  {k:>10}: flip={v['flip_rate'] * 100:5.1f}%  acc={v['acc'] * 100:5.1f}%")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
