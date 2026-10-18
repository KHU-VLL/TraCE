"""Explain one action-recognition prediction, optionally with an intervention.

From a video (needs the VideoMAE backbone of the bundle)::

    python tools/explain_ar.py --bundle checkpoints/ar/chiral_ssv2 --video clip.mp4 \
        --videomae_ckpt pretrained/videomae_ssv2_finetune.pth --out_dir outputs/explain \
        --remove_top_trajectory 3 --overlay 3

From a cached feature row (no backbone needed)::

    python tools/explain_ar.py --bundle checkpoints/ar/chiral_ssv2 \
        --features features/ar/chiral_ssv2/val_videomae.pt --index 0 --out_dir outputs/explain

Outputs ``explanation.json`` (prediction, per-concept contributions, bias,
intervention result) and ``contributions.png``; with ``--overlay`` also
``overlay.gif`` / ``overlay.mp4`` (paper-figure style trajectory overlay).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.ar import TraCEAR  # noqa: E402
from src.ar.data import load_features  # noqa: E402
from src.explain.visualize import plot_contributions  # noqa: E402


def parse_set(items):
    out = {}
    for s in items or []:
        k, v = s.split("=")
        out[int(k)] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--video", default="")
    ap.add_argument("--videomae_ckpt", default="", help="default: the backbone named in the bundle, from checkpoints/backbones/ or pretrained/")
    ap.add_argument("--sampling", default="", help="segment | dense (default: from bundle meta)")
    ap.add_argument("--features", default="")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--target", type=int, default=None, help="class to explain (default: predicted)")
    ap.add_argument("--topk", type=int, default=5, help="top concepts listed per group")
    ap.add_argument("--remove", type=int, nargs="*", default=[], help="concept ids to remove (z := 0)")
    ap.add_argument("--set", nargs="*", default=[], help="id=value pairs: set standardized activations")
    ap.add_argument("--remove_top_trajectory", type=int, default=0,
                    help="remove the K trajectory concepts with the largest contribution")
    ap.add_argument("--overlay", type=int, default=0,
                    help="animated overlay (GIF + MP4) of the top-N concepts; trajectory concepts are drawn "
                         "at their Grad-CAM peak (needs --video)")
    ap.add_argument("--gradcam_layer", default="blocks[11].norm1", help="layer used to place the overlay")
    ap.add_argument("--trajectory_concepts", default="", help="default: checkpoints/trajectory_concepts/<dataset>")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    model = TraCEAR.load(a.bundle).to(a.device)
    backbone = clip = None
    if a.video:
        from src.features import videomae
        sampling = a.sampling or model.meta.get("backbone", {}).get("sampling", "segment")
        ckpt = a.videomae_ckpt or videomae.find_backbone(model.meta["backbone"]["checkpoint_file"], ROOT)
        backbone = videomae.load_videomae(ckpt, a.device)
        clip = videomae.load_clip(a.video, sampling)
        x = videomae.encode(backbone, clip[None])[0]
        src = {"video": os.path.basename(a.video), "sampling": sampling}
    else:
        x = load_features(a.features)[a.index]
        src = {"features": os.path.basename(a.features), "index": a.index}
    x = x.to(a.device)

    e = model.explain(x, a.target)
    recon = float(e.contributions.sum() + e.bias)
    probs = torch.softmax(e.logits, 0)
    top_cls = torch.argsort(e.logits, descending=True)[:5]
    res = {
        "input": src, "prediction": model.classes[e.pred], "pred_index": e.pred,
        "explained_class": model.classes[e.target],
        "top_classes": [{"class": model.classes[int(i)], "logit": float(e.logits[i]), "prob": float(probs[i])}
                        for i in top_cls],
        "score_decomposition": {"sum_of_concept_contributions": float(e.contributions.sum()),
                                "bias": e.bias, "reconstructed_logit": recon,
                                "model_logit": float(e.logits[e.target])},
        "top_concepts": {g: [{"id": i, "name": n, "contribution": v,
                              "activation": float(e.activations[i])}
                             for i, _, n, v in model.top_concepts(e.contributions, a.topk, g)]
                         for g in model.vocab.groups},
    }
    print(f"prediction: {res['prediction']}  (logit {e.logits[e.pred]:.3f}, p={probs[e.pred]:.3f})")
    print(f"logit[{model.classes[e.target]}] = sum(contrib) {e.contributions.sum():.4f} + bias {e.bias:.4f} "
          f"= {recon:.4f} (model {float(e.logits[e.target]):.4f})")
    for g, rows in res["top_concepts"].items():
        print(f"  {g}: " + ", ".join(f"{r['name']} ({r['contribution']:+.3f})" for r in rows))
    top_all = model.top_concepts(e.contributions, 12)
    plot_contributions(top_all, f"{res['explained_class']}", os.path.join(a.out_dir, "contributions.png"),
                       extra={"bias": e.bias})

    # intervention on the standardized activations
    edits = {i: 0.0 for i in a.remove}
    if a.remove_top_trajectory:
        edits.update({i: 0.0 for i, *_ in model.top_concepts(e.contributions, a.remove_top_trajectory, "trajectory")})
    edits.update(parse_set(a.set))
    if edits:
        z = e.activations[None].to(a.device)
        z2 = z.clone()
        for i, v in edits.items():
            z2 = model.intervene(z2, [i], v)
        after = model.classify(z2)[0].cpu()
        res["intervention"] = {
            "space": "standardized activation z (classifier input); 0 = training-set mean score",
            "edits": [{"id": i, "concept": model.vocab.label(i), "before": float(z[0, i]), "after": v}
                      for i, v in edits.items()],
            "prediction_before": model.classes[e.pred], "prediction_after": model.classes[int(after.argmax())],
            "logit_before": {model.classes[int(i)]: float(e.logits[i]) for i in top_cls},
            "logit_after": {model.classes[int(i)]: float(after[i]) for i in top_cls},
        }
        print(f"intervention on {len(edits)} concepts: {res['prediction']} -> {res['intervention']['prediction_after']}")

    if a.overlay:
        if backbone is None:
            raise SystemExit("--overlay needs --video")
        from src.discovery.trajectory.concepts import load_bundle
        from src.explain.gradcam import videomae_concept_cam
        from src.explain.overlay import concept_overlay_grid, save_animation
        from src.features.videomae import IMAGENET_MEAN, IMAGENET_STD
        frames = (clip * torch.tensor(IMAGENET_STD)[:, None, None, None] +
                  torch.tensor(IMAGENET_MEAN)[:, None, None, None]).clamp(0, 1)
        frames = (frames.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
        tc_dir = a.trajectory_concepts or os.path.join(ROOT, "checkpoints", "trajectory_concepts",
                                                       model.meta.get("dataset", ""))
        medoids = load_bundle(tc_dir)[0] if os.path.isdir(tc_dir) else None
        items = model.top_concepts(e.contributions, a.overlay)
        cams = {i: videomae_concept_cam(backbone, model, clip, i, a.gradcam_layer).numpy()
                for i, g, *_ in items if g == "trajectory"}
        grid = concept_overlay_grid(frames, items, cams, medoids, model.vocab,
                                    title=f"Pred: {model.classes[e.pred]}")
        res["overlay"] = [save_animation(grid, os.path.join(a.out_dir, f"overlay.{ext}"), a.fps)
                          for ext in ("gif", "mp4")]
        res["overlay_concepts"] = [model.vocab.label(i) for i, *_ in items]
    with open(os.path.join(a.out_dir, "explanation.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
