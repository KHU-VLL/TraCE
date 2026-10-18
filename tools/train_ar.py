"""Train the TraCE action-recognition head on cached features and save a bundle.

    python tools/train_ar.py --config configs/ar/ntu_mutual.yaml \
        --features_dir features/ar/ntu_mutual --trajectory_labels concepts/ntu_mutual/trajectory_labels \
        --iv2_ckpt pretrained/InternVideo2-stage2_1b-224p-f4.pt --bert_path pretrained/bert-large-uncased \
        --out_dir runs/ar/ntu_mutual

Inputs
  features_dir/{train,val}_videomae.pt      VideoMAE features (tools/extract_ar_features.py)
  features_dir/{train,val}_internvideo2.pt  InternVideo2 video features (pseudo-labels only)
  trajectory_labels/{train,val}.pkl         soft trajectory labels (tools/discovery/trajectory_soft_labels.py)
  object / scene candidate vocabularies     (config: object_candidates, scene_candidates)

Concept text embeddings ("A video of {concept}") are computed with
InternVideo2 unless ``--concept_text_feats`` points to a cached file
({"object": Tensor, "scene": Tensor} in candidate order).
"""
import argparse
import json
import os
import pickle
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ar.data import load_features, read_annotation, read_class_list  # noqa: E402
from src.ar.model import TraCEAR  # noqa: E402
from src.ar import train as T  # noqa: E402
from src.concepts import ConceptVocab, trajectory_names  # noqa: E402

DEFAULTS = dict(seed=456, proj_steps=3000, proj_batch_size=50000, clip_cutoff=0.2,
                interpretability_cutoff=0.3, lam=0.0007, n_iters=30000, saga_batch_size=128,
                concept_prompt="A video of {}", text_batch_size=64)


def read_lines(p):
    with open(p) as f:
        return f.read().split("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--trajectory_labels", required=True, help="dir with train.pkl / val.pkl")
    ap.add_argument("--annotation_dir", default="", help="overrides config annotation_dir")
    ap.add_argument("--iv2_ckpt", default="")
    ap.add_argument("--bert_path", default="")
    ap.add_argument("--concept_text_feats", default="", help="cached concept text embeddings (.pt)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = {**DEFAULTS, **yaml.safe_load(f)}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel = lambda p: p if os.path.isabs(p) else os.path.join(root, p)
    ann = a.annotation_dir or rel(cfg["annotation_dir"])
    os.makedirs(a.out_dir, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    T.setup_seed(cfg["seed"])
    classes = read_class_list(os.path.join(ann, "class_list.txt"))
    tr_v, tr_y = read_annotation(os.path.join(ann, "train.csv"))
    va_v, va_y = read_annotation(os.path.join(ann, "val.csv"))
    X = load_features(os.path.join(a.features_dir, "train_videomae.pt"), tr_v)
    Xv = load_features(os.path.join(a.features_dir, "val_videomae.pt"), va_v)
    log(f"features: train {tuple(X.shape)} val {tuple(Xv.shape)}")

    # pseudo-label inputs for object / scene
    words = {g: read_lines(rel(cfg[f"{g}_candidates"])) for g in ("object", "scene")}
    if a.concept_text_feats:
        tfeat = torch.load(a.concept_text_feats, map_location="cpu", weights_only=True)
    else:
        from src.features.internvideo2 import encode_concepts, load_internvideo2
        iv2 = load_internvideo2(a.iv2_ckpt, a.bert_path, a.device)
        tfeat = {g: encode_concepts(iv2, w, cfg["concept_prompt"], cfg["text_batch_size"]).cpu()
                 for g, w in words.items()}
        torch.save(tfeat, os.path.join(a.out_dir, "concept_text_feats.pt"))
        del iv2
        torch.cuda.empty_cache()
    V = load_features(os.path.join(a.features_dir, "train_internvideo2.pt"), tr_v)
    Vv = load_features(os.path.join(a.features_dir, "val_internvideo2.pt"), va_v)

    groups, src, Wc, mean, std, report = {}, {}, {}, {}, {}, {}
    # 1) trajectory
    with open(os.path.join(a.trajectory_labels, "train.pkl"), "rb") as f:
        Ytr = T.labels_from_pkl(pickle.load(f), tr_v)
    with open(os.path.join(a.trajectory_labels, "val.pkl"), "rb") as f:
        Yva = T.labels_from_pkl(pickle.load(f), va_v)
    r = T.train_trajectory_layer(X, Xv, Ytr, Yva, cfg["proj_steps"], cfg["proj_batch_size"], device=a.device, log=log)
    Mt = r["W"].shape[0]
    groups["trajectory"], src["trajectory"] = trajectory_names(Mt), list(range(Mt))
    Wc["trajectory"], mean["trajectory"], std["trajectory"] = r["W"], r["mean"], r["std"]
    report["trajectory"] = {"concepts": Mt, "best_step": r["best_step"]}

    # 2) object, scene
    for g in ("object", "scene"):
        sim, simv = T.concept_similarity(V, tfeat[g]), T.concept_similarity(Vv, tfeat[g])
        act = T.activation_filter(sim, cfg["clip_cutoff"])
        idx = torch.where(act)[0]
        log(f"[{g}] activation cutoff {cfg['clip_cutoff']}: {len(words[g])} -> {len(idx)} concepts")
        r = T.train_text_concept_layer(X, Xv, sim[:, idx], simv[:, idx], cfg["interpretability_cutoff"],
                                       cfg["proj_steps"], cfg["proj_batch_size"], device=a.device, name=g, log=log)
        kept = idx[r["keep"]]
        groups[g] = [words[g][int(i)] for i in kept]
        src[g] = [int(i) for i in kept]
        Wc[g], mean[g], std[g] = r["W"], r["mean"], r["std"]
        report[g] = {"candidates": len(words[g]), "after_activation_cutoff": len(idx), "concepts": len(kept),
                     "best_step": r["best_step"]}

    # 3) sparse classifier
    vocab = ConceptVocab.from_groups(groups, src)
    z = torch.cat([T.standardize(X, Wc[g], mean[g], std[g]) for g in vocab.groups], 1)
    zv = torch.cat([T.standardize(Xv, Wc[g], mean[g], std[g]) for g in vocab.groups], 1)
    t0 = time.time()
    c = T.train_sparse_classifier(z, torch.tensor(tr_y), zv, torch.tensor(va_y), len(classes), cfg["lam"],
                                  n_iters=cfg["n_iters"], batch_size=cfg["saga_batch_size"], device=a.device)
    log(f"[classifier] lam={c['lam']:.6g} nonzero={c['nonzero']} val acc={c['metrics']['acc_val']:.4f} "
        f"({time.time() - t0:.0f}s)")

    meta = {"dataset": cfg.get("dataset"), "backbone": cfg.get("backbone", {}),
            "activation": "standardized raw concept score (no sigmoid)",
            "training": {k: cfg[k] for k in DEFAULTS}, "report": report,
            "training_log": {"val_acc": c["metrics"]["acc_val"], "glm_saga_lam": c["lam"],
                             "nonzero_weights": c["nonzero"]}}
    model = TraCEAR(vocab, classes, Wc, mean, std, c["W"], c["b"], meta)
    model.save(a.out_dir)
    with open(os.path.join(a.out_dir, "train_log.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"saved bundle to {a.out_dir}: {vocab.group_sizes()}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
