"""Train the TraCE VQA head on cached InternVideo2 features and save a bundle.

    python tools/train_vqa.py --config configs/vqa/ntu_mutual.yaml \
        --train_dir features/vqa/ntu_mutual/train --val_dir features/vqa/ntu_mutual/val \
        --trajectory_labels concepts/ntu_mutual/trajectory_labels \
        --iv2_ckpt pretrained/InternVideo2-stage2_1b-224p-f4.pt --bert_path pretrained/bert-large-uncased \
        --out_dir runs/vqa/ntu_mutual

Feature splits are written by tools/extract_vqa_features.py. Object/scene
concept embeddings ("A video of {concept}") are computed with InternVideo2
unless ``--concept_text_feats`` points to a cached {"object", "scene"} file.
"""
import argparse
import json
import os
import pickle
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ar.train import setup_seed  # noqa: E402
from src.concepts import ConceptVocab, trajectory_names  # noqa: E402
from src.vqa import train as T  # noqa: E402
from src.vqa.data import load_split  # noqa: E402
from src.vqa.model import TraCEVQA  # noqa: E402

DEFAULTS = dict(seed=42, topk=5, lr=1e-3, stage1_steps=5000, stage1_batch_size=256, epochs=100, batch_size=256,
                lambda_qa=0.2, tau=0.07, align_warmup_steps=50, text_steps=5000, text_batch_size=256,
                use_half=True, concept_prompt="A video of {}")


def read_lines(p):
    with open(p) as f:
        return [w.strip() for w in f.read().split("\n") if w.strip()]


def video_dict(split_dir):
    v = torch.load(os.path.join(split_dir, "video_features.pt"), map_location="cpu", weights_only=True)
    return {T._stem(n): f.float() for n, f in zip(v["videos"], v["features"])}


def label_dict(pkl):
    with open(pkl, "rb") as f:
        items = pickle.load(f)
    return [it["video_name"] for it in items], {T._stem(it["video_name"]): list(it["attribute_label"]) for it in items}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--trajectory_labels", default="", help="dir with train.pkl / val.pkl")
    ap.add_argument("--iv2_ckpt", default="")
    ap.add_argument("--bert_path", default="")
    ap.add_argument("--concept_text_feats", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = {**DEFAULTS, **yaml.safe_load(f)}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel = lambda p: p if os.path.isabs(p) else os.path.join(root, p)
    os.makedirs(a.out_dir, exist_ok=True)
    lines = []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    setup_seed(cfg["seed"])
    tr = T.research_order(load_split(a.train_dir))
    va = T.research_order(load_split(a.val_dir))
    log(f"questions: train {len(tr.answer)} val {len(va.answer)}")
    groups, src, Wv, Wt, tau, report = {}, {}, {}, {}, {}, {}

    otr, ltr = label_dict(os.path.join(a.trajectory_labels, "train.pkl"))
    ova, lva = label_dict(os.path.join(a.trajectory_labels, "val.pkl"))
    s = T.train_trajectory_block(tr, va, video_dict(a.train_dir), video_dict(a.val_dir), ltr, lva, otr, ova,
                                 cfg["stage1_steps"], cfg["stage1_batch_size"], cfg["epochs"], cfg["batch_size"],
                                 cfg["lr"], cfg["topk"], cfg["lambda_qa"], cfg["tau"], cfg["align_warmup_steps"],
                                 a.device, log)
    Mt = s["Wv"].shape[0]
    groups["trajectory"], src["trajectory"] = trajectory_names(Mt), list(range(Mt))
    Wv["trajectory"], Wt["trajectory"], tau["trajectory"] = s["Wv"], s["Wt"], s["log_tau"]
    report["trajectory"] = {"concepts": Mt, "best_epoch": s["epoch"], "val_acc_block": s["val_acc"]}

    words = {g: read_lines(rel(cfg[f"{g}_candidates"])) for g in ("object", "scene")}
    if a.concept_text_feats:
        tfeat = torch.load(a.concept_text_feats, map_location="cpu", weights_only=True)
    else:
        from src.features.internvideo2 import encode_concepts, load_internvideo2
        dtype = torch.float16 if cfg["use_half"] else torch.float32
        iv2 = load_internvideo2(a.iv2_ckpt, a.bert_path, a.device, dtype=dtype)
        tfeat = {g: torch.nn.functional.normalize(encode_concepts(iv2, words[g], cfg["concept_prompt"]), dim=1).cpu()
                 for g in ("object", "scene")}
        torch.save(tfeat, os.path.join(a.out_dir, "concept_text_feats.pt"))
        del iv2
        torch.cuda.empty_cache()
    for g in ("object", "scene"):
        s = T.train_text_block(tr, va, words[g], tfeat[g], cfg["text_steps"], cfg["text_batch_size"], cfg["lr"],
                               cfg["topk"], cfg["lambda_qa"], cfg["use_half"], a.device, g, log=log)
        groups[g], src[g] = words[g], list(range(len(words[g])))
        Wv[g], Wt[g], tau[g] = s["Wv"], s["Wt"], s["log_tau"]
        report[g] = {"concepts": len(words[g]), "best_step": s["step"]}

    vocab = ConceptVocab.from_groups(groups, src)
    meta = {"dataset": cfg.get("dataset"), "backbone": cfg.get("backbone", {}),
            "training": {k: cfg[k] for k in DEFAULTS}, "report": report}
    model = TraCEVQA(vocab, Wv, Wt, tau, topk=cfg["topk"], meta=meta)
    x, u, m, y = va.video, va.options, va.mask, va.answer
    acc = float((model.predict(x, u, m) == y).float().mean())
    model.meta["training_log"] = {"val_acc": acc}
    model.save(a.out_dir)
    log(f"val acc (Eq. 5, all groups) = {acc:.4f}; saved {a.out_dir}: {vocab.group_sizes()}")
    with open(os.path.join(a.out_dir, "train_log.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
