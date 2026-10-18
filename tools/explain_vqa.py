"""Explain one multiple-choice VQA prediction, optionally with an intervention.

From a video and answer candidates (needs InternVideo2 + BERT)::

    python tools/explain_vqa.py --bundle checkpoints/vqa/ucf101 --video v_PullUps_g01_c01.avi \
        --candidates "PlayingTabla" "PullUps" "Diving" "MilitaryParade" "Rowing" \
        --iv2_ckpt pretrained/InternVideo2-stage2_1b-224p-f4.pt --bert_path pretrained/bert-large-uncased \
        --out_dir outputs/explain_vqa --remove_top_trajectory 3

From a cached feature split (no backbone needed)::

    python tools/explain_vqa.py --bundle checkpoints/vqa/ucf101 --split_dir features/vqa/ucf101/val \
        --index 0 --out_dir outputs/explain_vqa

Only the candidate strings are encoded (the released models never encode the
question text). Outputs ``explanation.json`` and ``matching.png`` (top video
concepts x candidates).
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.explain.visualize import plot_vqa_matrix  # noqa: E402
from src.vqa import TraCEVQA  # noqa: E402
from src.vqa.data import load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--video", default="")
    ap.add_argument("--candidates", nargs="*", default=[])
    ap.add_argument("--iv2_ckpt", default=os.path.join(ROOT, "pretrained", "InternVideo2-stage2_1b-224p-f4.pt"))
    ap.add_argument("--bert_path", default=os.path.join(ROOT, "pretrained", "bert-large-uncased"))
    ap.add_argument("--split_dir", default="")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5, help="video concepts listed per group")
    ap.add_argument("--remove", type=int, nargs="*", default=[], help="concept ids to remove (z := 0)")
    ap.add_argument("--remove_top_trajectory", type=int, default=0,
                    help="remove the K trajectory concepts contributing most to the predicted answer")
    ap.add_argument("--overlay", type=int, default=0,
                    help="animated overlay (GIF + MP4) of the top-N concepts of the predicted answer; trajectory "
                         "concepts are drawn at their Grad-CAM peak (needs --video)")
    ap.add_argument("--gradcam_layer", default="vision_encoder.blocks[38].norm1", help="layer used to place the overlay")
    ap.add_argument("--trajectory_concepts", default="", help="default: checkpoints/trajectory_concepts/<dataset>")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    model = TraCEVQA.load(a.bundle).to(a.device)
    iv2 = frames = None
    if a.video:
        from src.features.internvideo2 import frames_to_tensor, load_internvideo2, read_frames_cv2
        iv2 = load_internvideo2(a.iv2_ckpt, a.bert_path, a.device)
        raw_frames = read_frames_cv2(a.video)
        frames = frames_to_tensor(raw_frames)
        x = iv2.get_vid_feat(frames.to(a.device))[0]
        u = iv2.get_txt_feat(a.candidates)
        cands, answer, src = list(a.candidates), None, {"video": os.path.basename(a.video)}
    else:
        d = load_split(a.split_dir)
        n = int(d.option_mask[a.index].sum())
        x, u = d.video_feats[a.index], d.option_feats[a.index, :n]
        cands, answer = d.options[a.index], int(d.answers[a.index])
        src = {"split_dir": os.path.basename(a.split_dir.rstrip("/")), "index": a.index, "video": d.videos[a.index]}
    x, u = x.to(a.device), u.to(a.device)

    e = model.explain(x, u, candidates=cands)
    p = e.pred
    res = {"input": src, "candidates": cands, "prediction": cands[p], "pred_index": p,
           "answer": None if answer is None else cands[answer],
           "scores": {c: float(s) for c, s in zip(cands, e.scores)},
           "group_scores": {g: {c: float(s) for c, s in zip(cands, v)} for g, v in e.group_scores.items()},
           "group_weight_exp_gamma": {g: float(model.group_weight(g)) for g in model.groups},
           "score_decomposition": {"sum_of_concept_contributions": float(e.contributions[p].sum()),
                                   "model_score": float(e.scores[p]), "bias": 0.0},
           "top_concepts_for_prediction": {
               g: [{"id": i, "name": nme, "contribution": v, "video_activation": float(e.video_activations[i]),
                    "relevance": float(e.text_relevance[p, i])}
                   for i, _, nme, v in model.top_concepts(e.contributions[p], a.topk, g)]
               for g in model.vocab.groups}}
    print(f"prediction: {cands[p]}" + ("" if answer is None else f"  (answer: {cands[answer]})"))
    print(f"score = sum(contrib) {e.contributions[p].sum():.4f} (model {e.scores[p]:.4f}, no bias term)")
    for g, rows in res["top_concepts_for_prediction"].items():
        print(f"  {g}: " + ", ".join(f"{r['name']} ({r['contribution']:.3f})" for r in rows))

    # matrix: the concepts that enter any candidate's top-K, ordered by contribution to the prediction
    used = torch.nonzero(e.contributions.sum(0)).flatten().tolist()
    used = sorted(used, key=lambda i: -float(e.contributions[p, i]))[:15]
    plot_vqa_matrix(e.contributions[:, used].T.numpy(), [model.vocab.label(i) for i in used], cands,
                    f"prediction: {cands[p]}", os.path.join(a.out_dir, "matching.png"))

    ids = list(a.remove)
    if a.remove_top_trajectory:
        ids += [i for i, *_ in model.top_concepts(e.contributions[p], a.remove_top_trajectory, "trajectory")]
    if ids:
        before, after = model.score_with_intervention(x[None], u[None], ids, 0.0)
        res["intervention"] = {
            "space": "raw video-side activation z (before ReLU / normalization), set to 0",
            "removed": [model.vocab.label(i) for i in ids],
            "prediction_before": cands[int(before.argmax())], "prediction_after": cands[int(after.argmax())],
            "scores_before": {c: float(s) for c, s in zip(cands, before[0])},
            "scores_after": {c: float(s) for c, s in zip(cands, after[0])}}
        print(f"intervention on {len(ids)} concepts: {res['intervention']['prediction_before']} -> "
              f"{res['intervention']['prediction_after']}")

    if a.overlay:
        if iv2 is None:
            raise SystemExit("--overlay needs --video")
        import cv2
        import numpy as np
        from src.discovery.trajectory.concepts import load_bundle
        from src.explain.gradcam import iv2_concept_cam
        from src.explain.overlay import concept_overlay_grid, interpolate_cam, save_animation, temporal_resample
        n_show = 4 * frames.shape[1]           # 4 display frames per model frame (segment)
        shown = temporal_resample(np.stack([cv2.resize(f[:, :, ::-1], (224, 224)) for f in raw_frames]), n_show)
        tc_dir = a.trajectory_concepts or os.path.join(ROOT, "checkpoints", "trajectory_concepts",
                                                       model.meta.get("dataset", ""))
        medoids = load_bundle(tc_dir)[0] if os.path.isdir(tc_dir) else None
        items = model.top_concepts(e.contributions[p], a.overlay)
        cams = {i: interpolate_cam(iv2_concept_cam(iv2, model, frames, i, a.gradcam_layer).numpy(), n_show)
                for i, g, *_ in items if g == "trajectory"}
        if medoids is not None and len(medoids) != len(model.vocab.ids("trajectory")):
            medoids = None                    # trajectory metadata of another clustering
        grid = concept_overlay_grid(shown, items, cams, medoids, model.vocab, title=f"Pred: {cands[p]}",
                                    max_start=0.25)
        res["overlay"] = [save_animation(grid, os.path.join(a.out_dir, f"overlay.{ext}"), a.fps)
                          for ext in ("gif", "mp4")]
        res["overlay_concepts"] = [model.vocab.label(i) for i, *_ in items]
    with open(os.path.join(a.out_dir, "explanation.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
