"""Extract InternVideo2 features for a multiple-choice QA json (one split).

    python tools/extract_vqa_features.py --qa_json annotations/ntu_mutual/qa_val.json \
        --format ar_qa --video_root $DATA/nturgb+d_rgb \
        --iv2_ckpt pretrained/InternVideo2-stage2_1b-224p-f4.pt --bert_path pretrained/bert-large-uncased \
        --out_dir features/vqa/ntu_mutual/val

Supported json formats (``--format``):
  ar_qa   {"<i>": {"question", "options", "answer": int, "key": <video path>}}   (tools/data/build_qa_from_ar.py)
  tomato  {"<i>": {"options", "answer": int, "key", "demonstration_type"}}       video = <type>/<key>.mp4
  star    [{"video", "option_text": [...], "answer": <int or option string>}]   (tools/data/build_star_qa.py)

Only the candidate strings are encoded, one text per call, exactly as the
released features were produced. Output: video_features.pt,
text_features.pt, sample_index.json (see src/vqa/data.py).
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.internvideo2 import frames_to_tensor, load_internvideo2, read_frames_cv2  # noqa: E402


def load_samples(path, fmt):
    with open(path) as f:
        data = json.load(f)
    items = data.values() if isinstance(data, dict) else data
    out = []
    for it in items:
        if fmt == "tomato":
            cands, video = it.get("options") or [], f'{it["demonstration_type"]}/{it["key"]}.mp4'
        elif fmt == "ar_qa":
            cands, video = it.get("options") or [], it.get("key")
        elif fmt == "star":
            cands, video = it.get("option_text") or it.get("Option_text") or [], it.get("video")
        else:
            raise ValueError(fmt)
        ans = it.get("answer")
        if isinstance(ans, str):
            ans = cands.index(ans) if ans in cands else None
        elif isinstance(ans, int) and not 0 <= ans < len(cands):
            ans = None
        out.append({"video": video, "candidates": cands, "answer_idx": ans})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa_json", required=True)
    ap.add_argument("--format", choices=["ar_qa", "tomato", "star"], required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--iv2_ckpt", required=True)
    ap.add_argument("--bert_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    samples = load_samples(a.qa_json, a.format)
    model = load_internvideo2(a.iv2_ckpt, a.bert_path, a.device)
    videos = list(dict.fromkeys(s["video"] for s in samples if s["video"] is not None))
    vfeat = {}
    for i in tqdm(range(0, len(videos), a.batch_size), desc="video"):
        batch, names = [], []
        for v in videos[i:i + a.batch_size]:
            frames = read_frames_cv2(os.path.join(a.video_root, v))
            if len(frames) < a.num_frames:
                continue
            batch.append(frames_to_tensor(frames, a.num_frames))
            names.append(v)
        if batch:
            f = model.get_vid_feat(torch.cat(batch).to(a.device)).cpu()
            vfeat.update(zip(names, f))
    texts = list(dict.fromkeys(t for s in samples for t in s["candidates"]))
    tfeat = {t: model.get_txt_feat(t)[0].cpu() for t in tqdm(texts, desc="text")}

    vnames, tnames = sorted(vfeat), sorted(tfeat)
    vidx, tidx = {v: i for i, v in enumerate(vnames)}, {t: i for i, t in enumerate(tnames)}
    index, dropped = [], 0
    for s in samples:
        if s["video"] not in vidx or not s["candidates"] or s["answer_idx"] is None:
            dropped += 1
            continue
        index.append({"video": s["video"], "video_idx": vidx[s["video"]],
                      "option_text_idxs": [tidx[t] for t in s["candidates"]], "answer_idx": s["answer_idx"]})
    os.makedirs(a.out_dir, exist_ok=True)
    torch.save({"videos": vnames, "features": torch.stack([vfeat[v] for v in vnames])},
               os.path.join(a.out_dir, "video_features.pt"))
    torch.save({"texts": tnames, "features": torch.stack([tfeat[t] for t in tnames])},
               os.path.join(a.out_dir, "text_features.pt"))
    with open(os.path.join(a.out_dir, "sample_index.json"), "w") as f:
        json.dump({"num_samples": len(index), "dropped_samples": dropped, "samples": index}, f, indent=2,
                  ensure_ascii=False)
    print(f"saved {len(index)} questions ({dropped} dropped) to {a.out_dir}")


if __name__ == "__main__":
    main()
