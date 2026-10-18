"""Extract action-recognition features for every video of an annotation csv.

Writes ``<out_dir>/<split>_videomae.pt`` (VideoMAE-B/16, 768-d, used by the
concept layer and classifier) and, with ``--iv2_ckpt``, also
``<out_dir>/<split>_internvideo2.pt`` (InternVideo2 512-d, used only to build
object/scene pseudo-labels during training).

    python tools/extract_ar_features.py --annotation annotations/ntu_mutual/val.csv \
        --video_root $DATA/nturgb+d_rgb --videomae_ckpt pretrained/videomae_ntu_mutual.pth \
        --sampling segment --out_dir features/ar/ntu_mutual --split val
"""
import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ar.data import read_annotation, save_features  # noqa: E402
from src.features import videomae  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--videomae_ckpt", required=True)
    ap.add_argument("--sampling", choices=["segment", "dense"], required=True,
                    help="segment: NTU RGB+D Mutual / Chiral SSv2 / KTH; dense: UCF-101")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", required=True, help="output file prefix, e.g. train or val")
    ap.add_argument("--iv2_ckpt", default="", help="also extract InternVideo2 features for pseudo-labels")
    ap.add_argument("--bert_path", default="")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    videos, _ = read_annotation(a.annotation)
    os.makedirs(a.out_dir, exist_ok=True)
    vm = videomae.load_videomae(a.videomae_ckpt, a.device)
    iv2 = None
    if a.iv2_ckpt:
        from src.features.internvideo2 import load_internvideo2
        iv2 = load_internvideo2(a.iv2_ckpt, a.bert_path, a.device)

    feats, ivfeats = [], []
    for i in tqdm(range(0, len(videos), a.batch_size)):
        clips = torch.stack([videomae.load_clip(os.path.join(a.video_root, v), a.sampling)
                             for v in videos[i:i + a.batch_size]])
        feats.append(videomae.encode(vm, clips))
        if iv2 is not None:
            frames = torch.cat([videomae.iv2_frames_from_clip(c) for c in clips])
            ivfeats.append(iv2.get_vid_feat(frames.to(iv2.device, iv2.dtype)).cpu())
    save_features(os.path.join(a.out_dir, f"{a.split}_videomae.pt"), videos, torch.cat(feats))
    if iv2 is not None:
        save_features(os.path.join(a.out_dir, f"{a.split}_internvideo2.pt"), videos, torch.cat(ivfeats))
    print(f"saved {len(videos)} videos to {a.out_dir}")


if __name__ == "__main__":
    main()
