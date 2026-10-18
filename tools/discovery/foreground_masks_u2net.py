"""Per-frame binary foreground masks with U^2-Net.

U^2-Net (Qin et al., github.com/xuebinqin/U-2-Net, Apache-2.0) is NOT bundled. Clone
it, download ``u2net.pth`` and pass ``--u2net_root`` / ``--weights``:

    python tools/discovery/foreground_masks_u2net.py --config configs/discovery/trajectory_ntu_mutual.yaml \
        --u2net_root third_party_src/U-2-Net --weights weights/u2net.pth \
        --video_root data/videos/ntu --video_list annotations/ntu_mutual/train.csv \
        --output_dir data/trajectory/ntu_mutual/masks

Every decoded frame (optionally short-side resized exactly like the flow input) is
converted to RGB, resized to 320x320 (bilinear), scaled to [0, 1] and
ImageNet-normalised; the first U^2-Net side output d1 is min-max normalised per
frame, bilinearly upsampled to the frame size and thresholded at 0.5.

Output: ``<output_dir>/<relative video dir>/<video stem>/frame_XXXXXX.{npz,npy}``,
one mask per video frame (frame index = position in the decoded stream).
``npz`` stores ``{"packed": np.packbits(mask.ravel()), "shape": (H, W)}``; ``npy`` a uint8 0/1 array.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True  # never write caches into the external U-2-Net checkout
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools", "discovery"))
from extract_flow_waft import read_video_list, resize_short_edge  # noqa: E402
from src.discovery.trajectory.config import parse_args_with_config  # noqa: E402
from src.discovery.trajectory.masks import save_mask_packed  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_u2net(u2net_root: str, weights: str, device: str):
    sys.path.insert(0, os.path.abspath(u2net_root))
    from model import U2NET

    net = U2NET(3, 1)
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    return net.to(device).eval()


@torch.no_grad()
def predict_masks(net, frames_bgr: list[np.ndarray], device: str) -> np.ndarray:
    """``[B, H, W]`` uint8 masks for a batch of same-size BGR frames."""
    h, w = frames_bgr[0].shape[:2]
    batch = []
    for frame in frames_bgr:
        rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 320), interpolation=cv2.INTER_LINEAR)
        img = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        batch.append(torch.from_numpy(np.transpose(img, (2, 0, 1))))
    x = torch.stack(batch).to(device)
    pred = net(x)[0][:, 0:1].float()  # d1, (B, 1, 320, 320)
    flat = pred.view(len(batch), -1)
    mi, ma = flat.min(dim=1, keepdim=True)[0], flat.max(dim=1, keepdim=True)[0]
    pred = ((flat - mi) / (ma - mi).clamp(min=1e-8)).view(len(batch), 1, 320, 320)
    pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
    return (pred[:, 0].cpu().numpy() >= 0.5).astype(np.uint8)


def process_video(net, src: Path, out_dir: Path, args) -> int:
    ext = f".{args.format}"
    cap = cv2.VideoCapture(str(src))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if out_dir.is_dir() and n_frames > 0 and sum(1 for f in out_dir.iterdir() if f.suffix == ext) >= n_frames:
        cap.release()
        return -1  # already done
    out_dir.mkdir(parents=True, exist_ok=True)
    idx, done = 0, False
    while not done:
        frames = []
        while len(frames) < args.batch_size:
            ok, frame = cap.read()
            if not ok:
                done = True
                break
            frames.append(resize_short_edge(frame, args.resize_short))
        if not frames:
            break
        for mask in predict_masks(net, frames, args.device):
            path = out_dir / f"frame_{idx:06d}{ext}"
            if args.format == "npz":
                save_mask_packed(mask, path)
            else:
                np.save(path, mask)
            idx += 1
    cap.release()
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--u2net_root", help="path to a clone of the U-2-Net repository")
    ap.add_argument("--weights", help="u2net.pth")
    ap.add_argument("--video_root")
    ap.add_argument("--video_list", nargs="*", default=None)
    ap.add_argument("--output_dir")
    ap.add_argument("--resize_short", type=int, default=-1, help="must match the flow extraction setting")
    ap.add_argument("--format", choices=("npz", "npy"), default="npz")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    args = parse_args_with_config(ap, "masks", required=("u2net_root", "weights", "video_root", "output_dir"))

    video_root, out_root = Path(args.video_root), Path(args.output_dir)
    videos = read_video_list(args.video_list, video_root)[args.shard_index::args.num_shards]
    net = load_u2net(args.u2net_root, args.weights, args.device)
    for i, rel in enumerate(videos, start=1):
        src = video_root / rel
        if not src.exists():
            print(f"[{i}/{len(videos)}] missing video {src}")
            continue
        t0 = time.time()
        n = process_video(net, src, out_root / Path(rel).parent / Path(rel).stem, args)
        status = "already done" if n < 0 else f"{n} masks"
        print(f"[{i}/{len(videos)}] {rel}: {status} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
