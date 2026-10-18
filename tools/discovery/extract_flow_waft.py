"""Dense forward optical flow between all consecutive frames with WAFT.

WAFT (princeton-vl/WAFT, BSD-3-Clause) is NOT bundled. Clone it and download its
weights yourself, then point ``--waft_root`` at the clone:

    git clone https://github.com/princeton-vl/WAFT
    # weights: tar-c-t.pth (WAFT, config/a1/tar-c-t.json) and depth_anything_v2_vits.pth (Depth-Anything-V2-Small)

    python tools/discovery/extract_flow_waft.py --config configs/discovery/trajectory_ntu_mutual.yaml \
        --waft_root third_party_src/WAFT --ckpt weights/tar-c-t.pth --dav2_ckpt weights/depth_anything_v2_vits.pth \
        --video_root data/videos/ntu --video_list annotations/ntu_mutual/train.csv annotations/ntu_mutual/val.csv \
        --output_dir data/trajectory/ntu_mutual/flow

Frames are decoded with OpenCV (BGR -> RGB), optionally resized so that the short
side equals ``--resize_short`` (aspect ratio kept, rounded down to even sizes,
bilinear; only applied when the short side is larger), and fed to WAFT as float
0-255 tensors. Output: ``<output_dir>/<relative video path without extension>.npy``,
float32 ``[T-1, H, W, 2]`` (x, y displacement in pixels of the resized frame).
Existing outputs are skipped.

Video list: text/CSV files whose first column (comma or whitespace separated) is
the video path relative to ``--video_root``; or a JSON file (a list of
``{"video": ...}`` entries as in STAR, or a dict of ``{"key", "demonstration_type"}``
entries as in TOMATO, mapped to ``<demonstration_type>/<key>.mp4``). Without a
list, all videos under ``--video_root`` are used.
"""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.dont_write_bytecode = True  # never write caches into the external WAFT checkout
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.discovery.trajectory.config import parse_args_with_config  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------- video list
def read_video_list(list_paths: list[str], video_root: Path) -> list[str]:
    if not list_paths:
        return sorted(str(p.relative_to(video_root)) for p in video_root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    rels: list[str] = []
    for lp in list_paths:
        if lp.endswith(".json"):
            with open(lp, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.values() if isinstance(data, dict) else data
            for e in entries:
                if "video" in e:
                    rels.append(str(e["video"]))
                else:
                    rels.append(f"{e['demonstration_type']}/{e['key']}.mp4")
        else:
            with open(lp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        rels.append(line.replace(",", " ").split()[0])
    return list(dict.fromkeys(rels))  # unique, order kept


# ---------------------------------------------------------------- frames
def resize_short_edge(frame: np.ndarray, short_edge: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if short_edge <= 0 or min(h, w) <= short_edge:
        return frame
    if h < w:
        new_h, new_w = short_edge, int(w * short_edge / h)
    else:
        new_w, new_h = short_edge, int(h * short_edge / w)
    new_h -= new_h % 2
    new_w -= new_w % 2
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def read_frames(video_path: Path, resize_short: int = -1) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(resize_short_edge(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), resize_short))
    cap.release()
    return frames


# ---------------------------------------------------------------- WAFT
@contextlib.contextmanager
def _redirect_dav2_checkpoint(dav2_ckpt: str | None):
    """WAFT loads Depth-Anything-V2 from the relative path ``depth-anything-ckpts/...``;
    redirect that load to ``dav2_ckpt`` without modifying the WAFT checkout."""
    if not dav2_ckpt:
        yield
        return
    original = torch.load

    def patched(f, *args, **kwargs):
        if isinstance(f, (str, os.PathLike)) and str(f).replace("\\", "/").startswith("depth-anything-ckpts/"):
            f = dav2_ckpt
        return original(f, *args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def load_waft(waft_root: str, cfg: str, ckpt: str, dav2_ckpt: str | None, device: str):
    waft_root = os.path.abspath(waft_root)
    sys.path.insert(0, waft_root)
    from config.parser import json_to_args
    from inference_tools import InferenceWrapper
    from model import fetch_model
    from utils.utils import load_ckpt

    cfg_path = cfg if os.path.isabs(cfg) else os.path.join(waft_root, cfg)
    model_args = json_to_args(cfg_path)
    with _redirect_dav2_checkpoint(dav2_ckpt):
        model = fetch_model(model_args)
    load_ckpt(model, ckpt)
    model = model.to(device).eval()
    return InferenceWrapper(model, scale=0, train_size=model_args.image_size)


@torch.no_grad()
def video_flow(model, frames: list[np.ndarray], device: str, batch_size: int, amp_dtype: str) -> np.ndarray:
    """Flow for all pairs (i, i+1); returns float32 ``[T-1, H, W, 2]``."""
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()  # [T, 3, H, W], 0-255
    flows = []
    for start in range(0, len(frames) - 1, batch_size):
        idx = torch.arange(start, min(start + batch_size, len(frames) - 1))
        img1, img2 = video[idx].to(device), video[idx + 1].to(device)
        if amp_dtype == "fp32":
            out = model.calc_flow(img1, img2)
        else:
            dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
            with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
                out = model.calc_flow(img1, img2)
        flows.append(out["flow"][-1].float().permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(flows, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waft_root", help="path to a clone of princeton-vl/WAFT")
    ap.add_argument("--cfg", default="config/a1/tar-c-t.json", help="WAFT config (relative to --waft_root)")
    ap.add_argument("--ckpt", help="WAFT weights, e.g. tar-c-t.pth")
    ap.add_argument("--dav2_ckpt", default=None, help="Depth-Anything-V2-Small weights (depth_anything_v2_vits.pth)")
    ap.add_argument("--video_root")
    ap.add_argument("--video_list", nargs="*", default=None)
    ap.add_argument("--output_dir")
    ap.add_argument("--resize_short", type=int, default=-1, help="-1 keeps the original resolution")
    ap.add_argument("--batch_size", type=int, default=1, help="frame pairs per forward pass")
    ap.add_argument("--amp_dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=None, help="use only the first N frames of each video (debugging)")
    args = parse_args_with_config(ap, "flow", required=("waft_root", "ckpt", "video_root", "output_dir"))

    video_root, out_root = Path(args.video_root), Path(args.output_dir)
    videos = read_video_list(args.video_list, video_root)[args.shard_index::args.num_shards]
    model = load_waft(args.waft_root, args.cfg, args.ckpt, args.dav2_ckpt, args.device)
    print(f"{len(videos)} videos, device={args.device}, resize_short={args.resize_short}, amp={args.amp_dtype}")

    for i, rel in enumerate(videos, start=1):
        out_path = out_root / Path(rel).with_suffix(".npy")
        src = video_root / rel
        if out_path.exists():
            continue
        if not src.exists():
            print(f"[{i}/{len(videos)}] missing video {src}")
            continue
        t0 = time.time()
        frames = read_frames(src, args.resize_short)[: args.max_frames]
        if len(frames) < 2:
            print(f"[{i}/{len(videos)}] {rel}: fewer than 2 frames, skipped")
            continue
        flow = video_flow(model, frames, args.device, args.batch_size, args.amp_dtype)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.stem + ".tmp.npy")
        np.save(tmp, flow)
        os.replace(tmp, out_path)
        print(f"[{i}/{len(videos)}] {rel}: {flow.shape} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
