"""Query Qwen3-VL-8B-Instruct once per training video for object or scene candidates.

Writes one JSON line per video ({"video", "status", "output", "used_max_frames",
"used_total_pixels", "error"}) to ``<out_dir>/<prompt>_results.jsonl``. Re-running
the command resumes (videos already in the file are skipped). Feed the result to
tools/discovery/filter_text_concepts.py to get the vocabulary.

Use training videos only. Video lists:
  * AR datasets: ``annotations/<dataset>/train.csv`` (first column, relative to --video_root)
  * TOMATO: ``annotations/tomato/direction_ood_train.json`` (videos resolve to <video_root>/<type>/<key>.mp4)
  * STAR: ``STAR_preprocessed_train.json`` (field "video")

Example:
    python tools/discovery/generate_text_candidates_qwen3vl.py \
        --config configs/discovery/text_ntu_mutual.yaml --prompt object_v1 \
        --model_path Qwen/Qwen3-VL-8B-Instruct --video_root <NTU>/nturgb+d_rgb --out_dir outputs/text_concepts/ntu_mutual
"""
import argparse
import csv
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.discovery.text.prompts import get_prompt  # noqa: E402
from src.discovery.text.qwen3vl import (LADDER, MIN_PIXELS, QWEN3VL_MODEL_ID,  # noqa: E402
                                              extract_frames, load_model, query_video, subsample)


def read_video_list(path: str):
    """Video names (unique, in file order) from a split csv, a TraCE/TOMATO QA dict or a STAR QA list."""
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        items = data.values() if isinstance(data, dict) else data
        names = []
        for it in items:
            if "demonstration_type" in it:  # TOMATO
                names.append(f"{it['demonstration_type']}/{it['key']}.mp4")
            else:
                names.append(it.get("video") or it.get("key"))
    else:
        with open(path, newline="") as f:
            names = [row[0].split()[0] for row in csv.reader(f) if row and row[0].strip()]
    return list(dict.fromkeys(n for n in names if n))


def load_config(args):
    if not args.config:
        return args
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in cfg.get("generation", {}).items():
        if hasattr(args, k) and getattr(args, k) in (None, ""):
            setattr(args, k, v)
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None, help="configs/discovery/text_<dataset>.yaml (fills unset args)")
    ap.add_argument("--prompt", required=True, choices=["object_v1", "scene_v1"])
    ap.add_argument("--video_list", default=None, help="train.csv / QA json listing the training videos")
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", default=None, help=f"HF id or local snapshot (default {QWEN3VL_MODEL_ID})")
    ap.add_argument("--revision", default=None, help="HF revision; the paper used 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--sample_fps", type=float, default=None, help="default 2.0")
    ap.add_argument("--max_new_tokens", type=int, default=None, help="default 256")
    args = load_config(ap.parse_args())
    args.model_path = args.model_path or QWEN3VL_MODEL_ID
    args.sample_fps = float(args.sample_fps or 2.0)
    args.max_new_tokens = int(args.max_new_tokens or 256)
    if not args.video_list:
        ap.error("--video_list is required (or set generation.video_list in --config)")

    import torch

    prompt = get_prompt(args.prompt)
    videos = read_video_list(args.video_list)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.prompt}_results.jsonl"
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            done = {json.loads(line)["video"] for line in f if line.strip()}
    print(f"{len(videos)} videos, {len(done)} already done -> {out_path}")

    model, processor = load_model(args.model_path, args.attn_impl, args.revision)
    with out_path.open("a", encoding="utf-8") as fout:
        for i, name in enumerate(videos):
            if name in done:
                continue
            rec = {"video": name, "status": "error", "output": None,
                   "used_max_frames": None, "used_total_pixels": None, "error": None}
            path = os.path.join(args.video_root, name)
            if not os.path.isfile(path):
                rec["error"] = "video file not found"
            else:
                with tempfile.TemporaryDirectory(prefix="qwen_frames_") as tmp:
                    try:
                        frames, raw_fps = extract_frames(path, tmp, args.sample_fps, LADDER[0][0])
                    except Exception as e:  # broken / unreadable video
                        frames, rec["error"] = None, f"frame extraction failed: {e}"
                    for max_frames, px in LADDER if frames else ():
                        rec["used_max_frames"], rec["used_total_pixels"] = max_frames, px * 32 * 32
                        try:
                            rec["output"] = query_video(model, processor, subsample(frames, max_frames), prompt,
                                                        raw_fps, args.sample_fps, px * 32 * 32, MIN_PIXELS,
                                                        args.max_new_tokens)
                            rec["status"], rec["error"] = "ok", None
                            break
                        except torch.cuda.OutOfMemoryError as e:
                            rec["status"], rec["error"] = "oom", str(e)
                            torch.cuda.empty_cache()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i + 1}/{len(videos)}] {name} -> {rec['status']}")
            gc.collect()


if __name__ == "__main__":
    main()
