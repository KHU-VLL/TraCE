"""Qwen3-VL-8B-Instruct video querying for object / scene candidates.

Settings used for the released vocabularies:
  * checkpoint ``Qwen/Qwen3-VL-8B-Instruct`` (HF revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b),
    ``dtype="auto"``, ``device_map="auto"``, SDPA attention;
  * frames decoded with ffmpeg at 2 fps (at most 2048 frames) and passed as a frame
    list to ``qwen_vl_utils`` with ``total_pixels = 20480*32*32``, ``min_pixels = 64*32*32``;
  * greedy decoding (``do_sample=False``), ``max_new_tokens=256``;
  * on CUDA OOM the video is retried with fewer frames / pixels (``LADDER``).
    All released runs succeeded at the first rung (2048 frames).
"""
import glob
import json
import os
import subprocess
from typing import List, Tuple

QWEN3VL_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3VL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

# (max_frames, total_pixels in units of 32x32 patches)
LADDER = ((2048, 20480), (128, 16384), (64, 12288), (32, 8192), (16, 6144))
MIN_PIXELS = 64 * 32 * 32


def _probe(video_path: str) -> Tuple[float, float]:
    """(duration in seconds, average fps) via ffprobe."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=avg_frame_rate:format=duration", "-of", "json", video_path],
                       capture_output=True, text=True, timeout=120)
    data = json.loads(r.stdout or "{}")
    dur = float(data.get("format", {}).get("duration", 0) or 0)
    streams = data.get("streams", [])
    num, _, den = (streams[0].get("avg_frame_rate", "0/1") if streams else "0/1").partition("/")
    try:
        fps = float(num) / float(den) if den and float(den) != 0 else float(num or 0)
    except ValueError:
        fps = 0.0
    return dur, fps


def subsample(items: List, k: int) -> List:
    """Evenly pick k items, order preserved."""
    n = len(items)
    if k >= n:
        return list(items)
    if k <= 1:
        return [items[0]]
    return [items[j] for j in sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})]


def extract_frames(video_path: str, out_dir: str, sample_fps: float = 2.0,
                   max_frames: int = 2048) -> Tuple[List[str], float]:
    """Decode frames at ~sample_fps (lowered for very long videos) into out_dir as JPEGs."""
    dur, raw_fps = _probe(video_path)
    fps = max_frames / dur if dur > 0 and dur * sample_fps > max_frames else sample_fps
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-i", video_path,
                    "-vf", f"fps={fps:.6f}", "-q:v", "3", os.path.join(out_dir, "f_%06d.jpg")],
                   check=True, timeout=1200)
    frames = sorted(glob.glob(os.path.join(out_dir, "f_*.jpg")))
    if not frames:
        raise RuntimeError("ffmpeg extracted 0 frames")
    return subsample(frames, max_frames), (raw_fps or sample_fps)


def load_model(model_path: str = QWEN3VL_MODEL_ID, attn_impl: str = "sdpa", revision: str = None):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, revision=revision, dtype="auto", device_map="auto", attn_implementation=attn_impl)
    return model.eval(), processor


def query_video(model, processor, frames: List[str], prompt: str, raw_fps: float, sample_fps: float = 2.0,
                total_pixels: int = LADDER[0][1] * 32 * 32, min_pixels: int = MIN_PIXELS,
                max_new_tokens: int = 256) -> str:
    """One greedy Qwen3-VL answer for a list of frame image paths."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"video": frames, "total_pixels": total_pixels, "min_pixels": min_pixels,
         "raw_fps": raw_fps, "sample_fps": sample_fps},
        {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info([messages], return_video_kwargs=True,
                                                       image_patch_size=16, return_video_metadata=True)
    metadata = None
    if videos is not None:
        videos, metadata = map(list, zip(*videos))
    inputs = processor(text=[text], images=images, videos=videos, video_metadata=metadata,
                       **video_kwargs, do_resize=False, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
