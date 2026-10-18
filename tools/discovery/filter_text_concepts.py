"""Union per-video Qwen3-VL answers and apply the length + redundancy filters.

    answers --parse/union (sorted)--> length <= 30 chars --> redundancy filter
    ((MPNet + 3 * CLIP ViT-B/16) / 4 >= sim_cutoff, LF-CBM rule) --> <out_txt>

The output is one concept per line with no trailing newline (the format of the
released ``concepts/<dataset>/*_candidates.txt``). Candidates are processed in
sorted order, so the result is deterministic; the released files were produced
by the research code, which iterated an unordered set, so re-running this tool
need not reproduce them exactly. The released files are the reference.

The activation (``clip_cutoff``) and projection (``interpretability_cutoff``)
filters are NOT applied here; they run inside action-recognition training.

Example:
    python tools/discovery/filter_text_concepts.py \
        --results outputs/text_concepts/ntu_mutual/object_v1_results.jsonl \
        --out_txt outputs/text_concepts/ntu_mutual/object_candidates.txt --sim_cutoff 0.9
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.discovery.text.filtering import combined_similarity, remove_redundant, remove_too_long  # noqa: E402
from src.discovery.text.parsing import load_answers, union_candidates  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", nargs="+", required=True,
                    help="generator output(s): *_results.jsonl, *_results.json or *_outputs.json")
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--max_len", type=int, default=30)
    ap.add_argument("--sim_cutoff", type=float, default=0.9)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mpnet", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch16")
    ap.add_argument("--save_union", default=None, help="also write the unfiltered union here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    answers = [a for p in args.results for a in load_answers(p)]
    union = union_candidates(answers)
    print(f"answers={len(answers)} unique candidates={len(union)}")
    if args.save_union:
        Path(args.save_union).write_text("\n".join(union), encoding="utf-8")

    short = remove_too_long(union, args.max_len)
    print(f"length filter (<= {args.max_len} chars): {len(union)} -> {len(short)}")
    sim = combined_similarity(short, device=args.device, mpnet_name=args.mpnet, clip_name=args.clip)
    kept = remove_redundant(short, sim, args.sim_cutoff, verbose=args.verbose)
    print(f"redundancy filter (sim >= {args.sim_cutoff}): {len(short)} -> {len(kept)}")

    out = Path(args.out_txt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
