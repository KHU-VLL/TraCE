"""Turn an action-recognition split into 5-way multiple-choice QA (AR -> QA protocol).

For every row ``video,label`` of the split csv:
    options = rng.sample(all other unique class names, 4) + [answer]; rng.shuffle(options)
with one ``random.Random(42)`` per split file, so the output is reproducible and
byte-identical to the files used for the paper. Output (dict keyed by row index):

    {"0": {"question": str, "options": [str x 5], "answer": int, "key": video}, ...}

Only ``options`` are encoded by InternVideo2 downstream; ``question`` is stored
for readability. Note that class names may repeat in class_list.txt (Chiral SSv2);
distractors are drawn from the de-duplicated names.

Example:
    python tools/data/build_qa_from_ar.py --dataset_dir annotations/ntu_mutual
    python tools/data/build_qa_from_ar.py --dataset_dir annotations/ucf101 --splits val.csv \
        --out_dir /tmp/qa_check
"""
import argparse
import csv
import json
import random
from collections import OrderedDict
from pathlib import Path

QUESTION = "What is the action shown in this video?"


def load_class_list(path: Path) -> list:
    with path.open() as f:
        return [line.rstrip("\n") for line in f if line.rstrip("\n")]


def build_qa(split_csv: Path, classes: list, seed: int = 42, num_choices: int = 5,
             question: str = QUESTION) -> OrderedDict:
    unique_classes = list(dict.fromkeys(classes))
    rng = random.Random(seed)
    qa = OrderedDict()
    with split_csv.open() as f:
        for i, row in enumerate(csv.reader(f)):
            if not row:
                continue
            video, answer = row[0], classes[int(row[1])]
            pool = [c for c in unique_classes if c != answer]
            options = rng.sample(pool, num_choices - 1) + [answer]
            rng.shuffle(options)
            qa[str(i)] = {"question": question, "options": options,
                          "answer": options.index(answer), "key": video}
    return qa


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset_dir", required=True, help="folder with class_list.txt and split csvs")
    ap.add_argument("--splits", nargs="+", default=["train.csv", "val.csv"])
    ap.add_argument("--class_list", default="class_list.txt")
    ap.add_argument("--out_dir", default=None, help="default: --dataset_dir; writes qa_<split>.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_choices", type=int, default=5)
    args = ap.parse_args()

    ddir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir or ddir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = load_class_list(ddir / args.class_list)
    for split in args.splits:
        qa = build_qa(ddir / split, classes, args.seed, args.num_choices)
        out = out_dir / f"qa_{Path(split).stem}.json"
        with out.open("w") as f:
            json.dump(qa, f, ensure_ascii=False, indent=2)
        print(f"wrote {len(qa)} questions -> {out}")


if __name__ == "__main__":
    main()
