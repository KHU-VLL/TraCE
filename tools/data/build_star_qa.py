"""Convert the official STAR question files into the QA format used by TraCE.

Input: STAR_train.json / STAR_val.json from the STAR release. Each question becomes

    {"video": "<video_id>.mp4", "option_text": ["Q: <question> A: <choice>", ...], "answer": <choice_id>}

in the original question order. The four options keep STAR's ``choice_id`` order and
the question is part of every option string (InternVideo2 encodes each option string).
The question's start/end timestamps are not used: features are computed on the whole
Charades video.

Example:
    python tools/data/build_star_qa.py --star_json <STAR>/STAR_train.json --out annotations/star/STAR_preprocessed_train.json
"""
import argparse
import json


def convert(samples):
    out = []
    for s in samples:
        options = [f"Q: {s['question']} A: {c['choice']}" for c in s["choices"]]
        answer = None
        for c in s["choices"]:
            if c["choice"] == s["answer"]:
                answer = c["choice_id"]
        if answer is None:
            raise ValueError(f"answer not among choices for question_id={s.get('question_id')}")
        out.append({"video": f"{s.get('video_id')}.mp4", "option_text": options, "answer": answer})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--star_json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.star_json) as f:
        data = convert(json.load(f))
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {len(data)} questions -> {args.out}")


if __name__ == "__main__":
    main()
