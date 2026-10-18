"""Parse Qwen3-VL bullet-list answers into concept candidates (union per dataset).

Same rules as the research code:
  * strip a leading bullet ("-", "*", "•"),
  * split on newlines, commas, "/", "|" and on the words " and " / " or ",
  * strip surrounding whitespace and ``" .;:"``,
  * drop parenthesised parts, e.g. "cup (red)" -> "cup".
Nothing is lower-cased and the literal answer "none" is kept as a candidate
(the released vocabularies contain it for most datasets).
"""
import json
import re
from typing import Iterable, List, Set

_BULLET = re.compile(r"^\s*[-*•]\s*")


def parse_answer(text: str) -> List[str]:
    """Split one model answer into phrases."""
    text = text.strip()
    if not text:
        return []
    text = _BULLET.sub("", text)
    text = text.replace("\n", ", ")
    text = re.sub(r"\s+(and|or)\s+", ", ", text, flags=re.IGNORECASE)
    parts = [_BULLET.sub("", p).strip(" .;:") for p in re.split(r"[,/|]", text)]
    return [p for p in parts if p]


def clean_parentheses(concepts: Iterable[str]) -> List[str]:
    return [re.sub(r"\s*\(.*?\)", "", c).strip() for c in concepts if c and c.strip()]


def union_candidates(answers: Iterable[str]) -> List[str]:
    """Union of parsed phrases over all videos, in sorted order (deterministic).

    The research code iterated over an unordered Python ``set`` here, so its
    order (and therefore the later redundancy filter) depended on PYTHONHASHSEED.
    """
    pool: Set[str] = set()
    for a in answers:
        if isinstance(a, str):
            pool.update(parse_answer(a))
        elif isinstance(a, list):
            for x in a:
                if isinstance(x, str):
                    pool.update(parse_answer(x))
    cleaned = clean_parentheses(sorted(pool))
    # cleaning can create duplicates ("cup (red)" -> "cup"); keep first occurrence
    return [c for c in dict.fromkeys(cleaned) if c]


def load_answers(path: str) -> List[str]:
    """Read answers from a generator output.

    Accepts ``*_results.json`` / ``*.jsonl`` records (``{"video", "status", "output"}``;
    only ``status == "ok"`` is used) or a ``*_outputs.json`` list of strings.
    """
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {path}")
    answers = []
    for item in data:
        if isinstance(item, str):
            answers.append(item)
        elif isinstance(item, dict) and item.get("status") == "ok" and item.get("output") is not None:
            answers.append(item["output"])
    return answers
