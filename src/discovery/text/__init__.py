"""Object / scene (text) concept discovery: Qwen3-VL candidates, filtering, pseudo-labels.

Heavy dependencies (transformers, sentence-transformers, qwen-vl-utils) are only
imported inside the functions that need them.
"""
from .parsing import clean_parentheses, load_answers, parse_answer, union_candidates
from .prompts import OBJECT_V1_PROMPT, PROMPTS, SCENE_V1_PROMPT, get_prompt
from .pseudo_labels import (CONCEPT_PROMPT, activation_filter, activation_scores,
                            concept_prompts, concept_pseudo_labels)

__all__ = [
    "OBJECT_V1_PROMPT", "SCENE_V1_PROMPT", "PROMPTS", "get_prompt",
    "parse_answer", "clean_parentheses", "union_candidates", "load_answers",
    "CONCEPT_PROMPT", "concept_prompts", "concept_pseudo_labels", "activation_scores", "activation_filter",
]
