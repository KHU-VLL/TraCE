"""Qwen3-VL prompts for object / scene concept candidate generation (paper Fig. 9).

These strings are the ``object_v1`` and ``scene_v1`` prompts used to build the
released vocabularies (identical to the research code).
"""

OBJECT_V1_PROMPT = (
    "List the main physical objects that are clearly visible in this video.\n"
    "\n"
    "Rules:\n"
    "- Include ONLY objects you can clearly see and identify.\n"
    "- If no clear object is identifiable, output exactly: none\n"
    "- Do NOT guess or list objects that might be present but are not clearly visible.\n"
    "- Include only tangible, movable objects.\n"
    "- Do NOT include people or body parts.\n"
    "- Do NOT include actions or activities.\n"
    "- Do NOT include background locations (e.g., park, street, living room).\n"
    "- Use short, singular, lowercase noun phrases (e.g., \"cup\", \"bicycle\", \"laptop\").\n"
    "- One concept per line. No compound phrases joined by \"and\".\n"
    "- Maximum 5 items. Fewer is fine.\n"
    "- No full sentences. Bullet list format.\n"
    "\n"
    "Examples (text descriptions of hypothetical videos and the expected output):\n"
    "\n"
    "Video: Hands pick up a red cup from a wooden table next to a book and a pen.\n"
    "Output:\n"
    "- cup\n"
    "- book\n"
    "- pen\n"
    "\n"
    "Video: A blurry, dark scene where nothing is clearly identifiable.\n"
    "Output:\n"
    "- none\n"
    "\n"
    "Video: Someone places a smartphone, a pair of glasses, and a set of keys onto a desk.\n"
    "Output:\n"
    "- smartphone\n"
    "- glasses\n"
    "- key\n"
    "\n"
    "Video: A hand pours water from a bottle into a glass.\n"
    "Output:\n"
    "- bottle\n"
    "- glass\n"
    "\n"
    "Now produce the output for the given video.\n"
)

SCENE_V1_PROMPT = (
    "List the background scenes or environment types that are clearly visible in this video.\n"
    "\n"
    "Rules:\n"
    "- Include ONLY scenes you can clearly identify from direct visual evidence.\n"
    "- If no scene is clearly identifiable, output exactly: none\n"
    "- Do NOT guess, infer, or imagine scenes that are not visually obvious.\n"
    "- Do NOT include people, actions, objects, or equipment.\n"
    "- Use short, singular, lowercase noun phrases (e.g., \"kitchen\", \"city street\", \"forest\").\n"
    "- One concept per line. No compound phrases joined by \"and\".\n"
    "- Maximum 3 items. Fewer is fine.\n"
    "- No full sentences. Bullet list format.\n"
    "\n"
    "Examples (text descriptions of hypothetical videos and the expected output):\n"
    "\n"
    "Video: A person chops vegetables on a marble counter, with cabinets and a stove behind them.\n"
    "Output:\n"
    "- kitchen\n"
    "- countertop\n"
    "\n"
    "Video: Two hands rotate a small object directly in front of a plain white background, no surroundings visible.\n"
    "Output:\n"
    "- none\n"
    "\n"
    "Video: A child runs across grass with trees in the distance and a slide nearby.\n"
    "Output:\n"
    "- park\n"
    "- playground\n"
    "\n"
    "Video: Cars drive past tall glass buildings on a busy road with traffic lights.\n"
    "Output:\n"
    "- city street\n"
    "\n"
    "Now produce the output for the given video.\n"
)

PROMPTS = {"object_v1": OBJECT_V1_PROMPT, "scene_v1": SCENE_V1_PROMPT}


def get_prompt(name: str) -> str:
    """Return the prompt for ``name`` in {"object_v1", "scene_v1"} (or "object"/"scene")."""
    key = name if name in PROMPTS else f"{name}_v1"
    if key not in PROMPTS:
        raise KeyError(f"unknown prompt {name!r}; choose from {sorted(PROMPTS)}")
    return PROMPTS[key]

