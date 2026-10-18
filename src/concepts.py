"""Concept vocabulary shared by the AR and VQA checkpoints.

A vocabulary is an ordered list of concepts. The order is the column order of
the concept layer, so ``vocab[i]`` describes concept dimension ``i`` of the
checkpoint it ships with. Concepts are grouped as ``trajectory``, ``object``
and ``scene`` and the groups are contiguous, in that order.

``concepts.json`` layout::

    {
      "groups": ["trajectory", "object", "scene"],
      "concepts": [
        {"id": 0, "group": "trajectory", "name": "trajectory_000", "group_index": 0,
         "source_index": 0},
        ...
      ]
    }

* ``id``           position in the concatenated concept vector (0..M-1).
* ``group_index``  position inside its group.
* ``source_index`` index in the vocabulary file the group was built from
                   (trajectory cluster id, or line number of the object/scene
                   candidate list before filtering). Use it to map a concept
                   back to clustering outputs (medoids) or raw candidates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

GROUPS = ("trajectory", "object", "scene")


@dataclass
class Concept:
    id: int
    group: str
    name: str
    group_index: int
    source_index: Optional[int] = None


@dataclass
class ConceptVocab:
    concepts: List[Concept]
    groups: List[str] = field(default_factory=lambda: list(GROUPS))

    def __post_init__(self):
        for i, c in enumerate(self.concepts):
            if c.id != i:
                raise ValueError(f"concept ids must be 0..M-1 in order; got id={c.id} at position {i}")
        seen = []
        for c in self.concepts:
            if not seen or seen[-1] != c.group:
                if c.group in seen:
                    raise ValueError(f"group '{c.group}' is not contiguous")
                seen.append(c.group)
        self.groups = [g for g in self.groups if g in seen]

    def __len__(self) -> int:
        return len(self.concepts)

    def __getitem__(self, i: int) -> Concept:
        return self.concepts[i]

    def slices(self) -> Dict[str, slice]:
        """Group name -> slice into the concatenated concept vector."""
        out = {}
        for g in self.groups:
            ids = [c.id for c in self.concepts if c.group == g]
            out[g] = slice(ids[0], ids[-1] + 1)
        return out

    def group_sizes(self) -> Dict[str, int]:
        return {g: s.stop - s.start for g, s in self.slices().items()}

    def ids(self, group: str) -> List[int]:
        return [c.id for c in self.concepts if c.group == group]

    def names(self) -> List[str]:
        return [c.name for c in self.concepts]

    def label(self, i: int) -> str:
        c = self.concepts[i]
        return f"[{c.group}] {c.name}"

    @classmethod
    def from_groups(cls, groups: Dict[str, Sequence[str]],
                    source_indices: Optional[Dict[str, Sequence[int]]] = None) -> "ConceptVocab":
        concepts = []
        for g in GROUPS:
            if g not in groups:
                continue
            src = source_indices.get(g) if source_indices else None
            for j, name in enumerate(groups[g]):
                concepts.append(Concept(id=len(concepts), group=g, name=str(name), group_index=j,
                                        source_index=None if src is None else int(src[j])))
        return cls(concepts)

    @classmethod
    def load(cls, path: str) -> "ConceptVocab":
        with open(path) as f:
            d = json.load(f)
        return cls([Concept(**c) for c in d["concepts"]], d.get("groups", list(GROUPS)))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"groups": self.groups,
                       "concepts": [c.__dict__ for c in self.concepts]}, f, indent=1)


def trajectory_names(n: int) -> List[str]:
    """Trajectory concepts have no language names; they are referred to by cluster id."""
    return [f"trajectory_{i:03d}" for i in range(n)]
