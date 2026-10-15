"""TrainingExample dataclass + JSONL read/write. Storing the interpretable
predicate dict (not raw tensors) decouples generation (needs tampanda) from
training (pure features.py, no MuJoCo).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sahs.predicates import PredicateSet


@dataclass
class TrainingExample:
    instance: str
    step: int
    operator: str  # "pick" | "put"
    target: str
    cylinders: list[str]
    delta: dict  # {"object": str, "region": str | None}
    q: float
    applicable_deltas: list[dict] = field(default_factory=list)
    predicates: dict = field(default_factory=dict)  # serialized PredicateSet


def predicate_set_to_dict(ps: PredicateSet) -> dict:
    return {
        "is_object": ps.is_object,
        "is_region": ps.is_region,
        "is_goal": ps.is_goal,
        "in_region": ps.in_region,
        "pre_free": ps.pre_free,
        "manip_free": {f"{o}|{r}": v for (o, r), v in ps.manip_free.items()},
        "occludes_pre": {f"{o1}|{o2}": v for (o1, o2), v in ps.occludes_pre.items()},
        "occludes_manip": {f"{o1}|{o2}|{r}": v for (o1, o2, r), v in ps.occludes_manip.items()},
    }


def predicate_set_from_dict(d: dict) -> PredicateSet:
    return PredicateSet(
        is_object=d["is_object"],
        is_region=d["is_region"],
        is_goal=d["is_goal"],
        in_region=d["in_region"],
        pre_free=d["pre_free"],
        manip_free={tuple(k.split("|")): v for k, v in d["manip_free"].items()},
        occludes_pre={tuple(k.split("|")): v for k, v in d["occludes_pre"].items()},
        occludes_manip={tuple(k.split("|")): v for k, v in d["occludes_manip"].items()},
    )


def write_jsonl(examples: list[TrainingExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


def read_jsonl(path: Path) -> list[TrainingExample]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(TrainingExample(**json.loads(line)))
    return examples
