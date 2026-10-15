"""TrainingExample: one labeled training problem for PLOI's object scorer.
Same JSONL read/write pattern as baselines/sahs/sahs/dataset.py.

Stores the problem and domain file paths and the labels (object_name -> 0/1), not feature tensors: the features are
computed from the parsed problem when the labels are loaded (no MuJoCo needed).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


def _data_root() -> Path:
    return Path(os.environ.get("REAP_DATA") or "data")


def _portable(path: str) -> str:
    """A path below $REAP_DATA is stored relative to it, so a label file does
    not depend on where the dataset asset was unpacked."""
    try:
        return str(Path(path).resolve().relative_to(_data_root().resolve()))
    except ValueError:
        return path


def _resolve(path: str) -> str:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return path
    return str(_data_root() / p)


@dataclass
class TrainingExample:
    problem_path: str  # path to the ORIGINAL (full, unreduced) .pddl problem file
    domain_path: str
    labels: dict[str, int]  # object_name -> 1 (important) or 0
    target_cylinder: str  # convenience -- also derivable from labels/goal, kept for readability

    @staticmethod
    def from_dict(d: dict) -> "TrainingExample":
        d = dict(d, problem_path=_resolve(d["problem_path"]), domain_path=_resolve(d["domain_path"]))
        return TrainingExample(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["problem_path"] = _portable(d["problem_path"])
        d["domain_path"] = _portable(d["domain_path"])
        return d


def write_jsonl(examples: list[TrainingExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")


def read_jsonl(path: Path) -> list[TrainingExample]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(TrainingExample.from_dict(json.loads(line)))
    return examples
