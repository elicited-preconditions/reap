"""Per-problem evaluation records of run_evaluation.py (JSONL, one line per problem), the input of the Table 1 PLOI
rows (reap.eval.table1_baselines).

mp_calls is MP / plan as for REAP: the distinct (state, action) motion-planner queries of the problem, the size of the
checker cache (`len(checker._cache)`, see checker.py's `cache_size()`/`clear_cache()`), not the number of
filter_action calls.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ProblemEvalResult:
    problem: str
    method: str  # "ploi_bfs" | "ploi_hff" | "bfs" | "hff"
    solved: bool
    mp_calls: int          # distinct (state, action) motion-planner queries of this problem
    expansions: int        # total search-node expansions across every PLAN() attempt for this problem
    plan_length: int | None
    wall_time_s: float
    timed_out: bool
    n_plan_calls: int = 1  # for PLOI methods: how many object-set attempts (PLAN() calls) were made
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ProblemEvalResult":
        return ProblemEvalResult(**d)


def write_jsonl(results: list[ProblemEvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")

