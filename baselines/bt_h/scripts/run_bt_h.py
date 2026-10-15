#!/usr/bin/env python3
"""BT-h and classic backtracking on the 300 open-tabletop problems of Table 1: the row "BT-h, given the plan
ordering" (with the trained classifier). MP / plan counts distinct (state, action) motion-planner queries.

It writes the records of `bt_h/per_problem.jsonl`:

- Problems, order and seeds of the Table 1 run, listed in `baselines/bt_h/data/problem_seeds.tsv`: one line
  "<set>/config_N.pddl<TAB>seed" per problem, in the order of that run (obj10_plan3 first, obj10 third; the i-th
  problem, 0-based, has seed 8000 + i). Both searches of a problem get `random.Random(seed)`; the classifier gets its
  own `random.Random(0)`; tau = `config.BT_H_TAU`; 300 s per problem and method; `config.EVAL_MAX_CHECKER_CALLS`.
- The plan ordering pi = `reap_plan_to_pi(<problem>.pddl.plan)`: the cylinder visit order of REAP's reference plan,
  destinations discarded (BT-h searches its own).
- One `InstrumentedFeasibilityChecker` per method for the whole run (per shard when --start/--end split the run). Its
  cache keys carry the full cylinder layout, so a key cannot repeat across problems.
- Per problem and method one JSON line, classic first, with the fields of `bt_h/per_problem.jsonl`:
  `checker_calls` (the search's own count = cache misses), `mp_calls_dedup` (cache misses), `cache_hits`,
  `mp_calls_undedup` (misses + hits), `primary_calls` (check() calls without force_refresh, hits included),
  `retry_calls` (the force_refresh retries of `controller._retry_check`, all misses). MP / plan is
  `primary_calls - cache_hits` (`reap.eval.table1_baselines`, `load_bt_h`). The relations
  `mp_calls_undedup = mp_calls_dedup + cache_hits`, `primary_calls + retry_calls = mp_calls_undedup` and
  `checker_calls = mp_calls_dedup` hold on every record.

The problems are read from the dataset asset ($REAP_DATA/tabletop/eval/<set>/config_N.pddl and .pddl.plan). Only
the occupied cells and the plan are read, so no domain-name patch of the problem files is needed.

    python baselines/bt_h/scripts/run_bt_h.py \\
        --classifier checkpoints/baselines/bt_h/bt_h.pt \\
        --out outputs/baselines/bt_h/per_problem.jsonl [--start 0 --end 300]
(python baselines/bt_h/run.py eval runs it in 30 shards and merges them.)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bt_h._bootstrap import ensure_upstream_importable  # noqa: E402

ensure_upstream_importable()

import torch  # noqa: E402

from bt_h import config  # noqa: E402
from bt_h.backtracking import classic_backtrack  # noqa: E402
from bt_h.bt_h import bt_h_search  # noqa: E402
from bt_h.checker import InstrumentedFeasibilityChecker  # noqa: E402
from bt_h.classifier.inference import TrainedClassifier  # noqa: E402
from bt_h.classifier.model import FeasibilityClassifier  # noqa: E402
from bt_h.domain.reap_plan_parser import reap_plan_to_pi  # noqa: E402
from bt_h.state import initial_state  # noqa: E402

OCCUPIED_RE = re.compile(r"\(occupied (cell_\d+_\d+) (cylinder_\d+)\)")
REAP_TIME_LIMIT_S = 300.0  # Table 1: 300 s per problem
SEEDS_TSV = REPO_ROOT / "data" / "problem_seeds.tsv"  # the 300 problems of the Table 1 run, in its order, with seeds


class CountingChecker(InstrumentedFeasibilityChecker):
    """InstrumentedFeasibilityChecker that also counts check() calls with and without force_refresh."""

    def __init__(self) -> None:
        super().__init__()
        self.primary_calls = 0
        self.retry_calls = 0

    def check(self, robot_state, action, cyl, target_cell=None, call_label="", force_refresh=False):
        if force_refresh:
            self.retry_calls += 1
        else:
            self.primary_calls += 1
        return super().check(robot_state, action, cyl, target_cell=target_cell, call_label=call_label,
                             force_refresh=force_refresh)

    def counters(self) -> dict:
        return {"misses": self.stats.total_calls, "hits": self.stats.total_cache_hits,
                "primary": self.primary_calls, "retry": self.retry_calls}


def list_problems(eval_root: Path) -> list[tuple[str, Path, int]]:
    """(set, problem path, seed) for each line of SEEDS_TSV, in its order (the order of the Table 1 run)."""
    problems = []
    for line in SEEDS_TSV.read_text().splitlines():
        rel, seed = line.split("\t")
        p = eval_root / rel
        if not p.is_file():
            raise SystemExit(f"no {p} (extract the dataset asset and point REAP_DATA at its data/ directory)")
        problems.append((rel.split("/")[0], p, int(seed)))
    return problems


def parse_pddl_problem(path: Path) -> dict:
    return {cyl: cell for cell, cyl in OCCUPIED_RE.findall(path.read_text())}


def load_trained_classifier(ckpt_path: Path, seed: int = 0) -> TrainedClassifier:
    model = FeasibilityClassifier()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return TrainedClassifier(model, rng=random.Random(seed))


def record(method: str, bucket: str, name: str, pi: list, result, before: dict, after: dict) -> dict:
    d = {k: after[k] - before[k] for k in before}
    return {
        "method": method,
        "k": sum(1 for a in pi if a.kind == "relocate"),
        "instance": name,
        "bucket": bucket,
        "plan_length": len(pi),
        "solved": result.solved,
        "checker_calls": result.checker_calls,
        "mp_calls_dedup": d["misses"],
        "cache_hits": d["hits"],
        "mp_calls_undedup": d["misses"] + d["hits"],
        "primary_calls": d["primary"],
        "retry_calls": d["retry"],
        "wall_time_s": round(result.wall_time_s, 3),
        "termination_reason": result.termination_reason,
    }


def main() -> None:
    data = Path(os.environ.get("REAP_DATA") or "data")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-root", type=Path, default=data / "tabletop" / "eval", help="default: $REAP_DATA/tabletop/eval")
    ap.add_argument("--classifier", type=Path, required=True, help="BT-h classifier weights (state_dict)")
    ap.add_argument("--out", type=Path, required=True, help="JSONL, one line per problem and method")
    ap.add_argument("--start", type=int, default=0, help="first problem (line of data/problem_seeds.tsv, 0-based)")
    ap.add_argument("--end", type=int, default=None, help="end (exclusive); default: all 300")
    ap.add_argument("--time-limit-s", type=float, default=REAP_TIME_LIMIT_S)
    args = ap.parse_args()

    problems = list_problems(args.eval_root)
    end = len(problems) if args.end is None else min(args.end, len(problems))
    print(f"{len(problems)} problems; this run: [{args.start}, {end}), {args.time_limit_s:.0f} s per problem")
    classifier = load_trained_classifier(args.classifier)
    checker_classic = CountingChecker()
    checker_bt_h = CountingChecker()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with args.out.open("w") as fh:
        for i in range(args.start, end):
            bucket, problem_path, seed = problems[i]
            name = f"{bucket}__{problem_path.stem}"
            pi = reap_plan_to_pi(problem_path.with_name(problem_path.name + ".plan"))
            sigma0 = initial_state(parse_pddl_problem(problem_path))

            b = checker_classic.counters()
            r_classic = classic_backtrack(
                sigma0, pi, checker_classic, rng=random.Random(seed),
                max_calls=config.EVAL_MAX_CHECKER_CALLS, time_limit_s=args.time_limit_s,
            )
            rec_classic = record("classic", bucket, name, pi, r_classic, b, checker_classic.counters())

            b = checker_bt_h.counters()
            r_bt_h = bt_h_search(
                sigma0, pi, checker_bt_h, classifier, rng=random.Random(seed),
                max_calls=config.EVAL_MAX_CHECKER_CALLS, time_limit_s=args.time_limit_s, tau=config.BT_H_TAU,
            )
            rec_bt_h = record("bt_h", bucket, name, pi, r_bt_h, b, checker_bt_h.counters())

            for rec in (rec_classic, rec_bt_h):
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i + 1}/{len(problems)}] {name} (k={rec_bt_h['k']}, plan_len={len(pi)}): "
                  f"classic={r_classic.termination_reason}({rec_classic['primary_calls'] - rec_classic['cache_hits']}) "
                  f"bt_h={r_bt_h.termination_reason}({rec_bt_h['primary_calls'] - rec_bt_h['cache_hits']}) "
                  f"[{time.time() - t0:.0f} s]", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
