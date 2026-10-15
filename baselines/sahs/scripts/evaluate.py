#!/usr/bin/env python
"""SAHS on the 300 open-tabletop problems of Table 1 (15 evaluation sets of 20 problems, 10 to 24 cylinders: object
count, plan length and arrangement), 300 s per problem. The time limit stops the search; --max-pops is a backstop.

Sharding: --start/--end index the list of the 300 problems (the evaluation sets in the order of EVAL_SETS, the
problems of a set sorted by name), so several processes can split the problems, each with its own
ParallelFeasibilityPool.

    python baselines/sahs/scripts/evaluate.py \\
        --eval-root $REAP_DATA/tabletop/eval \\
        --domain $REAP_DATA/tabletop/domain.pddl \\
        --q-pick $REAP_CKPT/baselines/sahs/sahs_pick.pt \\
        --q-put $REAP_CKPT/baselines/sahs/sahs_put.pt \\
        --start 0 --end 25 --workers 16 --time-limit-s 300 --max-pops 3000 \\
        --out outputs/baselines/sahs/shard_0.json
(python baselines/sahs/run.py eval runs the 12 shards of 25 problems of the Table 1 run.)

One record per problem: cell (the evaluation set), problem, num_cylinders, target, reap_plan_length (length of the
reference plan), solved, plan_length, expansions, exp_per_step, wall_s, feature_building_calls (checker calls that
compute the predicates), realization_calls (checker calls that realize popped actions) and total_calls (their sum,
MP / plan in Table 1).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sahs._bootstrap import ensure_upstream_importable  # noqa: E402

ensure_upstream_importable()

import torch  # noqa: E402
import reap.xmimir as xmi  # noqa: E402

from sahs import state_adapter  # noqa: E402
from sahs.checker import InstrumentedFeasibilityChecker  # noqa: E402
from sahs.gnn import TripleRelationalQNet, load_qnet  # noqa: E402
from sahs.occlusion import OcclusionChecker  # noqa: E402
from sahs.parallel_pool import ParallelFeasibilityPool  # noqa: E402
from sahs.sahs import sahs_search  # noqa: E402

_TARGET_RE = re.compile(r";\s*Target:\s*(\S+)")
_TOTAL_ACTIONS_RE = re.compile(r";\s*Total actions:\s*(\d+)")

# The 15 evaluation sets of Table 1, 20 problems each.
EVAL_SETS = [
    "obj10",
    "obj10_plan3",
    "obj10_plan5_biased2",
    "obj12",
    "obj12_plan3",
    "obj12_plan7_biased3",
    "obj14",
    "obj14_plan5",
    "obj14_plan9_biased3",
    "obj16",
    "obj16_plan5",
    "obj18",
    "obj20",
    "obj22",
    "obj24",
]


def _target_from_plan(plan_path: Path) -> str | None:
    for line in plan_path.read_text().splitlines():
        m = _TARGET_RE.match(line.strip())
        if m:
            return m.group(1)
    return None


def _reap_plan_length(plan_path: Path) -> int | None:
    for line in plan_path.read_text().splitlines():
        m = _TOTAL_ACTIONS_RE.match(line.strip())
        if m:
            return int(m.group(1))
    return None


def list_all_problems(eval_root: Path) -> list[tuple[str, Path]]:
    problems = []
    for cell in EVAL_SETS:
        cell_dir = eval_root / cell
        for pddl_path in sorted(cell_dir.glob("*.pddl"), key=lambda p: p.stem):
            problems.append((cell, pddl_path))
    return problems


def load_nets(q_pick_path: Path, q_put_path: Path):
    """Both networks; the edge feature set follows from the checkpoint's edge width (gnn.load_qnet)."""
    q_pick = load_qnet(q_pick_path)
    q_put = load_qnet(q_put_path)
    print(f"edge features: q_pick {q_pick.edge_features}, q_put {q_put.edge_features}", flush=True)
    return q_pick, q_put


def eval_one(
    cell: str, pddl_path: Path, domain_path: Path, q_pick, q_put,
    checker: InstrumentedFeasibilityChecker, occlusion_checker: OcclusionChecker,
    pool: ParallelFeasibilityPool, time_limit_s: float, max_pops: int,
) -> dict:
    plan_path = pddl_path.with_suffix(".pddl.plan")
    target = _target_from_plan(plan_path) if plan_path.exists() else None
    reap_plan_length = _reap_plan_length(plan_path) if plan_path.exists() else None

    _domain, problem = xmi.parse(str(domain_path), str(pddl_path))
    gen_state = xmi.XSuccessorGenerator(problem).initial_state
    cylinders = sorted(
        state_adapter.state_to_robot_state(gen_state)["cylinders"].keys(), key=lambda n: int(n.split("_")[1]),
    )
    if target is None:
        # every problem of the suite has a .plan file that names the target;
        # without one, the first cylinder is used
        target = cylinders[0]

    pool.reset_call_counter()
    checker.stats = type(checker.stats)()
    stats_out: dict = {}
    t0 = time.perf_counter()
    plan = sahs_search(
        problem, target, cylinders, q_pick, q_put, checker, occlusion_checker,
        max_pops=max_pops, pool=pool, time_limit_s=time_limit_s, stats_out=stats_out,
    )
    elapsed = time.perf_counter() - t0
    feature_building_calls = pool.total_calls_since_reset()
    realization_calls = checker.stats.total_calls
    plan_length = len(plan) if plan is not None else None
    expansions = stats_out.get("expansions")

    return {
        "cell": cell,
        "problem": pddl_path.stem,
        "num_cylinders": len(cylinders),
        "target": target,
        "reap_plan_length": reap_plan_length,
        "solved": plan is not None,
        "plan_length": plan_length,
        "expansions": expansions,
        "exp_per_step": (expansions / plan_length) if (plan is not None and plan_length) else None,
        "wall_s": round(elapsed, 2),
        "feature_building_calls": feature_building_calls,
        "realization_calls": realization_calls,
        "total_calls": feature_building_calls + realization_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--domain", required=True, type=Path)
    parser.add_argument("--q-pick", required=True, type=Path)
    parser.add_argument("--q-put", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--time-limit-s", type=float, default=300.0)
    parser.add_argument("--max-pops", type=int, default=3000)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    all_problems = list_all_problems(args.eval_root)
    shard = all_problems[args.start:args.end]
    print(f"shard: {len(shard)} problems ({args.start}:{args.end}) of {len(all_problems)} total")

    q_pick, q_put = load_nets(args.q_pick, args.q_put)
    checker = InstrumentedFeasibilityChecker()
    occlusion_checker = OcclusionChecker()
    pool = ParallelFeasibilityPool(n_workers=args.workers)

    # written after every problem, so a stopped shard keeps its finished problems
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.perf_counter()
    try:
        for i, (cell, pddl_path) in enumerate(shard):
            r = eval_one(
                cell, pddl_path, args.domain, q_pick, q_put, checker, occlusion_checker,
                pool, args.time_limit_s, args.max_pops,
            )
            results.append(r)
            print(f"  [{i+1}/{len(shard)}] {cell}/{r['problem']} (N={r['num_cylinders']}): "
                  f"solved={r['solved']} plan_len={r['plan_length']} wall_s={r['wall_s']} "
                  f"total_calls={r['total_calls']} ({time.perf_counter() - t0:.0f}s elapsed)")
            args.out.write_text(json.dumps(results, indent=2))
    finally:
        pool.shutdown()

    solved = [r for r in results if r["solved"]]
    print(f"\nsolved {len(solved)}/{len(results)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
