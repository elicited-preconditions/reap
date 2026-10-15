"""PLOI on the 300 open-tabletop problems of Table 1 (15 evaluation sets of 20 problems), 300 s per problem
(config.EVAL_TIME_LIMIT_S):

  ploi_bfs / ploi_hff  -- PLOI's incremental loop (Algorithm 1) with the
                          trained GNN scorer and bfs_with_plan /
                          gbfs_hff_with_plan as PLAN: the Table 1 rows
                          PLOI + BrFS and PLOI + hFF.
  bfs / hff             -- the same searches on the full problem, without
                           object reduction (not a Table 1 row).

One record per problem (report.py): solved, mp_calls (MP / plan: distinct (state, action) motion-planner queries),
expansions, plan_length, wall time.

Usage (one shard of the 300 problems; python baselines/ploi/run.py eval runs the 30 shards of 10 of the Table 1 runs):
    python baselines/ploi/scripts/run_evaluation.py \\
        --eval-root $REAP_DATA/tabletop/eval \\
        --domain $REAP_DATA/tabletop/domain.pddl \\
        --mode ploi_hff --model $REAP_CKPT/baselines/ploi/ploi.pt \\
        --start 0 --end 10 \\
        --out outputs/baselines/ploi_hff/shard_0.jsonl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ploi._bootstrap import ensure_upstream_importable  # noqa: E402

ensure_upstream_importable()

import reap.xmimir as xmi  # noqa: E402
import torch  # noqa: E402

from ploi import config, features, plan_search, ploi, report  # noqa: E402
from ploi.checker import get_checker  # noqa: E402
from ploi.model import ObjectImportanceGNN, make_gnn_scorer  # noqa: E402

EVAL_SETS = config.EVAL_SETS


def list_eval_problems(eval_root: Path) -> list[Path]:
    problems = []
    for name in EVAL_SETS:
        set_dir = eval_root / name
        problems.extend(sorted(set_dir.glob("config_*.pddl"), key=lambda p: p.stem))
    return problems


def eval_plain(problem_path: Path, domain_path: Path, plan_fn, method: str) -> report.ProblemEvalResult:
    checker = get_checker()
    checker.clear_cache()
    _, problem = xmi.parse(str(domain_path), str(problem_path))
    generator = xmi.XSuccessorGenerator(problem)

    t0 = time.time()
    result = plan_fn(
        generator, generator.initial_state,
        max_expansion=config.EVAL_MAX_EXPANSION, time_limit_s=config.EVAL_TIME_LIMIT_S,
    )
    wall = time.time() - t0

    return report.ProblemEvalResult(
        problem=problem_path.stem, method=method, solved=result.solved,
        mp_calls=checker.cache_size(), expansions=result.expanded,
        plan_length=(len(result.plan) if result.solved else None),
        wall_time_s=wall, timed_out=(result.reason == "timeout"), n_plan_calls=1, reason=result.reason,
    )


def eval_ploi(problem_path: Path, domain_path: Path, plan_fn, scorer, method: str) -> report.ProblemEvalResult:
    """PLOI's incremental loop can call PLAN() several times (once per
    growing N). run_ploi's overall_time_limit_s (a wall-clock budget, not a
    SIGALRM) caps the whole call at config.EVAL_TIME_LIMIT_S: each PLAN()
    attempt gets only the time left, as in tomsilver/ploi's
    IncrementalPlanner, and the loop stops when it is used up.
    Not a SIGALRM wrapper: filter_action (inside every PLAN() call) uses
    its own SIGALRM timeout per call, and SIGALRM is one process-wide timer
    (the inner call's `signal.alarm(0)` would cancel an outer alarm).
    """
    checker = get_checker()
    checker.clear_cache()

    t0 = time.time()
    result = ploi.run_ploi(
        problem_path, domain_path, scorer, plan_fn,
        gamma=config.GAMMA, plan_max_expansion=config.EVAL_MAX_EXPANSION,
        plan_time_limit_s=config.EVAL_TIME_LIMIT_S, overall_time_limit_s=config.EVAL_TIME_LIMIT_S,
    )
    wall = time.time() - t0

    total_expansions = sum(pr.expanded for pr in result.plan_results)
    timed_out = result.reason == "overall_timeout" or any(pr.reason == "timeout" for pr in result.plan_results)

    return report.ProblemEvalResult(
        problem=problem_path.stem, method=method, solved=result.solved,
        mp_calls=checker.cache_size(), expansions=total_expansions,
        plan_length=(len(result.plan) if result.solved else None),
        wall_time_s=wall, timed_out=timed_out, n_plan_calls=result.n_plan_calls, reason=result.reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--mode", choices=["ploi_bfs", "ploi_hff", "bfs", "hff"], required=True)
    parser.add_argument("--model", type=Path, default=None, help="required for ploi_* modes")
    parser.add_argument("--features", choices=["auto", "default", "positional"], default="auto",
                        help="node features of the --model weights: auto (default) reads them from the weights "
                             "(6 node inputs = default, the reported Table 1 weights; 8 = positional, + cylinder "
                             "pose); default / positional states them and stops if the weights disagree")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    problems = list_eval_problems(args.eval_root)
    shard = problems[args.start:args.end]
    print(f"shard: {len(shard)} problems (of {len(problems)} total), mode={args.mode}")

    if args.mode in ("ploi_bfs", "ploi_hff"):
        if args.model is None:
            raise SystemExit("--model is required for ploi_bfs/ploi_hff")
        state_dict = torch.load(args.model, map_location="cpu")
        trained_with = features.feature_set_of(state_dict)
        if args.features not in ("auto", trained_with):
            raise SystemExit(f"--features {args.features}, but {args.model} has the {trained_with} node features")
        features.set_feature_set(trained_with)
        print(f"node features: {trained_with} ({features.NODE_DIM} inputs)")
        model = ObjectImportanceGNN()
        model.load_state_dict(state_dict)  # strict
        model.eval()
        scorer = make_gnn_scorer(model)
        plan_fn = plan_search.bfs_with_plan if args.mode == "ploi_bfs" else plan_search.gbfs_hff_with_plan
    else:
        plan_fn = plan_search.bfs_with_plan if args.mode == "bfs" else plan_search.gbfs_hff_with_plan

    results = []
    for i, problem_path in enumerate(shard):
        if args.mode in ("ploi_bfs", "ploi_hff"):
            r = eval_ploi(problem_path, args.domain, plan_fn, scorer, args.mode)
        else:
            r = eval_plain(problem_path, args.domain, plan_fn, args.mode)
        results.append(r)
        print(f"  [{i+1}/{len(shard)}] {r.problem}: solved={r.solved} mp_calls={r.mp_calls} "
              f"plan_len={r.plan_length} wall={r.wall_time_s:.1f}s")

    report.write_jsonl(results, args.out)
    print(f"wrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
