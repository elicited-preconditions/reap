"""PLOI training labels for REAP's 600 training problems (python baselines/ploi/run.py labels regenerates the shipped
labels with --strategy single_plan, --plan-fn bfs_then_hff, --max-expansion 5000, --time-limit-s 120).

Same CLI/sharding conventions as SAHS's generate_training_data.py
(--problems-dir/--domain/--start/--end), plus --workers for the per-problem
process pool (parallel_pool.py).

Usage (one shard, sharded by --start/--end; python baselines/ploi/run.py labels
ploi runs every shard):
    python baselines/ploi/scripts/generate_training_labels.py \\
        --problems-dir $REAP_DATA/tabletop/train \\
        --domain $REAP_DATA/tabletop/domain.pddl \\
        --out-dir $REAP_OUT/baselines/ploi_labels/train --start 0 --end 50 --workers 16 \\
        --strategy single_plan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ploi._bootstrap import ensure_upstream_importable  # noqa: E402

ensure_upstream_importable()

from ploi.parallel_pool import run_parallel_label_generation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems-dir", type=Path, required=True)
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--plan-fn", choices=["bfs", "hff", "bfs_then_hff"], default="bfs_then_hff")
    # PLAN budget per labelling attempt (smaller than the evaluation budget)
    parser.add_argument("--max-expansion", type=int, default=5000)
    parser.add_argument("--time-limit-s", type=float, default=120.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--strategy", choices=["greedy", "single_plan"], default="greedy",
                         help="greedy: greedy backward removal, as the PLOI paper describes it; "
                              "single_plan: the strategy of the authors' released code "
                              "(run PLAN once on the full set, label the touched objects)")
    args = parser.parse_args()

    problem_paths = sorted(args.problems_dir.glob("*.pddl"), key=lambda p: p.stem)
    shard = problem_paths[args.start:args.end]
    print(f"shard: {len(shard)} problems (of {len(problem_paths)} total), workers={args.workers}, strategy={args.strategy}")

    results = run_parallel_label_generation(
        shard, args.domain, args.out_dir,
        plan_fn_name=args.plan_fn, n_workers=args.workers,
        plan_max_expansion=args.max_expansion, plan_time_limit_s=args.time_limit_s,
        checkpoint_every=args.checkpoint_every, strategy=args.strategy,
    )

    n_skipped = sum(1 for r in results if r.skipped)
    n_excluded = sum(1 for r in results if r.excluded)
    n_errors = sum(1 for r in results if r.error is not None)
    n_fresh = len(results) - n_skipped - n_excluded
    total_wall = sum(r.wall_time_s or 0.0 for r in results if not r.skipped)
    print(f"done: {n_fresh} fresh, {n_skipped} skipped (already labeled), "
          f"{n_excluded} excluded (PLAN couldn't solve full set, single_plan only), {n_errors} errors, "
          f"total worker-seconds={total_wall:.0f}")
    for r in results:
        if r.error is not None:
            print(f"  ERROR {r.problem_path}: {r.error}")


if __name__ == "__main__":
    main()
