#!/usr/bin/env python3
"""Run classic_backtrack (Algorithm 1) over the seed-demo instance pool (the
same instances written to <out>/data/problems/train/ by
generate_instances.py, regenerated in memory here via the identical
generate_batch(prefix="train", ...) call, deterministic given the same
seeds) to produce demonstrations for Algorithm 3's bootstrap data collection.
<out> is $REAP_OUT/baselines/bt_h.

Resumable: writes each demo to <out>/data/demos/ as soon as it is solved,
and skips any instance already collected on a re-invocation -- safe to
re-run after an interruption without losing progress or re-spending
compute. Final tallies are always computed by re-reading <out>/data/demos/
from disk, so they are correct regardless of how many invocations it took
to finish.

    python baselines/bt_h/scripts/collect_seed_demos.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# outputs: $REAP_OUT/baselines/bt_h/
WORK = Path(__import__("os").environ.get("REAP_OUT") or "outputs") / "baselines" / "bt_h"

from bt_h._bootstrap import ensure_upstream_importable  # noqa: E402

ensure_upstream_importable()

from bt_h import config  # noqa: E402
from bt_h.checker import InstrumentedFeasibilityChecker  # noqa: E402
from bt_h.data_collection import collect_seed_demonstrations, load_demonstrations  # noqa: E402
from bt_h.domain.problem_gen import generate_batch  # noqa: E402


def main() -> None:
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    instances = generate_batch(
        prefix="train",
        k_counts={k: config.N_TRAIN_PER_K for k in config.K_TRAIN_RANGE},
        seed_start=0,
    )
    out_dir = WORK / "data" / "demos"
    already = sum(1 for i in instances if (out_dir / f"{i.name}.json").exists())
    if already:
        print(f"resuming: {already}/{len(instances)} instances already collected, skipping those")

    checker = InstrumentedFeasibilityChecker()
    _new_demos, logs = collect_seed_demonstrations(instances, checker, out_dir=out_dir, seed=0)

    log_path = out_dir / "_collection_log.json"
    prior_logs = json.loads(log_path.read_text()) if log_path.exists() else []
    log_path.write_text(json.dumps(prior_logs + logs, indent=2))

    all_demos = load_demonstrations(out_dir)
    by_k: dict[int, int] = {}
    for demo in all_demos:
        by_k[demo.k] = by_k.get(demo.k, 0) + 1
    k_counts = {k: config.N_TRAIN_PER_K for k in config.K_TRAIN_RANGE}

    print(f"\n{len(all_demos)}/{len(instances)} instances solved (all-time, across any resumed "
          f"invocations) -> {out_dir}")
    for k in sorted(k_counts):
        solved = by_k.get(k, 0)
        pct = 100 * solved / k_counts[k]
        print(f"  K={k}: {solved}/{k_counts[k]} solved ({pct:.0f}%)")
    print(f"this invocation: {checker.stats.total_calls} checker calls, "
          f"{checker.stats.total_wall_s:.1f}s wall time")


if __name__ == "__main__":
    main()
