#!/usr/bin/env python3
"""Regenerate PDDL problem instances into <out>/data/problems/train/
(<out> = $REAP_OUT/baselines/bt_h).

Pure string generation -- no xmimir/tampanda/MuJoCo dependency, runs with
any Python 3 interpreter:

    python3 baselines/bt_h/scripts/generate_instances.py

`train/` is the seed-demo pool consumed by scripts/collect_seed_demos.py
(K in config.K_TRAIN_RANGE, config.N_TRAIN_PER_K each), generated from
seed 0 on.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# outputs: $REAP_OUT/baselines/bt_h/
WORK = Path(__import__("os").environ.get("REAP_OUT") or "outputs") / "baselines" / "bt_h"

from bt_h import config  # noqa: E402
from bt_h.domain.problem_gen import emit_pddl, generate_batch  # noqa: E402


def _write_batch(instances, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for inst in instances:
        (out_dir / f"{inst.name}.pddl").write_text(emit_pddl(inst))
    print(f"wrote {len(instances)} instances -> {out_dir}")


def main() -> None:
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    train = generate_batch(
        prefix="train",
        k_counts={k: config.N_TRAIN_PER_K for k in config.K_TRAIN_RANGE},
        seed_start=0,
    )
    _write_batch(train, WORK / "data" / "problems" / "train")


if __name__ == "__main__":
    main()
