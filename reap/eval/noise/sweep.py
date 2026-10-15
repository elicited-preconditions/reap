#!/usr/bin/env python3
"""One (search_mode, noise_model, value) sweep cell over the 300-problem
open-tabletop suite. Plans are executed only for problems REAP solves at
sigma=0/p=0 (the definition of the headline metric).

Usage:
    python -m reap.eval.noise.sweep --checkpoint CKPT --mode gbfs --noise n1 --value 0.010 --n 20
    python -m reap.eval.noise.sweep --checkpoint CKPT --mode greedy --noise n2 --value 0.02
    python -m reap.eval.noise.sweep --checkpoint CKPT --mode gbfs --noise n3 --value 0.05

`--value` is sigma IN METERS for n1 (e.g. 0.010 = 10mm), or the raw
probability for n2/n3.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


import reap.xmimir as xmi  # noqa: E402

from reap.paths import out_root  # noqa: E402
from reap.rl.av_star.agent import Model  # noqa: E402
from reap.rl.av_star.utils import StateEmbedder  # noqa: E402

from reap.eval.noise import config  # noqa: E402
from reap.eval.noise.execution import (  # noqa: E402
    execute_plan_n1, execute_plan_n2, execute_plan_n3, plan_for_mode,
)


def iter_problems(n: int):
    domain_path = str(config.domain_path())
    count = 0
    for cell in config.EVAL_CELLS:
        cell_dir = config.eval_cell_dir(cell)
        for pddl_file in sorted(cell_dir.glob("config_*.pddl")):
            if n and count >= n:
                return
            _domain, problem = xmi.parse(domain_path, str(pddl_file))
            yield cell, pddl_file.name, problem
            count += 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="checkpoint directory (loaded as given)")
    ap.add_argument("--mode", choices=("gbfs", "greedy"), required=True)
    ap.add_argument("--noise", choices=("n1", "n2", "n3"), required=True)
    ap.add_argument("--value", type=float, required=True,
                     help="sigma in METERS for n1 (e.g. 0.010), probability for n2/n3")
    ap.add_argument("--n", type=int, default=0, help="limit total problems (0 = all 300)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    model = Model.load_from_folder(str(args.checkpoint))
    model.to("cpu")

    rows = []
    n_solved_baseline = 0
    n_attempted = 0
    t0 = time.time()
    for cell, name, problem in iter_problems(args.n):
        gen = xmi.XSuccessorGenerator(problem)
        emb = StateEmbedder(model, low_memory=True)
        problem_name = f"{cell}/{name}"
        try:
            solved, plan = plan_for_mode(args.mode, gen, gen.initial_state, emb)
        except Exception as exc:  # noqa: BLE001
            print(f"  {problem_name}  BASELINE PLAN ERROR: {exc}", flush=True)
            continue
        if not solved:
            continue  # only execute plans REAP solves at sigma=0/p=0
        n_solved_baseline += 1

        gen2 = xmi.XSuccessorGenerator(problem)
        emb2 = StateEmbedder(model, low_memory=True)
        try:
            if args.noise == "n1":
                result = execute_plan_n1(problem_name, args.mode, gen2, emb2, gen2.initial_state, plan, sigma_m=args.value)
            elif args.noise == "n2":
                result = execute_plan_n2(problem_name, args.mode, gen2, emb2, gen2.initial_state, plan, occupancy_flip_p=args.value)
            else:
                result = execute_plan_n3(problem_name, args.mode, gen2, emb2, gen2.initial_state, plan, failure_p=args.value)
        except Exception as exc:  # noqa: BLE001 -- one bad problem must not kill the sweep cell
            print(f"  {problem_name}  EXECUTION ERROR: {exc}", flush=True)
            continue
        n_attempted += 1
        rows.append(asdict(result))
        print(f"  {problem_name}  completed_without_replanning={result.completed_without_replanning} "
              f"replanned={result.replanned} replan_solved={result.replan_solved}", flush=True)

    completed = sum(1 for r in rows if r["completed_without_replanning"])
    replanned = sum(1 for r in rows if r["replanned"])
    replan_recovered = sum(1 for r in rows if r["replanned"] and r["replan_solved"])
    print(f"\n=== mode={args.mode} noise={args.noise} value={args.value} ===")
    print(f"baseline-solved: {n_solved_baseline}, executed: {n_attempted}")
    if n_attempted:
        print(f"completed without replanning: {completed}/{n_attempted} = {100.0*completed/n_attempted:.1f}%")
        print(f"replanned: {replanned}, recovered: {replan_recovered}"
              + (f" ({100.0*replan_recovered/replanned:.1f}%)" if replanned else ""))
    print(f"wall_time={time.time()-t0:.1f}s")

    out_path = Path(args.out) if args.out else (
        out_root() / "noise" / "sweep" / f"{args.mode}_{args.noise}_{args.value}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "mode": args.mode, "noise": args.noise, "value": args.value,
        "n_solved_baseline": n_solved_baseline, "n_attempted": n_attempted,
        "rows": rows,
    }, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
