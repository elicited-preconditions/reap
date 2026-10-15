"""Uniform GBFS-deployment eval: one (domain, model, set) per invocation.

Runs GBFS (gbfs_value semantics, weight=0) with filter_action ON, and records
per problem: solved, expansions, plan length (steps), infeasible_steps,
mp_calls (unique motion-planner queries), total_filter_calls, the full
reconstructed plan, wall_time, termination_reason.

No retries: GBFS handles infeasibility by skipping the action and popping the
next-best node off the open list (that IS the backtracking); `val_retries`
applies only to the greedy rollout path.

Parallelism: --workers W shards the problem list across W spawned processes,
each with its OWN feasibility checker + MuJoCo env (the eval analog of
training's num_workers). The MuJoCo session is built once per worker and kept
alive across that worker's problems.

Usage:
    CONFINED_CHECKER=honest python -m reap.eval.eval_protocol \\
        --domain confined-shelf --set OOD --subdir eval_nonmonotone_16 \\
        --model_path checkpoints/confined/seed_1 --snapshot live --label seed_1 \\
        --output_dir $REAP_OUT/eval/protocol --time_limit_s 300 --max_budget 16384 --workers 8
"""
from __future__ import annotations

import argparse
import heapq
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
os.chdir(REPO_ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import reap.xmimir as xmi  # noqa: E402

from reap.eval.eval_config import EVAL_DOMAINS, set_subdir  # noqa: E402
from reap.eval.eval_all_datasets import latest_checkpoint  # noqa: E402
from reap.rl.data_layout import InputData  # noqa: E402
from reap.rl.av_star import sim  # noqa: E402 (registers domains + filter counters)
from reap.rl.av_star.sim import filter_action  # noqa: E402
from reap.rl.av_star.search import PrioritizedItem, next_real, _termination_reason  # noqa: E402
from reap.rl.av_star.agent import Model  # noqa: E402
from reap.rl.av_star.utils import StateEmbedder  # noqa: E402


def gbfs_value_eval(generator, initial_state, embedder, *, max_expansion, time_limit_s):
    """gbfs_value (weight=0) + came-from for plan reconstruction.

    Mirrors search.gbfs_value's loop exactly (infeasible -> skip via the open
    list, no retry cap); the filter_action calls are counted in sim._FILTER_STATS.
    The open list is ordered by -Q from the value head (get_q_values).
    Returns (expanded, plan_actions, solved, termination_reason, init_time).
    """
    t0 = time.time()
    # An init that already satisfies the goal is a 0-step solution, not a
    # failure. Without this check the search expands to max_budget and
    # reports solved=0.
    if initial_state.is_goal():
        return 0, [], True, "already_at_goal", time.time() - t0
    closed: set = set()
    closed.add(initial_state)  # prevent re-reach -> came back-edge -> reconstruction cycle
    came: dict = {}
    openh: list = []
    heapq.heappush(openh, PrioritizedItem(0, (initial_state, None)))
    init_time = time.time() - t0
    start = time.time()
    counter = 0
    goal = None
    while (openh and counter < max_expansion
           and (time.time() - start < time_limit_s or time_limit_s <= 0)):
        item = heapq.heappop(openh)
        parent, action = item.item
        node = parent
        if action is not None:
            if filter_action(parent, action):
                continue
            node = next_real(action.apply(parent), generator)
            if node in closed:
                continue
            closed.add(node)
            came[node] = (parent, action)
            if node.is_goal():
                goal = node
                break
        actions = list(generator.action_generator.generate_actions(node))
        actions.sort(key=lambda k: str(k))
        values, _ = embedder.get_q_values([node])
        counter += 1
        for a, v in zip(actions, values):
            heapq.heappush(openh, PrioritizedItem(-1.0 * float(v), (node, a)))

    plan = []
    if goal is not None:
        cur = goal; seen = set()
        while cur in came and came[cur][1] is not None and cur not in seen:
            seen.add(cur)
            parent, action = came[cur]
            plan.append(action)
            cur = parent
        plan.reverse()
    reason = _termination_reason(goal is not None, len(openh) == 0, counter,
                                 max_expansion, start, time_limit_s)
    return counter, plan, goal is not None, reason, init_time


def _eval_one_problem(prob, embedder, *, domain, set_kind, label, domain_name,
                      max_budget, time_limit_s):
    """Run one problem through gbfs_value_eval and build its result row."""
    gen = xmi.XSuccessorGenerator(prob)
    embedder.clean()
    sim.reset_filter_stats()
    t0 = time.time()
    expanded, plan, solved, reason, init_time = gbfs_value_eval(
        gen, gen.initial_state, embedder,
        max_expansion=max_budget, time_limit_s=time_limit_s)
    wall = time.time() - t0
    fstats = sim.get_filter_stats()
    chk = sim.get_checker_for(domain_name)
    mp_calls = len(chk._cache) if chk is not None and hasattr(chk, "_cache") else 0
    if chk is not None and hasattr(chk, "_cache"):
        chk._cache.clear()  # isolate next problem's mp_calls
    return {
        "domain": domain, "set": set_kind, "model": label,
        "problem": Path(prob.filepath).name,
        "solved": int(solved), "expanded": expanded, "steps": len(plan),
        "infeasible_steps": fstats["infeasible"], "mp_calls": mp_calls,
        "total_filter_calls": fstats["calls"], "wall_time": round(wall, 3),
        "termination_reason": reason,
        "plan": ";".join(str(a) for a in plan),
    }


def _run_shard(payload):
    """Spawned-worker entry: build model + data + checker, process a subset of
    problem indices. Each worker holds its own MuJoCo env (built once, lazily)."""
    (domain, set_kind, model_path, snapshot, label,
     time_limit_s, max_budget, shard_indices, subdir) = payload
    dom = EVAL_DOMAINS[domain]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ck = latest_checkpoint(REPO_ROOT / model_path) if snapshot == "best" else REPO_ROOT / model_path
    data = InputData(os.environ.get("EVAL_DATA") or dom.data, train_dir=subdir, instances="all",
                     domain_file=dom.domain_file)   # EVAL_DATA: optional data directory in place of dom.data
    problems = data.train_problems
    domain_name = data.domain.name
    model = Model.load_from_folder(str(ck))
    model.to(device)
    embedder = StateEmbedder(model)
    rows = []
    for i in shard_indices:
        rows.append(_eval_one_problem(
            problems[i], embedder, domain=domain, set_kind=set_kind, label=label,
            domain_name=domain_name, max_budget=max_budget, time_limit_s=time_limit_s))
    return rows


def main(domain, set_kind, model_path, label, *, output_dir,
         time_limit_s=300, max_budget=16384, snapshot="best", workers=1, subdir=None):
    if domain not in EVAL_DOMAINS:
        raise SystemExit(f"unknown domain {domain!r}; known: {list(EVAL_DOMAINS)}")
    dom = EVAL_DOMAINS[domain]
    subdir = subdir or set_subdir(domain, set_kind)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ck = latest_checkpoint(REPO_ROOT / model_path) if snapshot == "best" else REPO_ROOT / model_path
    data = InputData(os.environ.get("EVAL_DATA") or dom.data, train_dir=subdir, instances="all",
                     domain_file=dom.domain_file)   # EVAL_DATA: optional data directory in place of dom.data
    # which feasibility checker these numbers come from (raises if a shelf checker is not selected explicitly)
    from reap.domains import checker_info
    checker = checker_info.describe_data(data.pddl_domains_dir / dom.domain_file, [data.train_dir])
    print(checker_info.log_line(checker, "eval_protocol"), flush=True)
    n = len(data.train_problems)
    workers = max(1, min(int(workers), n))
    print(f"domain={domain} ({data.domain.name}) set={set_kind} subdir={os.environ.get('EVAL_DATA') or dom.data}/{subdir} "
          f"problems={n} model={ck.name} budget={max_budget} tl={time_limit_s}s workers={workers}", flush=True)

    if workers <= 1:
        rows = _run_shard((domain, set_kind, model_path, snapshot, label,
                           time_limit_s, max_budget, list(range(n)), subdir))
    else:
        shards = [list(range(i, n, workers)) for i in range(workers)]
        shards = [s for s in shards if s]
        payloads = [(domain, set_kind, model_path, snapshot, label,
                     time_limit_s, max_budget, s, subdir) for s in shards]
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            results = pool.map(_run_shard, payloads)
        rows = [r for sub in results for r in sub]

    df = pd.DataFrame(rows).sort_values("problem").reset_index(drop=True)
    df["checker"] = checker["label"]
    df["tampanda_commit"] = checker["tampanda_commit"]
    df["code_commit"] = checker["code_commit"]
    fname = f"{label}__{domain}__{set_kind}.csv"
    df.to_csv(out / fname, index=False)
    ns = int(df["solved"].sum())
    print(f"RESULT {label} {domain} {set_kind}: {ns}/{len(df)} "
          f"({100*ns/max(len(df),1):.1f}%) -> {out / fname}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--set", dest="set_kind", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output_dir", default=(os.environ.get("REAP_OUT") or "outputs") + "/eval/protocol")
    ap.add_argument("--time_limit_s", type=int, default=300)
    ap.add_argument("--max_budget", type=int, default=16384)
    ap.add_argument("--snapshot", default="best", choices=("best", "live"))
    ap.add_argument("--workers", type=int, default=1,
                    help="problem-level parallel workers (each its own checker/env)")
    ap.add_argument("--subdir", default=None, help="override eval set subdir")
    args = ap.parse_args()
    main(args.domain, args.set_kind, args.model_path, args.label,
         output_dir=args.output_dir, time_limit_s=args.time_limit_s,
         max_budget=args.max_budget, snapshot=args.snapshot, workers=args.workers, subdir=args.subdir)
