"""GBFS plans of 3D-stacking (multilevel-blocks) problems, one shard of a problem set per call (the plans step of
reap.eval.stacking; reap.eval.stacking_verify then verifies every plan).

Per problem of a shard: GBFS from --model (a checkpoint dir, loaded AS GIVEN -- the run root = final weights), 300 s /
budget 16384, the domain's current checker (MLB_CHECKER selects it).

Writes <out>/<label>/<problem>.json with the plan, the search statistics and the checker.

    python -m reap.eval.stacking_plans --model CHECKPOINT --data stacking --subdir eval \\
        --label eval --shard 3/19 --out DIR --plan_only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "disabled")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plan_problem(prob, embedder, a):
    import reap.xmimir as xmi
    from reap.rl.av_star import sim
    from reap.eval.eval_protocol import gbfs_value_eval

    t0 = time.time()
    gen = xmi.XSuccessorGenerator(prob)
    rec = {"problem": prob.filepath, "model": a.model}
    sim.reset_filter_stats()
    expanded, plan, solved, reason, _ = gbfs_value_eval(gen, gen.initial_state, embedder,
                                                       max_expansion=a.max_budget, time_limit_s=a.time_limit_s)
    rec["search"] = {"solved": bool(solved), "expanded": int(expanded), "reason": reason,
                     "filter_calls": int(sim._FILTER_STATS["calls"]), "wall_s": round(time.time() - t0, 1)}
    plan_txt = ["(%s %s)" % (x.name, " ".join(o.name for o in x.objects)) for x in plan]
    rec["plan"] = plan_txt
    rec["plan_len"] = len(plan_txt)
    rec["checker"] = os.environ.get("MLB_CHECKER")
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def run(a):
    import torch
    torch.set_num_threads(1)
    from reap.rl.data_layout import InputData
    from reap.rl.av_star.agent import Model
    from reap.rl.av_star.utils import StateEmbedder

    data = InputData(a.data, train_dir=a.subdir, instances="all", domain_file="domain.pddl")
    probs = sorted(data.train_problems, key=lambda p: os.path.basename(p.filepath))
    k, n = (int(x) for x in a.shard.split("/"))
    probs = probs[k::n]
    model = Model.load_from_folder(a.model); model.to("cpu"); model.eval()
    embedder = StateEmbedder(model)
    od = os.path.join(a.out, a.label); os.makedirs(od, exist_ok=True)
    for prob in probs:
        name = os.path.basename(prob.filepath).replace(".pddl", "")
        dst = os.path.join(od, name + ".json")
        if os.path.exists(dst):
            continue
        try:
            rec = plan_problem(prob, embedder, a)
        except Exception as e:
            rec = {"problem": prob.filepath, "error": "%s: %s" % (type(e).__name__, e), "tb": traceback.format_exc()}
        json.dump(rec, open(dst, "w"), indent=1)
        print(name, rec.get("plan_len"), rec.get("exec_ok"), rec.get("error", ""), rec.get("wall_s"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model"); ap.add_argument("--data"); ap.add_argument("--subdir", default="validation")
    ap.add_argument("--label"); ap.add_argument("--shard", default="0/1"); ap.add_argument("--out")
    ap.add_argument("--max_budget", type=int, default=16384); ap.add_argument("--time_limit_s", type=int, default=300)
    ap.add_argument("--plan_only", action="store_true", help="GBFS plans only (the only mode)")
    a = ap.parse_args()
    run(a)
