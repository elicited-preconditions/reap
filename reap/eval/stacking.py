"""3D stacking evaluation under the grip-aware check (MLB_CHECKER=honest_grip): protocol numbers, plans, continuous
verification of every plan, and the per-set, per-level report. Runs on one machine; the sharded steps run ``nproc``
processes at a time.

  1. protocol: GBFS (300 s, budget 16384, filter on) over the ID set (eval/) and the OOD set (eval_ood/),
     MP calls = unique queries (MP_UNIQUE=1) -> <out>/protocol.json
  2. plans: reap.eval.stacking_plans --plan_only (same model, check and protocol), one shard per problem
     -> <out>/plans/{eval,eval_ood}
  3. verification: reap.eval.stacking_verify run (``verify_shards`` shards): per-action walk and continuous
     verification of every plan (placement, motion, exact clearance) -> <out>/verify
  4. reap.eval.stacking_verify summarize and reap.eval.stacking_level_report -> <out>/REPORT.txt

Needs A19V2_PYLIB (the directory, or os.pathsep-joined directories, holding frantik and vamp). Fails loudly when the
protocol step evaluates no problem, a set errors, a shard fails or a per-problem record holds an error; a partial
result is still summarised first.

    python -m reap.eval.stacking preflight <dataset dir>
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

from reap.eval.runners import parallel, run_module
from reap.paths import data_root


def preflight(dd: str) -> None:
    """A missing dependency of the honest_grip check (frantik / vamp in A19V2_PYLIB, the quadprog QP solver) does not
    raise inside a search: the checker then rejects every action and every problem ends as a dead end. So check the
    dependencies, and that the checker accepts the first action of a verified dataset plan, before anything runs."""
    errs = []
    try:
        import qpsolvers
        if "quadprog" not in qpsolvers.available_solvers:
            errs.append("qpsolvers finds no 'quadprog' solver (pip install quadprog==0.1.13, see requirements.txt)")
    except ImportError as e:
        errs.append("qpsolvers is not importable: %s" % e)
    for p in os.environ.get("A19V2_PYLIB", "").split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    for mod in ("frantik", "vamp"):
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            errs.append("%s is not importable from A19V2_PYLIB: %s" % (mod, e))
    if errs:
        sys.exit("FATAL: " + "\n       ".join(errs))
    import reap.xmimir as xmi
    from reap.domains.multilevel_blocks import MultilevelBlocksFeasibilityChecker as Checker
    f = next(f for f in sorted(glob.glob(dd + "/eval/*.pddl")) if os.path.exists(f + ".plan"))
    _, prob = xmi.parse(dd + "/domain.pddl", f)
    gen = xmi.XSuccessorGenerator(prob)
    s = gen.initial_state
    txt = [l.strip() for l in open(f + ".plan") if l.strip().startswith("(")][0]
    name, *args = txt.strip("()").split()
    act = [x for x in gen.action_generator.generate_actions(s)
           if x.name.lower() == name.lower() and [o.name.lower() for o in x.objects] == [y.lower() for y in args]][0]
    if Checker().check(s, act):
        sys.exit("FATAL: the checker rejects the first action of a verified dataset plan (%s: %s); a dependency of the "
                 "honest_grip check is missing or broken" % (os.path.basename(f), txt))
    print("pre-flight ok: %s %s accepted" % (os.path.basename(f), txt))


def check_protocol(f: Path, log: Path) -> None:
    if not f.exists():
        tail = open(log, errors="ignore").read().splitlines()[-15:] if log.exists() else []
        sys.exit("FATAL: the protocol step wrote no result (%s); last lines of %s:\n  %s" % (f, log, "\n  ".join(tail)))
    d = json.load(open(f))
    bad = [r for r in d.get("results", []) if r.get("errored") or not r.get("total")]
    for r in bad:
        print("FATAL: protocol set %s: %s" % (r.get("set"), r.get("error") or "0 problems evaluated"), file=sys.stderr)
    if bad or not d.get("results"):
        sys.exit("FATAL: protocol step failed (see %s)" % log)
    probs = [p for r in d["results"] for p in r["problems"]]
    if not any(p["solved"] for p in probs) and all(p.get("filter_calls", 0) == p.get("infeasible", -1) for p in probs):
        sys.exit("FATAL: nothing solved and every feasibility check returned infeasible: the checker is broken (see %s)"
                 % log)
    print("protocol: " + ", ".join("%s %d/%d" % (r["set"], r["solved"], r["total"]) for r in d["results"]))


def check_records(o: Path, ni: int, no: int) -> list[str]:
    bad = []
    for sub, n in (("eval", ni), ("eval_ood", no)):
        fs = sorted(glob.glob(str(o / "plans" / sub / "*.json")))
        if len(fs) != n:
            bad.append("plans/%s: %d records for %d problems" % (sub, len(fs), n))
        bad += ["plans/%s/%s: %s" % (sub, os.path.basename(f), json.load(open(f))["error"])
                for f in fs if "error" in json.load(open(f))]
        vs = sorted(glob.glob(str(o / "verify" / "per_plan" / sub / "*.json")))
        bad += ["verify/%s/%s: %s" % (sub, os.path.basename(f), json.load(open(f))["error"])
                for f in vs if "error" in json.load(open(f))]
    return bad


def run(*, checkpoint, out, data="stacking", nproc=4, verify_shards=40, env=None):
    if not os.environ.get("A19V2_PYLIB"):
        sys.exit("set A19V2_PYLIB to the directory (or os.pathsep-joined directories) holding frantik and vamp")
    if os.environ.get("MLB_CHECKER") != "honest_grip":
        sys.exit("run with MLB_CHECKER=honest_grip (configs/stacking.yaml sets it)")
    dd = data_root() / data
    if not (dd / "eval").is_dir():
        sys.exit(f"FATAL: no {dd}/eval")
    ni, no = len(list((dd / "eval").glob("*.pddl"))), len(list((dd / "eval_ood").glob("*.pddl")))
    o = Path(out)
    for d in ("plans", "verify", "logs"):
        (o / d).mkdir(parents=True, exist_ok=True)
    print(f"model={checkpoint} data={data} ID={ni} OOD={no} out={o}", flush=True)
    quiet = {"MUJOCO_GL": "disabled", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    # 0. pre-flight
    run_module("reap.eval.stacking", ["preflight", dd], env={**(env or {}), **quiet})
    # 1. protocol numbers
    proto, log = o / "protocol.json", o / "logs" / "protocol.log"
    run_module("reap.eval.reproduce_eval",
               ["--domain", "multilevel-blocks", "--checkpoint", checkpoint, "--mode", "gbfs", "--sets", "all",
                "--n", 0, "--time_limit_s", 300, "--max_budget", 16384, "--out", proto],
               env={**(env or {}), "MUJOCO_GL": "disabled", "FILTER_ACTIONS": "1", "FILTER_TIMEOUT_S": "180",
                    "EVAL_DATA": data, "MP_UNIQUE": "1"}, log=log, check=False)
    check_protocol(proto, log)
    # 2. plans, one shard per problem
    fail = 0
    plans_env = {**(env or {}), **quiet, "FILTER_ACTIONS": "1", "FILTER_TIMEOUT_S": "180"}
    for lab, sub, n in (("eval", "eval", ni), ("eval_ood", "eval_ood", no)):
        jobs = [lambda k=k, lab=lab, sub=sub, n=n: run_module(
            "reap.eval.stacking_plans", ["--model", checkpoint, "--data", data, "--label", lab, "--shard", f"{k}/{n}",
                                          "--out", o / "plans", "--subdir", sub, "--plan_only"],
            env=plans_env, log=o / "logs" / f"plans_{lab}_{k}.log") for k in range(n)]
        if parallel(jobs, nproc):
            print(f"ERROR: some {lab} plan shards failed, see {o}/logs/plans_{lab}_*.log", file=sys.stderr)
            fail = 1
    # 3. continuous verification of every plan
    specs = ["--spec", f"eval={o / 'plans' / 'eval'}", "--spec", f"eval_ood={o / 'plans' / 'eval_ood'}"]
    verify_env = {**(env or {}), **quiet, "OPENBLAS_NUM_THREADS": "1"}
    jobs = [lambda k=k: run_module("reap.eval.stacking_verify",
                                   ["run", "--out", o / "verify", *specs, "--shard", f"{k}/{verify_shards}"],
                                   env=verify_env, log=o / "logs" / f"verify_{k}.log") for k in range(verify_shards)]
    if parallel(jobs, nproc):
        print(f"ERROR: some verify shards failed, see {o}/logs/verify_*.log", file=sys.stderr)
        fail = 1
    # 4. summary and report
    run_module("reap.eval.stacking_verify", ["summarize", "--out", o / "verify", *specs],
               env={**(env or {}), "MUJOCO_GL": "disabled"}, log=o / "logs" / "summarize.log")
    run_module("reap.eval.stacking_level_report", [o, "--out", o / "REPORT.txt"], env=env)
    bad = check_records(o, ni, no)
    for b in bad:
        print("ERROR: " + b, file=sys.stderr)
    if fail or bad:
        sys.exit(f"FATAL: {o}: see the errors above and {o}/logs/")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "preflight":
        preflight(sys.argv[2])
    else:
        sys.exit(__doc__)
