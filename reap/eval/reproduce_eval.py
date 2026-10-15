"""GBFS evaluation of one domain from an EXPLICIT checkpoint (no wandb); the protocol step of reap.eval.stacking.

Self-contained CLI that loads a model directory (Model.load_from_folder) and runs
the GBFS-deployment eval over the ID validation set and the
full (multi-bucket) OOD set defined in eval_config.EVAL_DOMAINS. Reuses
eval_protocol.gbfs_value_eval (no GBFS reimplementation).

The model is loaded ONCE (on CPU) and reused across every set.

Usage:
    python -m reap.eval.reproduce_eval --domain tabletop \\
        --checkpoint <checkpoint dir> \\
        --mode gbfs --sets all --time_limit_s 90

Runs from the repository root, on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# --- path bootstrap (mirror eval_protocol.py) ------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
os.chdir(REPO_ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import reap.xmimir as xmi  # noqa: E402

from reap.eval.eval_config import EVAL_DOMAINS  # noqa: E402
from reap.eval.eval_protocol import gbfs_value_eval  # noqa: E402
from reap.rl.data_layout import InputData  # noqa: E402
from reap.paths import data_root, out_root  # noqa: E402
from reap.rl.av_star import sim  # noqa: E402 (registers domains + filter counters)
from reap.rl.av_star.agent import Model  # noqa: E402
from reap.rl.av_star.utils import StateEmbedder  # noqa: E402


def _build_sets(dom, which):
    """Build the ordered list of sets to evaluate.

    Each entry: (label, kind, data, subdir, domain_file, pddl_domains_dir).
    ID -> one entry (kind="ID", label="validation").
    OOD -> one entry per dom.ood_cells (kind="OOD").
    `which` in {"id", "ood", "all"} filters.
    """
    sets = []
    if which in ("id", "all"):
        sets.append((
            "validation", "ID", dom.data, dom.id_set,
            dom.domain_file, dom.pddl_domains_dir,
        ))
    if which in ("ood", "all"):
        for c in dom.ood_cells:
            sets.append((
                c.label, "OOD", c.data, c.subdir,
                c.domain_file, c.pddl_domains_dir,
            ))
    return sets


def _extra_kwargs(pddl_domains_dir):
    return {"pddl_domains_dir": Path(pddl_domains_dir)} if pddl_domains_dir else {}


# MP_UNIQUE=1 (3D stacking, so that MP/plan counts unique queries as for the other environments): per problem,
# clear the feasibility checker's result cache before the search and record mp_calls = its size afterwards (distinct
# (state, action) checks), next to the raw filter_calls. Default off: no mp_calls column, the cache is kept.
_MP_UNIQUE = os.environ.get("MP_UNIQUE", "0") == "1"


def _cached_checkers():
    return [c for c in getattr(sim, "_CHECKERS", {}).values() if c is not None and hasattr(c, "_cache")]


def _run_gbfs(data, model, *, n, time_limit_s, max_budget):
    """GBFS eval (value head) over data.train_problems. Returns per-problem rows."""
    emb = StateEmbedder(model)
    problems = list(data.train_problems)
    if n:
        problems = problems[:n]
    rows = []
    for prob in problems:
        gen = xmi.XSuccessorGenerator(prob)
        emb.clean()
        sim.reset_filter_stats()
        if _MP_UNIQUE:          # isolate this problem's unique queries (eval_protocol.py does the same)
            for _c in _cached_checkers():
                _c._cache.clear()
        _t0 = time.time()
        expanded, plan, solved, reason, _ = gbfs_value_eval(
            gen, gen.initial_state, emb,
            max_expansion=max_budget, time_limit_s=time_limit_s)
        _fs = sim.get_filter_stats()
        rows.append({
            "name": Path(prob.filepath).name,
            "solved": int(solved),
            "plan_len": len(plan),
            # search effort
            "expanded": int(expanded),
            "reason": str(reason),
            "filter_calls": int(_fs.get("calls", 0)),
            "infeasible": int(_fs.get("infeasible", 0)),
            "wall_s": round(time.time() - _t0, 2),
        })
        if _MP_UNIQUE:          # unique (state, action) checks = the checker's result cache after this problem
            rows[-1]["mp_calls"] = int(sum(len(_c._cache) for _c in _cached_checkers()))
    return rows


def _run_set(label, kind, data_name, subdir, domain_file, pddl_domains_dir,
             *, mode, model, n, time_limit_s, max_budget):
    """Run one set under one mode. Returns list of {set, kind, mode, ...} result dicts."""
    extra = _extra_kwargs(pddl_domains_dir)
    out = []
    m = mode
    try:
        data = InputData(
            data_name, train_dir=subdir, instances="all",
            domain_file=domain_file, **extra)
        rows = _run_gbfs(
            data, model, n=n, time_limit_s=time_limit_s, max_budget=max_budget)
        n_solved = sum(r["solved"] for r in rows)
        total = len(rows)
        pct = 100.0 * n_solved / total if total else 0.0
        out.append({
            "set": label, "kind": kind, "mode": m,
            "solved": n_solved, "total": total, "pct": round(pct, 1),
            "errored": False, "problems": rows,
        })
        print(f"  [{m:6s}] {label:28s} ({kind})  {n_solved}/{total} = {pct:.1f}%",
              flush=True)
    except Exception as exc:  # noqa: BLE001 (one bad set must not kill the run)
        print(f"  [{m:6s}] {label:28s} ({kind})  ERROR: {exc}", flush=True)
        out.append({
            "set": label, "kind": kind, "mode": m,
            "solved": 0, "total": 0, "pct": 0.0,
            "errored": True, "error": str(exc), "problems": [],
        })
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=sorted(EVAL_DOMAINS),
                    help="a domain of reap.eval.eval_config.EVAL_DOMAINS")
    ap.add_argument("--checkpoint", required=True,
                    help="model dir loadable by Model.load_from_folder")
    ap.add_argument("--mode", default="gbfs", choices=("gbfs",),
                    help="GBFS deployment with the value head (the only mode)")
    ap.add_argument("--sets", default="all", choices=("id", "ood", "all"))
    ap.add_argument("--n", type=int, default=0,
                    help="limit problems per set (0 = all)")
    ap.add_argument("--time_limit_s", type=int, default=90)
    ap.add_argument("--max_budget", type=int, default=16384)
    ap.add_argument("--out", default=None,
                    help="json output path (default $REAP_OUT/eval/reproduce/<domain>.json)")
    args = ap.parse_args()

    dom = EVAL_DOMAINS[args.domain]
    out_path = Path(args.out) if args.out else (
        out_root() / "eval" / "reproduce" / f"{args.domain}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"domain={args.domain} checkpoint={args.checkpoint} mode={args.mode} "
          f"sets={args.sets} n={args.n or 'all'} tl={args.time_limit_s}s "
          f"budget={args.max_budget}", flush=True)

    model = Model.load_from_folder(args.checkpoint)
    model.to("cpu")

    sets = _build_sets(dom, args.sets)
    # EVAL_DATA (optional): evaluate on another dataset with the same layout; replaces the domain's default dataset name in every set that uses it.
    if os.environ.get("EVAL_DATA"):
        sets = [(lab, kind, os.environ["EVAL_DATA"] if dn == dom.data else dn, sub, df, pdd)
                for lab, kind, dn, sub, df, pdd in sets]
    # which feasibility checker these numbers come from -- per set, BEFORE anything runs (raises if a shelf
    # checker is not selected explicitly; _run_set's per-set error handling must not swallow that)
    from reap.domains import checker_info
    checkers = {}
    for lab, kind, dn, sub, df, pdd in sets:
        ds = (Path(pdd) if pdd else data_root()) / dn
        checkers[lab] = checker_info.describe_data(ds / df, [ds / sub])
        print(checker_info.log_line(checkers[lab], "reproduce_eval set=%s" % lab), flush=True)
    t0 = time.time()
    results = []
    for label, kind, data_name, subdir, domain_file, pddl_domains_dir in sets:
        results.extend(_run_set(
            label, kind, data_name, subdir, domain_file, pddl_domains_dir,
            mode=args.mode, model=model, n=args.n,
            time_limit_s=args.time_limit_s, max_budget=args.max_budget))

    for r in results:
        r["checker"] = checkers[r["set"]]["label"]
        for p in r["problems"]:
            p["checker"] = r["checker"]
    # --- summary table -----------------------------------------------------
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'set':30s} {'kind':5s} {'mode':7s} {'solved/total':>14s} {'pct':>7s}")
    print("-" * 70)
    agg = {}  # mode -> [solved, total]
    for r in results:
        flag = "  ERROR" if r["errored"] else ""
        print(f"{r['set']:30s} {r['kind']:5s} {r['mode']:7s} "
              f"{r['solved']:>6d}/{r['total']:<7d} {r['pct']:>6.1f}%{flag}")
        a = agg.setdefault(r["mode"], [0, 0])
        a[0] += r["solved"]
        a[1] += r["total"]
    print("-" * 70)
    for m, (s, t) in sorted(agg.items()):
        pct = 100.0 * s / t if t else 0.0
        print(f"{'AGGREGATE':30s} {'':5s} {m:7s} {s:>6d}/{t:<7d} {pct:>6.1f}%")
    print(f"\nwall_time={time.time() - t0:.1f}s", flush=True)

    payload = {
        "domain": args.domain,
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "sets": args.sets,
        "n": args.n,
        "time_limit_s": args.time_limit_s,
        "max_budget": args.max_budget,
        "eval_data": os.environ.get("EVAL_DATA"),
        "checkers": checkers,
        "tampanda_commit": checker_info.tampanda_commit(),
        "code_commit": checker_info.code_commit(),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
