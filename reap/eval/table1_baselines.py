#!/usr/bin/env python3
"""Table 1: every row (REAP, hFF, BrFS and the baselines SAHS, PLOI, BT-h and PDDLStream) from the per-problem outputs.

MP/plan is the number of distinct (state, action) motion-planner queries of a problem, with the checker cache cleared
at the start of the problem; the table gives its mean and median over the solved problems (unsolved problems are left
out). Exp./steps is search expansions over plan length, checks/exp. is MP/plan over expansions, and time is the mean
over solved problems. Columns, labels, rounding and row order are those of Table 1; a cell that Table 1 gives as n/a
is printed as n/a (Row.na).

Each loader reads the per-problem outputs that the README commands write for one method (layout below the imports)
into the same per-problem record.

Every row is read under the 300 s per-problem cap of Table 1 (--cap, default 300): a solve whose total wall time
exceeds the cap counts as unsolved. This affects only PLOI, whose overall limit is checked only between planner calls;
every other evaluation stops at the cap.

    python -m reap.eval.table1_baselines                      # the table
    python -m reap.eval.table1_baselines --root DIR           # the outputs under DIR (default: $REAP_OUT, else outputs)
    python -m reap.eval.table1_baselines --only ploi,reap     # subset
    python -m reap.eval.table1_baselines --csv out.csv        # the printed columns, one line per row
    python -m reap.eval.table1_baselines --cap 0              # without the 300 s per-problem cap
"""
from __future__ import annotations

import argparse
import csv
import functools
import glob
import json
import math
import os
import statistics as st
import sys
from dataclasses import dataclass, replace
from pathlib import Path

# The outputs of the README commands under <root> (--root; default $REAP_OUT, else outputs), as the Table 1 reports
# (this one, table1_report, figure3, pddlstream_appendix) read them:
#   tabletop/reap_gbfs/seed_<k>/per_problem.csv     REAP-GBFS, k = 1..5 (scripts/eval.py)
#   tabletop/reap_greedy/seed_<k>/per_problem.csv   REAP-greedy
#   tabletop/hff/hff__<set>.csv, tabletop/brfs/brfs__<set>.csv
#   baselines/sahs/shard_<n>.json, baselines/ploi_hff/shard_<n>.jsonl, baselines/ploi_brfs/shard_<n>.jsonl,
#   baselines/bt_h/per_problem.jsonl                (baselines/<name>/run.py eval)
#   baselines/pddlstream/<configuration>/NNN.csv    test_stream_ffastar2, test_stream_ffeager, cell_generator_ffeager,
#                                                   informed_cells_occlusion (baselines/pddlstream/bench/run.py)
SEEDS = ["seed_%d" % k for k in range(1, 6)]   # the five seeds of Table 1
REAP_GBFS, REAP_GREEDY = "reap_gbfs", "reap_greedy"
HFF, BRFS = "hff", "brfs"
SAHS, PLOI_HFF, PLOI_BRFS = "sahs", "ploi_hff", "ploi_brfs"
BT_H = "bt_h/per_problem.jsonl"
PDDLSTREAM = {"ffastar2": "pddlstream/test_stream_ffastar2", "ffeager": "pddlstream/test_stream_ffeager",
              "gen_ffeager": "pddlstream/cell_generator_ffeager",
              "informed_cells_occlusion": "pddlstream/informed_cells_occlusion"}

TT: Path = None   # <root>/tabletop, set by main()
BL: Path = None   # <root>/baselines, set by main()


def roots(root=None) -> tuple:
    """(<root>/tabletop, <root>/baselines); ``root`` defaults to $REAP_OUT, else outputs."""
    r = Path(root or os.environ.get("REAP_OUT") or "outputs")
    return r / "tabletop", r / "baselines"


@functools.lru_cache(maxsize=None)
def reference_plan_length(dataset: str, problem: str) -> int:
    """Number of actions in the reference plan of a Table 1 problem, $REAP_DATA/tabletop/eval/<set>/<problem>.plan."""
    name = Path(problem).name
    name = name if name.endswith(".pddl") else name + ".pddl"
    f = Path(os.environ.get("REAP_DATA") or "data") / "tabletop" / "eval" / dataset / (name + ".plan")
    if not f.exists():
        raise SystemExit(f"no reference plan {f}. The problem sets are missing: run "
                         "bash scripts/download_assets.sh datasets, or point REAP_DATA at the data/ directory of the "
                         "dataset asset.")
    return sum(1 for l in f.read_text().splitlines() if l.strip().startswith("("))

# --------------------------------------------------------------------------
# per-problem record
# --------------------------------------------------------------------------


@dataclass
class Rec:
    """One problem under one method. `mp_undedup` and `expansions` are None when the method's evaluation does not
    record them (not 0)."""

    method: str
    problem: str
    solved: bool
    mp_dedup: int | None          # distinct (state, action) queries (MP/plan)
    mp_undedup: int | None        # every call that reached the planner
    expansions: int | None        # search nodes opened
    plan_length: int | None
    wall_s: float | None
    seed: str | None = None


@dataclass
class Row:
    """One row of the table: its label, where its MP/plan count is computed, and the loader of its records."""

    key: str
    label: str
    kind: str                      # "aggregate"
    source: str                    # where the MP/plan count of the row is computed
    loader: object = None          # callable() -> list[Rec]
    na: tuple = ()                 # columns that Table 1 gives as n/a: "checks_per_exp", "eps"

    def load(self):
        return apply_cap(self.loader()) if self.loader else []


# Per-problem time cap of Table 1, in seconds (--cap; 0 or None = off).
CAP_S: float | None = 300.0


def apply_cap(recs) -> list:
    """A record solved after more than CAP_S seconds of total wall time counts as unsolved, as a timeout at the cap
    does."""
    out = []
    for r in recs:
        if CAP_S and r.solved and r.wall_s is not None and r.wall_s > CAP_S:
            r = replace(r, solved=False)
        out.append(r)
    return out


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------


def _csv_rows(path: Path):
    with path.open() as fh:
        yield from csv.DictReader(fh)


# The five seeds of Table 1 (seed_1 ... seed_5).


def load_reap(subdir: str, label: str, has_expansions: bool = True):
    """REAP-GBFS / REAP-greedy: mp_calls is sim.get_mp_calls(), the size of the checkers' verdict caches, which the
    Table 1 evaluation clears at the start of every problem (reset_filter_stats(clear_cache=True)). total_filter_calls
    counts every query.
    """

    def _load():
        root = TT / subdir
        seeds = SEEDS
        out = []
        for seed in seeds:
            path = root / seed / "per_problem.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"{label}: seed {seed} is missing at {path} (the row needs every seed of Table 1).")
            for r in _csv_rows(path):
                solved = r["solved"] == "1"
                out.append(Rec(
                    method=label, problem=r["problem"], solved=solved,
                    mp_dedup=int(r["mp_calls"]),
                    mp_undedup=int(r["total_filter_calls"]),
                    # greedy is a policy rollout: it opens no search node, so the row has no exp./steps
                    expansions=int(r["expanded"]) if has_expansions else None,
                    plan_length=int(r["steps"]) if solved else None,
                    wall_s=float(r["wall_time"]), seed=seed,
                ))
        return out

    return _load


def load_ploi(subdir: str, label: str):
    """PLOI + hFF / PLOI + BrFS: mp_calls is the size of the checker cache, cleared per problem
    (baselines/ploi/scripts/run_evaluation.py). The JSONL records hold no count of every query (mp_undedup None)."""

    def _load():
        out = []
        for p in sorted((BL / subdir).glob("shard_*.jsonl")):
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                out.append(Rec(
                    method=label, problem=d["problem"], solved=d["solved"],
                    mp_dedup=d["mp_calls"], mp_undedup=None,
                    expansions=d["expansions"], plan_length=d["plan_length"],
                    wall_s=d["wall_time_s"],
                ))
        return out

    return _load


def load_sahs():
    """SAHS: MP/plan is total_calls, the distinct queries of feature building (PreFree, ManipFree and Occludes, each an
    IK and RRT query) plus those of plan realization."""

    def _load():
        out = []
        for p in sorted((BL / SAHS).glob("shard_*.json")):
            for d in json.loads(p.read_text()):
                out.append(Rec(
                    method="SAHS", problem=d["problem"],
                    solved=d["solved"],
                    mp_dedup=d["total_calls"], mp_undedup=None,
                    expansions=d["expansions"], plan_length=d["plan_length"],
                    wall_s=d["wall_s"],
                ))
        return out

    return _load


def load_bt_h(method: str, label: str):
    """BT-h: MP/plan is the number of distinct queries, primary_calls - cache_hits. Retries with force_refresh replace
    a cached verdict (baselines/bt_h/bt_h/checker.py) and are counted only in mp_calls_dedup (mp_undedup)."""

    def _load():
        out = []
        for line in (BL / BT_H).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d["method"] != method:
                continue
            out.append(Rec(
                method=label, problem=d["instance"], solved=d["solved"],
                mp_dedup=d["primary_calls"] - d["cache_hits"],
                mp_undedup=d["mp_calls_dedup"],   # every planner invocation
                expansions=d["primary_calls"],
                plan_length=d["plan_length"], wall_s=d["wall_time_s"],
            ))
        return out

    return _load


def load_csv_rows(path_glob, label: str, reader: str, fallback_glob=""):
    """Per-problem CSVs of the hFF / BrFS search baselines and of PDDLStream. ``fallback_glob``: the per-set CSVs of
    eval_single_model.py when there is no per_problem.csv (written only at the end of a run)."""

    def _load():
        paths = sorted(glob.glob(str(path_glob)))
        if not paths and fallback_glob:
            paths = sorted(glob.glob(str(fallback_glob)))
        if not paths:
            return []
        out = []
        for p in paths:
            for r in _csv_rows(Path(p)):
                solved = str(r.get("solved")) == "1"
                mp_d = r.get("mp_calls")
                out.append(Rec(
                    method=label, problem=r.get("problem", ""), solved=solved,
                    mp_dedup=int(mp_d) if mp_d not in (None, "") else None,
                    mp_undedup=(int(r["total_filter_calls"])
                                if r.get("total_filter_calls") not in (None, "") else None),
                    expansions=(int(r["expanded"])
                                if reader == "search" and r.get("expanded") not in (None, "") else None),
                    plan_length=int(r["steps"]) if solved and r.get("steps") else None,
                    wall_s=float(r["wall_time"]) if r.get("wall_time") else None,
                ))
        return out

    return _load


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

SUITE_N = 300   # problems of Table 1

PDS_ARMS = [
    ("pddlstream-ffastar2", "PDDLStream (test-stream, ff-astar2)", "ffastar2"),
    ("pddlstream-ffeager", "PDDLStream (test-stream, ff-eager)", "ffeager"),
    ("pddlstream-gen", "PDDLStream (cell-generator, ff-eager)", "gen_ffeager"),
    ("pddlstream-informed_cells_occlusion", "PDDLStream (informed cells + occlusion axiom)",
     "informed_cells_occlusion"),
]


def build_registry() -> dict:
    reg: dict = {}

    # REAP rows: mean +- population sd over the five seeds of Table 1; REAP-greedy opens no search node
    reg["reap-gbfs"] = Row(
        key="reap-gbfs", label="REAP-GBFS", kind="aggregate",
        source="reap/rl/av_star/sim.py (get_mp_calls: size of the verdict cache)",
        loader=load_reap(REAP_GBFS, "REAP-GBFS"),
    )
    reg["reap-greedy"] = Row(
        key="reap-greedy", label="REAP-greedy", kind="aggregate",
        source="reap/rl/av_star/sim.py (get_mp_calls: size of the verdict cache)",
        loader=load_reap(REAP_GREEDY, "REAP-greedy",
                         has_expansions=False),
    )
    reg["ploi_hff"] = Row(
        key="ploi_hff", label="PLOI + hFF", kind="aggregate",
        source="baselines/ploi/scripts/run_evaluation.py (mp_calls = checker.cache_size())",
        loader=load_ploi(PLOI_HFF, "PLOI + hFF"),
    )
    reg["ploi_brfs"] = Row(
        key="ploi_brfs", label="PLOI + BrFS", kind="aggregate",
        source="baselines/ploi/scripts/run_evaluation.py (mp_calls = checker.cache_size())",
        loader=load_ploi(PLOI_BRFS, "PLOI + BrFS"),
    )
    # SAHS: Table 1 gives no checks/exp. (n/a)
    reg["sahs"] = Row(
        key="sahs", label="SAHS", kind="aggregate",
        source="baselines/sahs/scripts/evaluate.py (total_calls = feature_building + realization)",
        loader=load_sahs(), na=("checks_per_exp",),
    )
    # BT-h: one (action, theta) per search node, so checks/exp. is 1 by construction; Table 1 gives both n/a
    reg["bt_h"] = Row(
        key="bt_h", label="BT-h, given the plan ordering", kind="aggregate",
        source="baselines/bt_h/bt_h/checker.py + controller.py (primary calls - hits)",
        loader=load_bt_h("bt_h", "BT-h, given the plan ordering"), na=("checks_per_exp", "eps"),
    )

    for mode, label in ((HFF, "hFF"), (BRFS, "BrFS")):
        reg[f"search-{mode}"] = Row(
            key=f"search-{mode}", label=label, kind="aggregate",
            source="reap/eval/eval_single_model.py (search modes hff / bfs), the REAP rows' instrumented rollouts()",
            loader=load_csv_rows(TT / mode / "per_problem.csv", label, "search",
                                 fallback_glob=TT / mode / "*__*.csv"),
        )

    # PDDLStream: no expansion count, so exp./steps is n/a
    for key, label, arm in PDS_ARMS:
        reg[key] = Row(
            key=key, label=label, kind="aggregate",
            source="baselines/pddlstream/bench/bench_problem*.py (passive observer of the checker queries)",
            loader=load_csv_rows(BL / PDDLSTREAM[arm] / "*.csv", label, "pds"),
        )
    # the row order of Table 1
    order = ["reap-gbfs", "reap-greedy", f"search-{BRFS}", f"search-{HFF}", "sahs", "ploi_hff", "ploi_brfs", "bt_h"]
    return {k: reg[k] for k in order + [k for k, _, _ in PDS_ARMS]}


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def _stats_block(recs: list) -> dict:
    """The statistics of one run: one seed's records, or all records of a single-run row."""
    n = len(recs)
    solved = [r for r in recs if r.solved]
    if not solved:
        return {}
    d = {
        "solve_pct": 100 * len(solved) / n if n else None,
        "mpd_mean": _mean([r.mp_dedup for r in solved]),
        "mpd_median": _median([r.mp_dedup for r in solved]),
        "undedup_mean": _mean([r.mp_undedup for r in solved]),
        "exp_mean": _mean([r.expansions for r in solved if r.expansions is not None]),
        "plan_len_mean": _mean([r.plan_length for r in solved]),
        "time_mean": _mean([r.wall_s for r in solved]),
    }
    # exp./steps: per-problem ratio, then mean and median; None for rows without an expansion count
    ratios = [r.expansions / r.plan_length for r in solved
              if r.expansions is not None and r.plan_length]
    d["eps_mean"] = _mean(ratios) if ratios else None
    d["eps_median"] = _median(ratios) if ratios else None
    # checks/exp.: ratio of the means; n/a when it is 1 by construction (one distinct query per search node)
    if d["exp_mean"] and d["mpd_mean"] is not None:
        cpe = d["mpd_mean"] / d["exp_mean"]
        d["checks_per_exp"] = None if math.isclose(cpe, 1.0, abs_tol=0.02) else cpe
    else:
        d["checks_per_exp"] = None
    return d


def _pstdev(xs):
    """Population sd (numpy's np.std), the +- of Table 1."""
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def summarise(ad: Row, recs: list) -> dict:
    n = len(recs)
    solved = [r for r in recs if r.solved]
    row = {
        "key": ad.key, "method": ad.label, "kind": ad.kind,
        "n": n, "n_solved": len(solved), "source": ad.source,
        "solve_pct": round(100 * len(solved) / n, 1) if n else None,
    }
    if not solved:
        row["status"] = "no data"
        return row
    # a row read from an unfinished run (fewer than 300 problems) is labelled partial; its count goes to stderr
    row["status"] = "ok"
    if n < SUITE_N:
        row["partial"] = n
        row["method"] = f"{ad.label}  [partial]"
        print(f"{ad.label}: {n} problems, not the {SUITE_N} of Table 1", file=sys.stderr)

    seeds = sorted({r.seed for r in solved if r.seed})
    if len(seeds) > 1:
        # seeded row: the statistics of each seed, then their mean and population sd over the seeds
        per = [_stats_block([r for r in recs if r.seed == s]) for s in seeds]
        per = [p for p in per if p]
        for k in per[0]:
            vals = [p[k] for p in per if p.get(k) is not None]
            row[k] = st.fmean(vals) if vals else None
        row["mpd_std"] = _pstdev([p["mpd_mean"] for p in per])
        row["solve_std"] = _pstdev([p["solve_pct"] for p in per])
        row["n_seeds"] = len(seeds)
        row["solve_pct"] = row["solve_pct"]
    else:
        row.update(_stats_block(recs))
    if "eps" in ad.na:
        row["eps_mean"] = row["eps_median"] = None
    if "checks_per_exp" in ad.na:
        row["checks_per_exp"] = None
    return row


def fmt(v, nd=2, dash="--"):
    if v is None:
        return dash
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def render(rows: list) -> str:
    """The rows in the columns and rounding of Table 1."""
    hdr = ["Method", "Solve (%)", "MP / plan [med.]", "Checks/exp.", "Exp./steps [med.]", "Time (s/solved)"]
    out = ["| " + " | ".join(hdr) + " |",
           "|" + "---|" * len(hdr)]
    for r in rows:
        if r["status"] != "ok":
            out.append(f"| {r['method']} | " + " | ".join([f"_{r['status']}_"] + [""] * 4) + " |")
            continue
        mpd = fmt(r["mpd_mean"], 1)
        if r.get("mpd_std") is not None:
            mpd += f" ± {r['mpd_std']:.1f}"
        solve = fmt(r["solve_pct"], 1)
        if r.get("solve_std") is not None:
            solve += f" ± {r['solve_std']:.1f}"
        eps = ("n/a" if r["eps_mean"] is None else f"{r['eps_mean']:.2f} [{r['eps_median']:.2f}]")
        out.append("| " + " | ".join([
            r["method"], solve, f"{mpd} [{float(r['mpd_median']):.1f}]",
            fmt(r["checks_per_exp"], 2, "n/a"), eps, fmt(r["time_mean"], 1),
        ]) + " |")
    return "\n".join(out)


def main(argv=None) -> None:
    global TT, BL, CAP_S
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated row keys or substrings")
    ap.add_argument("--root", default=None,
                    help="outputs of the README commands: <root>/tabletop, <root>/baselines (default: $REAP_OUT, "
                         "else outputs)")
    ap.add_argument("--csv", type=Path, help="write the printed columns here, one line per row (full precision)")
    ap.add_argument("--list", action="store_true", help="list the row keys and exit")
    ap.add_argument("--cap", type=float, default=300.0,
                    help="per-problem time cap in seconds: a solve with a larger total wall time "
                         "counts as unsolved (default 300, the cap of Table 1; 0 = off)")
    args = ap.parse_args(argv)
    CAP_S = args.cap or None
    TT, BL = roots(args.root)

    reg = build_registry()
    if args.list:
        for k, a in reg.items():
            print(f"{k:24} {a.kind:10} {a.label}")
        return

    keys = list(reg)
    if args.only:
        pats = [s.strip() for s in args.only.split(",") if s.strip()]
        keys = [k for k in keys if any(p in k for p in pats)]
        if not keys:
            raise SystemExit(f"no row matches {args.only!r}; try --list")

    rows = []
    for k in keys:
        ad = reg[k]
        try:
            recs = ad.load()
        except FileNotFoundError as e:
            print(f"[warn] {k}: {e}", file=sys.stderr)
            recs = []
        rows.append(summarise(ad, recs))

    print("Table 1, open tabletop (REAP rows: mean ± population sd over the seeds; all baseline rows are single runs)")
    print(render(rows))

    if args.csv:
        cols = ["key", "method", "status", "solve_pct", "solve_std", "mpd_mean", "mpd_std", "mpd_median",
                "checks_per_exp", "eps_mean", "eps_median", "time_mean"]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
