"""Table 1 (open tabletop) rows of REAP-GBFS, REAP-greedy, BrFS and hFF from their per-problem CSVs, and the table of
the five REAP runs of Table 1 in the appendix "Open-Tabletop Ablations and Sample Efficiency".

Columns, as in the paper: Solve (%) · MP / plan [med.] · Checks/exp. · Exp./steps [med.] · Time (s/solved). The statistics are those of reap.eval.table1_baselines
(_stats_block / summarise), re-implemented here:
  per run      over SOLVED problems: MP mean and median (mp_calls, the distinct (state, action) queries), exp/step =
               per-problem expanded / steps, then mean and median; checks/exp = MP mean / expanded mean (n/a when
               the run has no expansion count, or when the ratio is 1 within 0.02); time = mean wall_time.
  REAP rows    per seed, then the mean over the five seeds of every column; "±" = population sd over seeds
               of the solve % and of the MP mean; [median] = mean of the per-seed medians.
  BrFS / hFF   single runs: the statistics over all their problems.
REAP-greedy opens no search node (expanded is 0), so its checks/exp and exp/step are n/a.
The five runs (appendix table): per seed, REAP-GBFS solve %, exp/step mean, MP mean [median] and REAP-greedy solve %
and plan length (mean steps over its solved problems), grouped as in the paper (RUNS: checkpoints seed_4 and seed_5
were trained without a time limit, seed_1 to seed_3 with 60 s); per group the mean of the per-seed values, with the
population sd of the solve % for the three-seed group, as in the paper. A one-seed group has no mean row.
A solve whose total wall time is over the 300 s per-problem cap (--cap; 0 = off) counts as unsolved, as in
table1_baselines.py; these evaluations stop at the cap, so it only applies to a run that did not.

    python -m reap.eval.table1_report [--root DIR] [--seeds seed_1,seed_2,...] [--cap 300]
Inputs: the outputs of scripts/eval.py under <root>/tabletop (--root; default $REAP_OUT, else outputs):
reap_gbfs/seed_<k>/ and reap_greedy/seed_<k>/ (per_problem.csv or seed_<k>__<set>.csv), hff/hff__<set>.csv,
brfs/brfs__<set>.csv. Seeds that were not evaluated, and runs with fewer than the 300 problems, are named on stderr,
never silently averaged in.
"""
import argparse
import csv
import glob
import math
import os
import statistics as st
import sys

from reap.eval import table1_baselines as T

ROOT = None   # <root>/tabletop, set by main()
CAP = 300.0   # per-problem time cap, seconds (--cap; 0 = off)
SEEDS = T.SEEDS
SUITE_N = 300  # problems of Table 1


def rows_of(files):
    out = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            out[(r.get("dataset") or os.path.basename(f), r["problem"])] = r      # one row per (set, problem)
    return list(out.values())


def num(r, k, f=float):
    v = r.get(k)
    return f(float(v)) if v not in (None, "") else None


def block(rows, has_expansions=True):
    """Statistics of one run (table1_baselines._stats_block)."""
    sol = [r for r in rows if str(r["solved"]) == "1"
           and not (CAP and num(r, "wall_time") is not None and num(r, "wall_time") > CAP)]
    if not sol:
        return {"n": len(rows), "k": 0}
    mp = [num(r, "mp_calls", int) for r in sol if num(r, "mp_calls") is not None]
    exp = [num(r, "expanded", int) for r in sol] if has_expansions else []
    eps = [num(r, "expanded") / num(r, "steps") for r in sol if has_expansions and num(r, "steps")]
    d = {"n": len(rows), "k": len(sol), "solve": 100.0 * len(sol) / len(rows),
         "mp": st.fmean(mp) if mp else None, "mp_med": st.median(mp) if mp else None,
         "exp": st.fmean(exp) if exp else None, "eps": st.fmean(eps) if eps else None,
         "eps_med": st.median(eps) if eps else None,
         "time": st.fmean([num(r, "wall_time") for r in sol]),
         "plan": st.fmean([num(r, "steps") for r in sol])}
    cpe = d["mp"] / d["exp"] if d["exp"] and d["mp"] is not None else None
    d["cpe"] = None if cpe is None or math.isclose(cpe, 1.0, abs_tol=0.02) else cpe
    return d


def f(v, nd=2):
    return "n/a" if v is None else ("%%.%df" % nd) % v


def reap_row(name, sub, seeds, has_expansions):
    per, missing = {}, []
    for s in seeds:
        files = (sorted(glob.glob("%s/%s/%s/per_problem.csv" % (ROOT, sub, s)))
                 or sorted(glob.glob("%s/%s/%s/%s__*.csv" % (ROOT, sub, s, s))))
        if files:
            per[s] = block(rows_of(files), has_expansions)
        else:
            missing.append(s)
    per = {s: b for s, b in per.items() if b.get("k")}
    coverage(name, missing, [b["n"] for b in per.values()])
    if not per:
        print("| %s | not evaluated | | | | |" % name)
        return per
    mean = lambda k: (st.fmean([b[k] for b in per.values() if b[k] is not None])
                      if any(b[k] is not None for b in per.values()) else None)
    sd = lambda k: st.pstdev([b[k] for b in per.values()]) if len(per) > 1 else 0.0
    print("| %s | %.1f ± %.1f | %s ± %s [%s] | %s | %s | %s |" % (
        name, mean("solve"), sd("solve"), f(mean("mp"), 1), f(sd("mp"), 1), f(mean("mp_med"), 1), f(mean("cpe")),
        eps(mean("eps"), mean("eps_med")), f(mean("time"), 1)))
    return per


def eps(m, med):
    """Exp./steps [med.] cell; n/a for a run without an expansion count."""
    return "n/a" if m is None else "%.2f [%.2f]" % (m, med)


def coverage(name, missing, ns):
    """Seeds that were not evaluated and runs with fewer than the 300 Table 1 problems, on stderr."""
    if missing:
        print("%s: not evaluated for %s" % (name, ", ".join(missing)), file=sys.stderr)
    if any(n != SUITE_N for n in ns):
        print("%s: problems per run %s, not the %d of Table 1" % (name, "/".join(map(str, ns)), SUITE_N), file=sys.stderr)


# The five runs of Table 1 as grouped in the appendix table: (group, sd shown, [(paper label, checkpoint)]).
RUNS = [("No time limit", False, [("Seed 2", "seed_4"), ("Seed 3", "seed_5")]),
        ("Time limit 60 s", True, [("Seed 1", "seed_1"), ("Seed 2", "seed_2"), ("Seed 3", "seed_3")])]


def seed_block(sub, s, has_expansions):
    files = (sorted(glob.glob("%s/%s/%s/per_problem.csv" % (ROOT, sub, s)))
             or sorted(glob.glob("%s/%s/%s/%s__*.csv" % (ROOT, sub, s, s))))
    b = block(rows_of(files), has_expansions) if files else None
    return b if b and b.get("k") else None


def runs_table(seeds):
    """The appendix table of the five runs of Table 1 (REAP-GBFS and REAP-greedy per seed, group means)."""
    print("\nThe five runs of Table 1 on the 300 test problems (appendix \"Open-Tabletop Ablations and Sample "
          "Efficiency\")")
    print("| Run | REAP-GBFS Solve (%) | Exp./steps | MP/plan | REAP-greedy Solve (%) | Plan len. |")
    print("|---|---|---|---|---|---|")
    for group, show_sd, members in RUNS:
        rows = [(lab, s, seed_block(T.REAP_GBFS, s, True), seed_block(T.REAP_GREEDY, s, False))
                for lab, s in members if s in seeds]
        rows = [r for r in rows if r[2] or r[3]]
        if not rows:
            continue
        print("| *%s* | | | | | |" % group)
        for lab, s, g, y in rows:
            print("| %s (%s) | %s | %s | %s | %s | %s |" % (
                lab, s, f(g and g["solve"], 1), f(g and g["eps"]),
                "%s [%g]" % (f(g["mp"], 1), g["mp_med"]) if g and g["mp"] is not None else "n/a",
                f(y and y["solve"], 1), f(y and y["plan"], 1)))
        if len(rows) < 2:
            continue
        col = lambda i, k: [r[i][k] for r in rows if r[i] and r[i][k] is not None]
        mean = lambda xs: st.fmean(xs) if xs else None
        pm = lambda xs: (f(mean(xs), 1) + (" ± %.1f" % st.pstdev(xs) if show_sd and len(xs) > 1 else "")
                         if xs else "n/a")
        print("| Mean | %s | %s | %s | %s | %s |" % (pm(col(2, "solve")), f(mean(col(2, "eps"))), f(mean(col(2, "mp")), 1),
                                                pm(col(3, "solve")), f(mean(col(3, "plan")), 1)))


def baseline_row(name, mode):
    files = sorted(glob.glob("%s/%s/%s__*.csv" % (ROOT, mode, mode)))
    if not files:
        print("| %s | not evaluated | | | | |" % name)
        return
    b = block(rows_of(files))
    coverage(name, [], [b["n"]])
    if not b.get("k"):
        print("| %s | 0.0 | | | | |" % name)
        return
    print("| %s | %.1f | %s [%s] | %s | %s | %s |" % (
        name, b["solve"], f(b["mp"], 1), f(b["mp_med"], 1), f(b["cpe"]), eps(b["eps"], b["eps_med"]), f(b["time"], 1)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    global CAP, ROOT
    ap.add_argument("--root", default=None,
                    help="outputs of the README commands, read from <root>/tabletop (default: $REAP_OUT, else outputs)")
    ap.add_argument("--seeds", default=",".join(SEEDS), help="REAP seeds (default: the five seeds of Table 1)")
    ap.add_argument("--cap", type=float, default=300.0, help="per-problem time cap in seconds (default 300; 0 = off)")
    args = ap.parse_args(argv)
    ROOT = T.roots(args.root)[0]
    seeds = [s for s in args.seeds.split(",") if s]
    CAP = args.cap or None
    print("Table 1, open tabletop (REAP rows: mean ± population sd over the seeds)")
    print("| Method | Solve (%) | MP / plan [med.] | Checks/exp. | Exp./steps [med.] | Time (s/solved) |")
    print("|---|---|---|---|---|---|")
    reap_row("REAP-GBFS", T.REAP_GBFS, seeds, True)
    reap_row("REAP-greedy", T.REAP_GREEDY, seeds, False)
    baseline_row("BrFS", T.BRFS)
    baseline_row("hFF", T.HFF)
    runs_table(seeds)


if __name__ == "__main__":
    main()
