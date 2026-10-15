"""The tables of the appendix "PDDLStream Baseline", from the outputs of the README commands under --root (default
$REAP_OUT, else outputs; layout in reap.eval.table1_baselines):

- solve rate (%) by reference plan length and per evaluation set, from the per-problem CSVs
  baselines/pddlstream/<configuration>/NNN.csv of the four PDDLStream configurations (baselines/pddlstream/bench/run.py
  writes them);
- motion-planner queries per plan on the problems solved by both PDDLStream and REAP, which also reads
  tabletop/reap_gbfs/seed_<k>/per_problem.csv and tabletop/reap_greedy/seed_<k>/per_problem.csv (k = 1 to 5). A
  problem counts as solved by REAP when all five seeds solve it; its REAP value is the mean over the five seeds.
  Ref. |pi| is the mean reference plan length of the problems the configuration solves.

Reference plan length = number of actions in each problem's .plan file of the released tabletop suite ($REAP_DATA).
Without the problem sets the report stops with an error. A table cell without problems is n/a.

    python -m reap.eval.pddlstream_appendix [--root DIR]
"""
import csv, glob, statistics as st
from pathlib import Path

from reap.eval import table1_baselines as T

GROUPS = [("Object count (random placement)", ["obj10", "obj12", "obj14", "obj16", "obj18", "obj20", "obj22", "obj24"]),
          ("Plan length (random placement)", ["obj10_plan3", "obj12_plan3", "obj14_plan5", "obj16_plan5"]),
          ("Arrangement (K direct blockers)", ["obj10_plan5_biased2", "obj12_plan7_biased3", "obj14_plan9_biased3"])]
BUCKETS = ["1 (no relocation)", "3 (one relocation)", "5-7 (two or three)", "9-15 (four or more)"]
# configuration: (label of the solve-rate tables, label of the common-solves table), as in the paper
CONFIGS = {"ffastar2": ("test, ff-astar2", "test-stream, ff-astar2"),
           "ffeager": ("test, ff-eager", "test-stream, ff-eager"),
           "gen_ffeager": ("cell gen.", "cell-generator, ff-eager"),
           "informed_cells_occlusion": ("occl. axiom", "informed cells + occlusion axiom")}


def key(dataset, problem):
    return dataset, Path(problem).name.replace(".pddl", "")


def bucket(rl):
    return BUCKETS[0] if rl <= 1 else BUCKETS[1] if rl <= 4 else BUCKETS[2] if rl <= 8 else BUCKETS[3]


def load_pddlstream(d):
    """(dataset, problem) -> (solved, MP/plan, reference plan length)."""
    out = {}
    for f in sorted(glob.glob(f"{d}/[0-9][0-9][0-9].csv")):
        for r in csv.DictReader(open(f)):
            out[key(r["dataset"], r["problem"])] = (r["solved"] == "1", int(r["mp_calls"]),
                                                    T.reference_plan_length(r["dataset"], r["problem"]))
    return out


def load_reap(dirs):
    """(dataset, problem) -> mean MP/plan over the seeds, for the problems every seed solves."""
    per = {}
    for d in dirs:
        for r in csv.DictReader(open(d / "per_problem.csv")):
            per.setdefault(key(r["dataset"], r["problem"]), []).append((r["solved"] == "1", int(r["mp_calls"])))
    return {k: st.fmean(m for _, m in v) for k, v in per.items() if len(v) == len(dirs) and all(s for s, _ in v)}


def pct(rows):
    return "%d" % round(100 * sum(rows) / len(rows)) if rows else "n/a"


def mean1(xs):
    xs = list(xs)
    return "%.1f" % st.fmean(xs) if xs else "n/a"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="outputs of the README commands: <root>/tabletop, <root>/baselines (default: $REAP_OUT, "
                         "else outputs)")
    tt, bl = T.roots(ap.parse_args(argv).root)
    pds = {c: load_pddlstream(bl / T.PDDLSTREAM[c]) for c in CONFIGS}
    reap = {lab: load_reap([tt / m / s for s in T.SEEDS])
            for lab, m in (("REAP-GBFS", T.REAP_GBFS), ("REAP-greedy", T.REAP_GREEDY))}

    print("Solve rate (%) of PDDLStream by reference plan length")
    cols = list(CONFIGS)
    print("Reference plan length".ljust(24) + "Problems".rjust(9) + "".join(CONFIGS[c][0].rjust(17) for c in cols))
    for b in BUCKETS:
        sv = {c: [s for s, _, rl in pds[c].values() if bucket(rl) == b] for c in cols}
        print(b.ljust(24) + str(len(sv[cols[0]])).rjust(9) + "".join(pct(sv[c]).rjust(17) for c in cols))

    print()
    print("Solve rate (%) of PDDLStream per evaluation set (20 problems each)")
    cols = ["ffastar2", "gen_ffeager", "informed_cells_occlusion"]
    print("Evaluation set".ljust(24) + "".join(CONFIGS[c][0].rjust(17) for c in cols))
    gap = 0
    for title, sets in GROUPS:
        print(title)
        for d in sets:
            sv = {c: [s for (ds, _), (s, _, _) in pds[c].items() if ds == d] for c in CONFIGS}
            gap = max(gap, abs(sum(sv["ffeager"]) - sum(sv["ffastar2"])))
            print(("  " + d).ljust(24) + "".join(pct(sv[c]).rjust(17) for c in cols))
    print("The ff-eager test-stream configuration is within %d problems of ff-astar2 in every set." % gap)

    print()
    print("Motion-planner queries per plan on the problems solved by both PDDLStream and REAP")
    print((" " * 44 + "".join(("vs. " + lab).center(33) for lab in reap)).rstrip())
    print("PDDLStream configuration".ljust(34) + "Ref. |pi|".rjust(10)
          + ("|both|".rjust(9) + "PDDLStream".rjust(12) + "REAP".rjust(12)) * len(reap))
    for c, (_, lab) in CONFIGS.items():
        own = [rl for s, _, rl in pds[c].values() if s]
        line = lab.ljust(34) + mean1(own).rjust(10)
        for r in reap.values():
            both = [k for k, (s, _, _) in pds[c].items() if s and k in r]
            line += (str(len(both)).rjust(9) + mean1(pds[c][k][1] for k in both).rjust(12)
                     + mean1(r[k] for k in both).rjust(12))
        print(line)


if __name__ == "__main__":
    main()
