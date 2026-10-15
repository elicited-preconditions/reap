"""Table 2 rows of the confined shelf and the multi-level shelf, and the appendix table of the non-monotone
confined-shelf problems, in the columns of Table 2: Solve (%) · Mean exp./steps · MP / plan [median].

Per (environment, set, seed), over SOLVED problems: MP/plan = mean and median of the per-problem oracle count (unique
queries; the library checkers cache verdicts); exp/step = mean of expanded / plan steps. Over the three seeds (final
weights), as in Table 1: solve % mean +- POPULATION sd (ddof 0), exp/step and MP/plan the mean of the per-seed means,
[bracket] = the mean of the per-seed medians. The original multi-level shelf in-distribution set is given by its solve
rate only (appendix "Environments": the original set gives the same solve rate).

Sources (per-problem CSVs written by scripts/eval.py, final checkpoints):
  confined shelf      <root>/confined/seed_<k>/seed_<k>__<set>__confined-shelf__<ID|OOD>.csv
  multi-level shelf   <root>/multilevel_shelf/seed_<k>/... (original ID set, OOD) and
                      <root>/multilevel_shelf/seed_<k>/eval/... (the ID set of Table 2)
The per-seed REAP-GBFS numbers of Table 1: reap.eval.table1_report (the appendix table of the five runs).

    python -m reap.eval.table2_shelves [--root DIR]      (default root: $REAP_OUT, else outputs)
"""
import csv, glob, os, statistics as st, sys

import argparse

ROOT = os.environ.get("REAP_OUT") or "outputs"
SEEDS = ("seed_1", "seed_2", "seed_3")
# (set, Table 2 label or None, label in the appendix table or None)
CONF_SETS = [("validation", "Monotone (in-distribution)", None),
             ("eval_nonmonotone_10", None, "10"), ("eval_nonmonotone_12", None, "12"),
             ("eval_nonmonotone_14", None, "14"),
             ("eval_nonmonotone_16", "Non-monotone, 16 objects (OOD)", "16")]
# (subdirectory, set, label, solve rate only)
MULTILEVEL_SETS = [("eval", "validation", "In-distribution (1-5 blockers)", False),
                   ("", "eval_ood", "OOD (4-10 blockers)", False),
                   ("", "validation", "Original in-distribution set (appendix: the same solve rate)", True)]
HDR = "| %s | Solve (%%) | Mean exp./steps | MP / plan |\n|---|---|---|---|"


def rows_of(files, solved="solved", exp="expanded", steps="steps", mp="mp_calls"):
    out = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            out[(r.get("dataset") or os.path.basename(f), r.get("problem"))] = (int(r[solved]), int(float(r[exp])), int(float(r[steps])),
                                                            int(float(r[mp])) if r.get(mp) not in (None, "") else None)
    return list(out.values())


def stats(rows):
    s = [r for r in rows if r[0]]
    if not s:
        return len(s), len(rows), None, None, None, None
    mp = [r[3] for r in s if r[3] is not None]; es = [r[1] / max(1, r[2]) for r in s]
    if not mp:
        return len(s), len(rows), None, None, st.mean(es), st.median(es)
    return len(s), len(rows), st.mean(mp), st.median(mp), st.mean(es), st.median(es)


def agg(xs, solve_only=False):
    """The Table 2 cells over the evaluated seeds (seeds without CSVs are left out and named on stderr)."""
    xs = [x for x in xs if x and x[1]]
    if not xs:
        return "not evaluated | | "
    pct = [100.0 * k / n for k, n, *_ in xs]
    solve = "%.1f ± %.1f" % (st.mean(pct), st.pstdev(pct))
    if solve_only:
        return solve + " | | "
    mpm = [x[2] for x in xs if x[2] is not None]; mpd = [x[3] for x in xs if x[3] is not None]
    es = [x[4] for x in xs if x[4] is not None]
    return "%s | %s | %s" % (solve, "%.2f" % st.mean(es) if es else "n/a",
                             "%.1f [%.1f]" % (st.mean(mpm), st.mean(mpd)) if mpm else "n/a")


def seed_stats(pattern, what):
    xs = []
    for run in SEEDS:
        fs = glob.glob(pattern % run)
        if not fs:
            print("%s: %s not evaluated" % (what, run), file=sys.stderr)
        xs.append(stats(rows_of(fs)) if fs else None)
    return xs


def confined():
    xs = {s: seed_stats("%s/confined/%%s/*__%s__confined-shelf__*.csv" % (ROOT, s), "confined shelf " + s)
          for s, _, _ in CONF_SETS}
    print("Table 2, confined-shelf rearrangement (three seeds, final weights)")
    print(HDR % "Setting")
    for s, lab, _ in CONF_SETS:
        if lab:
            print("| %s | %s |" % (lab, agg(xs[s])))
    print("\nNon-monotone confined-shelf problems (appendix \"Environments\", three seeds, final weights)")
    print(HDR % "Cylinders")
    for s, _, lab in CONF_SETS:
        if lab:
            print("| %s | %s |" % (lab, agg(xs[s])))


def multilevel_shelf():
    print("\nTable 2, multi-level shelf retrieval (three seeds, final weights)")
    print(HDR % "Setting")
    for sub, s, lab, solve_only in MULTILEVEL_SETS:
        xs = seed_stats(os.path.join(ROOT, "multilevel_shelf", "%s", sub, "*__%s__shelf-access__*.csv" % s),
                        "multi-level shelf %s/%s" % (sub or ".", s))
        print("| %s | %s |" % (lab, agg(xs, solve_only)))


def main(argv=None):
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT, help="outputs of scripts/eval.py (default: $REAP_OUT, else outputs)")
    a = ap.parse_args(argv)
    ROOT = a.root
    confined()
    multilevel_shelf()


if __name__ == "__main__":
    main()
