"""Joint-training numbers at the final weights (Section "Beyond Open-Tabletop Rearrangement", joint training): the solve
rate on the 300 open-tabletop problems (mean +- population sd over the seeds), and per seed the solved problems of the
250 confined-shelf problems (the ID and the four non-monotone sets) and of the 90 multi-level shelf problems (the ID
and OOD sets).

Inputs: the per-problem CSVs of scripts/eval.py --config configs/joint.yaml, <root>/joint/seed_<k>/ (and
eval_multilevel_shelf/, the multi-level shelf ID set). Tabletop suite = the 15 x 20 Table 1 problems.

    python -m reap.eval.joint_report [--root DIR] [--seeds 1,2]      (default root: $REAP_OUT, else outputs)
"""
import argparse
import csv
import glob
import os
import statistics as S

TT = ["obj10", "obj12", "obj14", "obj16", "obj18", "obj20", "obj22", "obj24", "obj10_plan3", "obj12_plan3",
      "obj14_plan5", "obj16_plan5", "obj10_plan5_biased2", "obj12_plan7_biased3", "obj14_plan9_biased3"]
# (label, subdirectory, sets) of the three environments
SETS = [("Open tabletop", "", [f"{t}__u-tabletop__OOD" for t in TT]),
        ("Confined shelf", "", ["id__u-confined__ID"] + [f"eval_confined_nonmonotone_{k}__u-confined__OOD"
                                                         for k in (10, 12, 14, 16)]),
        ("Multi-level shelf", "", ["eval_ood__u-shelf__OOD"]),
        ("Multi-level shelf", "eval_multilevel_shelf", ["eval_multilevel_shelf__u-shelf__ID"])]


def load(root, run, sub, s):
    f = glob.glob(os.path.join(root, "joint", run, sub, f"{run}__{s}.csv"))
    assert len(f) == 1, (run, sub, s, f)
    return list(csv.DictReader(open(f[0])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("REAP_OUT") or "outputs")
    ap.add_argument("--seeds", default="1,2")
    a = ap.parse_args(argv)
    runs = ["seed_%s" % k for k in a.seeds.split(",") if k]
    solved = {}                              # (environment, run) -> [solved, problems]
    for run in runs:
        for lab, sub, ss in SETS:
            rows = sum((load(a.root, run, sub, s) for s in ss), [])
            c = solved.setdefault((lab, run), [0, 0])
            c[0] += sum(r["solved"] == "1" for r in rows); c[1] += len(rows)
    pct = [100.0 * solved[("Open tabletop", r)][0] / solved[("Open tabletop", r)][1] for r in runs]
    print("Joint training (final weights)")
    print("Open tabletop, solve (%%): %.1f ± %.1f" % (S.mean(pct), S.pstdev(pct)))
    print("| Environment | " + " | ".join(runs) + " |\n|---|" + "---|" * len(runs))
    for lab in ("Confined shelf", "Multi-level shelf"):
        print("| %s, solved | %s |" % (lab, " | ".join("%d of %d" % tuple(solved[(lab, r)]) for r in runs)))


if __name__ == "__main__":
    main()
