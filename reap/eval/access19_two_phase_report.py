"""Access-19 two-phase pipeline (target-only extraction policy, then reload policy), 3 seeds: the solve rate on the 81
in-distribution problems whose layout neither policy saw in training (appendix "Environments", Access-19,
Evaluation), mean +- population sd over the seeds.

Layout: <root>/access19/composed/id_ood/seed_<k>/seed_<k>__shard*.csv, written by
scripts/eval.py --config configs/access19.yaml. A problem is unseen when its layout is in neither training set (rule
in seen()).

    python -m reap.eval.access19_two_phase_report [--root DIR]      (default root: $REAP_OUT, else outputs)
(the rule reads the extraction and reload training sets under $REAP_DATA, which the dataset asset holds; without them
every problem would count as unseen)
"""
import csv, glob, os, re, statistics as st, sys

import argparse

D = (os.environ.get("REAP_DATA") or "data") + "/access19/"
# "set" column -> directory
SETDIR = {"validation": "eval", "eval_ood": "eval_ood"}
NSH, NPROB = 10, 101          # shards and problems of a complete id_ood run (93 in distribution, 8 benchmark files)
OCC = re.compile(r"\(occupied\s+(\S+)\s+(\S+)\)")
_parsed = {}


def parse(f):
    if f not in _parsed:
        t = open(f).read()
        _parsed[f] = ({o: c for c, o in OCC.findall(t[t.index("(:init"):t.index("(:goal")])},
                      {o: c for c, o in OCC.findall(t[t.index("(:goal"):])})
    return _parsed[f]


def pfile(r):
    return D + SETDIR[r["set"]] + "/" + r["problem"]


def init_layout(f):
    init, _ = parse(f)
    return (init.get("ooi"), tuple(sorted(c for o, c in init.items() if o != "ooi")))


def goal_layout(f):
    _, goal = parse(f)
    return (goal.get("ooi"), tuple(sorted(c for o, c in goal.items() if o != "ooi")))


_TRAIN = {}


def train_layouts(ds, fn):
    if ds not in _TRAIN:
        _TRAIN[ds] = {fn(f) for f in glob.glob(D + ds + "/*.pddl") if not os.path.basename(f).startswith("domain")}
    return _TRAIN[ds]


def seen(r):
    """Seen rule: in extraction training if the initial layout is an initial layout of the extraction training set; in
    reload training if a _return problem's goal layout is a goal layout of the reload training set."""
    f = pfile(r)
    return init_layout(f) in train_layouts("train_extraction", init_layout) or \
        (r["task"] == "_return" and goal_layout(f) in train_layouts("train_reload", goal_layout))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("REAP_OUT") or "outputs")
    O = os.path.join(ap.parse_args(argv).root, "access19", "composed")
    pct = []
    for vdir in sorted(glob.glob("%s/id_ood/seed_*/" % O)):
        files = glob.glob(vdir + "*__shard*.csv")
        rows = [r for f in files for r in csv.DictReader(open(f))]
        if not rows:
            continue
        if len(files) < NSH or len(rows) < NPROB:
            print("incomplete: %s" % vdir, file=sys.stderr)
        rs = [r for r in rows if r["set"] == "validation" and not seen(r)]
        if rs:
            pct.append(100.0 * sum(r["solved"] == "1" for r in rs) / len(rs))

    print("Access-19 two-phase pipeline on the in-distribution problems that neither policy saw in training "
          "(appendix \"Environments\", three seeds)")
    print("| Setting | Solve (%) |\n|---|---|")
    print("| In-distribution, layout in neither extraction nor reload training | %s |" % (
        "%.1f ± %.1f" % (st.mean(pct), st.pstdev(pct)) if pct else "not evaluated"))


if __name__ == "__main__":
    main()
