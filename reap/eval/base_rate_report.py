"""Base rate of infeasible actions and rank-1 infeasibility (Section "Benchmark Results"), from the shards of
reap.eval.base_rate_rank1: python -m reap.eval.base_rate_report <dir>

Initial states of the 300 Table 1 problems: the share of symbolically applicable (state, action) pairs that are
infeasible, and how often the action ranked first by Q is infeasible. States reached by short random walks: how often
a uniformly random action and the action ranked first by Q are infeasible. The Q rank-1 rate is the mean over the five
Table 1 seeds (columns q_seed_1 ... q_seed_5 of the shards; other q_ columns are left out), +- the population sd over
these seeds, as in the paper. Also printed: the share of verdicts that differ between the two checks of every initial
state action (appendix "Environments"), and the feasible shares 100 - x of the random-walk states (abstract). The
report also writes the merged rows of the shards to <dir>/base_rate_rank1_rows.csv.
"""
import csv, glob, heapq, os, sys, statistics as st
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

T1_SEEDS = ["seed_%d" % k for k in range(1, 6)]   # the five Table 1 seeds


@dataclass(order=True)
class PI:                      # identical to search.PrioritizedItem
    priority: float
    item: Any = field(compare=False)

def first_pop(pr):
    h = []
    for i, p in enumerate(pr):
        heapq.heappush(h, PI(p, i))
    return heapq.heappop(h).item

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    D = argv[0]
    rows = []
    for f in sorted(glob.glob(D + "/rows_*.csv")):
        rows += list(csv.DictReader(open(f)))
    found = [c[2:] for c in rows[0] if c.startswith("q_")] if rows else []
    SEEDS = [s for s in T1_SEEDS if s in found]                 # the five Table 1 seeds only
    if not SEEDS:
        raise SystemExit("no Q column of the Table 1 seeds (q_%s ... q_%s) in %s" % (T1_SEEDS[0], T1_SEEDS[-1], D))
    if len(SEEDS) < len(T1_SEEDS):
        print("missing Table 1 seeds: %s" % ", ".join(s for s in T1_SEEDS if s not in found), file=sys.stderr)
    B = lambda x: x == "True"
    states = defaultdict(list)            # rows kept in the file's (sorted-action) order
    for r in rows:
        states[(r["problem"], r["state"])].append(r)
    probs = {p for p, _ in states}

    def summarise(keys):
        """Per state: the share of infeasible actions (pass 1) and whether each seed's Q rank-1 action is infeasible;
        pooled over (state, action) pairs: the infeasible share and the verdicts that differ between the two passes."""
        per = []                          # per state dict
        pooled = {"n": 0, "p1": 0, "dis": 0}
        for k in keys:
            rs = states[k]
            n = len(rs); inf1 = [B(r["infeasible_p1"]) for r in rs]; inf2 = [B(r["infeasible_p2"]) for r in rs]
            pooled["n"] += n; pooled["p1"] += sum(inf1)
            pooled["dis"] += sum(a != b for a, b in zip(inf1, inf2))
            d = {"n": n, "uni": sum(inf1) / n}
            for s in SEEDS:
                q = [float(r["q_" + s]) for r in rs]
                j = first_pop([-x for x in q])
                d["q_" + s] = inf1[j]
            per.append(d)
        m = lambda key: 100 * st.mean(float(d[key]) for d in per)
        return pooled, m("uni"), [m("q_" + s) for s in SEEDS]

    prim = sorted(k for k in states if k[1] == "s0")
    sec = sorted(k for k in states if k[1] != "s0")
    pooled, _, qs = summarise(prim)
    print("Initial states of the %d problems of Table 1" % len(probs))
    print("  symbolically applicable actions that are geometrically infeasible: %.0f%%" % (100 * pooled["p1"] / pooled["n"]))
    print("  a repeated check changes the verdict in %.1f%% of the cases" % (100 * pooled["dis"] / pooled["n"]))
    print("  action ranked first by Q infeasible: %.1f ± %.1f%% (mean ± population sd over the Table 1 seeds)" % (
        st.mean(qs), st.pstdev(qs)))
    _, uni, qs = summarise(sec)
    print("{:,} states reached by short random walks".format(len(sec)))
    print("  action ranked first by Q infeasible: %.1f ± %.1f%% (feasible: %.0f%%)" % (
        st.mean(qs), st.pstdev(qs), 100 - st.mean(qs)))
    print("  random choice infeasible: %.0f%% (feasible: %.0f%%)" % (uni, 100 - uni))
    # combined CSV
    cols = list(rows[0].keys())
    with open(D + "/base_rate_rank1_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print("wrote", D + "/base_rate_rank1_rows.csv")


if __name__ == "__main__":
    main()
