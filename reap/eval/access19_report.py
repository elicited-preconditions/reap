"""Access-19 rows of Table 2 and the Access-19 numbers of Section "Beyond Open-Tabletop Rearrangement": the single
goal-conditioned policy in distribution (Table 2), on the 8 benchmark files and on the 72 new 18-blocker layouts (text),
and the two-phase pipeline (target-only extraction policy, then reload policy) on the 72 new 18-blocker layouts
(Table 2).
Columns of Table 2, over the seeds: solve % mean +- population sd; mean exp./steps = mean of the per-seed means of
expanded/steps over solved problems; MP / plan = mean of the per-seed means [mean of the per-seed medians] of
mp_calls_dedup (the distinct (state, action) queries) over solved problems.

Inputs: the shard CSVs of scripts/eval.py --config configs/access19.yaml, <root>/access19/{single,composed}/<set>/seed_<k>/.

    python -m reap.eval.access19_report [--root DIR] [--seeds 1,2,3]      (default root: $REAP_OUT, else outputs)
"""
import argparse, csv, glob, os, statistics as st, sys


def rows(d):
    fs = sorted(glob.glob(d + "/*__shard*.csv"))
    return [r for f in fs for r in csv.DictReader(open(f))], len(fs)


def seedstat(rs):
    s = [r for r in rs if r["solved"] == "1"]
    eps = [int(r["expanded"]) / int(r["steps"]) for r in s if r["steps"] and int(r["steps"]) > 0]
    return {"pct": 100.0 * len(s) / len(rs), "eps": st.mean(eps) if eps else None,
            "mp": [int(r["mp_calls_dedup"]) for r in s]}


def cells(per, solve_only=False):
    """per: seed -> rows. The Table 2 cells over the seeds with rows."""
    S = [seedstat(v) for v in per.values() if v]
    if not S:
        return "not evaluated" + ("" if solve_only else " | | ")
    pct = [s["pct"] for s in S]
    solve = "%.1f ± %.1f" % (st.mean(pct), st.pstdev(pct))
    if solve_only:
        return solve
    eps = [s["eps"] for s in S if s["eps"] is not None]
    mp = [s["mp"] for s in S if s["mp"]]
    return "%s | %s | %s" % (solve, "%.2f" % st.mean(eps) if eps else "n/a",
                             "%.1f [%.1f]" % (st.mean(st.mean(x) for x in mp), st.mean(st.median(x) for x in mp))
                             if mp else "n/a")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("REAP_OUT") or "outputs")
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args(argv)
    A = os.path.join(a.root, "access19")
    seeds = ["seed_%s" % k for k in a.seeds.split(",") if k]

    def runs(kind, setname):
        return {k: rows(os.path.join(A, kind, setname, k)) for k in seeds}

    t3 = runs("single", "id_ood")
    o18a, o18b = runs("single", "eval_18blocker_a"), runs("single", "eval_18blocker_b")
    t18a, t18b = runs("composed", "eval_18blocker_a"), runs("composed", "eval_18blocker_b")
    inc = [k for k in t3 if t3[k][1] < 10 or len(t3[k][0]) < 101]     # 93 in distribution + 8 benchmark files
    inc += ["%s/eval_18blocker_%s" % (k, h) for h, o in (("a", o18a), ("b", o18b)) for k in o
            if o[k][1] < 10 or len(o[k][0]) < 36]
    if inc:
        print("incomplete single-policy outputs: %s" % ", ".join(inc), file=sys.stderr)

    single_id = {k: [r for r in v[0] if r["set"] == "validation"] for k, v in t3.items()}
    single_bench = {k: [r for r in v[0] if r["tier"] == "18"] for k, v in t3.items()}
    single_72 = {k: o18a[k][0] + o18b[k][0] for k in seeds if o18b[k][1] >= 10 and len(o18b[k][0]) >= 36}
    two_72 = {k: t18a[k][0] + t18b[k][0] for k in seeds if t18a[k][0] and t18b[k][0]}

    print("Table 2, Access-19 (three seeds)")
    print("| Setting | Solve (%) | Mean exp./steps | MP / plan |\n|---|---|---|---|")
    print("| Single policy, in-distribution (<= 14 blockers) | %s |" % cells(single_id))
    print("| Two-phase, 18-blocker (the 72 new layouts) | %s |" % cells(two_72))
    print("\nSingle goal-conditioned policy on the 18-blocker problems (Section \"Beyond Open-Tabletop Rearrangement\")")
    print("| Setting | Solve (%) |\n|---|---|")
    print("| 18-blocker benchmark instances | %s |" % cells(single_bench, True))
    print("| 72 unseen 18-blocker layouts | %s |" % cells(single_72, True))


if __name__ == "__main__":
    main()
