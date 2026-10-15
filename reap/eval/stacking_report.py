"""3D stacking rows of Table 2 and the 3D-stacking solve rates of the appendix "Environments": the best-validation
checkpoints (Table 2: in-distribution, 89 problems, and out-of-distribution; appendix: all 95 in-distribution problems)
and the final checkpoints (appendix: in-distribution, 89 problems, and out-of-distribution), and the solved problems
per level L0-L5 with the goal block counts (appendix: the seed that never learns the hardest level).
Over the three seeds: solve % mean +- population sd; mean exp./steps = mean of the per-seed means of expanded / plan
steps over solved problems; MP / plan = mean of the per-seed means [mean of the per-seed medians] of the unique
queries over solved problems.

Inputs: the evaluation directories of reap.eval.stacking, <root>/seed_<k>/{best_val,final}/ (protocol.json, plans/),
and the dataset data/stacking ($REAP_DATA).

    python -m reap.eval.stacking_report [<root>]      (default root: $REAP_OUT/stacking)
"""
import csv, json, os, re, statistics as S, collections
import sys
ROOT = os.path.join(os.environ.get("REAP_OUT") or "outputs", "stacking")
_D = os.environ.get("REAP_DATA") or "data"
T = "%s/seed_%d/%s/protocol.json"
DS = _D + "/stacking"
RULES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stacking_reporting_rules.json")))
OVERLAP = set(RULES["overlap"])   # the 6 evaluation problems not counted in the main ID row
OVERLAP = {o if o.endswith(".pddl") else o + ".pddl" for o in OVERLAP}
level = lambda name: name.split("_")[0] if name.startswith("L") else "L5"


def load(seed, kind):
    d = json.load(open(T % (ROOT, seed, kind)))
    out = {}
    for r in d["results"]:
        for p in r["problems"]:
            p = dict(p); p["set"] = r["set"]; p["lvl"] = level(p["name"]) if r["set"] != "eval_ood" else "L5"
            out[(r["set"], p["name"])] = p
    return d["checkpoint"], out


def stats(ps):
    sol = [p for p in ps if p["solved"]]
    if not ps:
        return None
    eps = [p["expanded"] / max(1, p["plan_len"]) for p in sol]
    mp = [p["mp_calls"] for p in sol]
    return {"n": len(ps), "solved": len(sol), "pct": 100.0 * len(sol) / len(ps), "eps": S.mean(eps) if eps else float("nan"),
            "mp": mp}


def fmt_agg(ss, solve_only=False):
    pct = [s["pct"] for s in ss]
    solve = "%.1f ± %.1f" % (S.mean(pct), S.pstdev(pct))
    if solve_only:
        return solve
    eps = [s["eps"] for s in ss if s["mp"]]
    seedmp = [S.mean(s["mp"]) for s in ss if s["mp"]]
    seedmed = [S.median(s["mp"]) for s in ss if s["mp"]]
    return "%s | %.2f | %.1f [%.1f]" % (solve, S.mean(eps), S.mean(seedmp), S.mean(seedmed))


def main(argv=None):
    global ROOT
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv:
        ROOT = argv[0]
    ID89 = lambda k, p: k[0] == "validation" and k[1] not in OVERLAP
    OOD = lambda k, p: k[0] == "eval_ood"
    ID95 = lambda k, p: k[0] == "validation"
    data = {}
    for kind in ("best_val", "final"):
        for s in (1, 2, 3):
            data[(kind, s)] = load(s, kind)
    row = lambda kind, f: [stats([p for k, p in data[(kind, s)][1].items() if f(k, p)]) for s in (1, 2, 3)]
    print("Table 2, three-dimensional block stacking (three seeds, selected checkpoints)")
    print("| Setting | Solve (%) | Mean exp./steps | MP / plan |\n|---|---|---|---|")
    print("| In-distribution (1-8 blocks), 89 problems | %s |" % fmt_agg(row("best_val", ID89)))
    print("| OOD (combined structures, 4-12 blocks) | %s |" % fmt_agg(row("best_val", OOD)))
    print("\nAppendix \"Environments\", three-dimensional block stacking (three seeds)")
    print("| Setting | Solve (%) |\n|---|---|")
    print("| Selected checkpoints, all 95 in-distribution problems | %s |" % fmt_agg(row("best_val", ID95), True))
    print("| Final weights, in-distribution (89 problems) | %s |" % fmt_agg(row("final", ID89), True))
    print("| Final weights, out-of-distribution | %s |" % fmt_agg(row("final", OOD), True))

    # per level
    print("\nSolved problems per level, in-distribution (89 problems) and out-of-distribution (L5) (selected "
          "checkpoints; final weights in brackets)")
    print("| Level | Goal blocks | seed_1 | seed_2 | seed_3 |\n|---|---|---|---|---|")


    def goal_blocks(path):
        t = open(path).read().lower(); return set(b for b, c in re.findall(r"\(in (\S+) (\S+)\)", t[t.index("(:goal"):]))


    blk = collections.defaultdict(list)
    for fn in os.listdir(DS + "/eval"):
        if fn.endswith(".pddl"):
            blk[level(fn)].append(len(goal_blocks(os.path.join(DS, "eval", fn))))
    for fn in os.listdir(DS + "/eval_ood"):
        if fn.endswith(".pddl"):
            blk["L5"].append(len(goal_blocks(os.path.join(DS, "eval_ood", fn))))
    for L in ("L0", "L1", "L2", "L3", "L4", "L5"):
        cells = []
        for s in (1, 2, 3):
            b = stats([p for k, p in data[("best_val", s)][1].items() if p["lvl"] == L and k[1] not in OVERLAP])
            f_ = stats([p for k, p in data[("final", s)][1].items() if p["lvl"] == L and k[1] not in OVERLAP])
            cells.append("%d/%d [%d/%d]" % (b["solved"], b["n"], f_["solved"], f_["n"]))
        print("| %s | %d-%d | %s | %s | %s |" % (L, min(blk[L]), max(blk[L]), *cells))


if __name__ == "__main__":
    main()
