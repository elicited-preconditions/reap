#!/usr/bin/env python3
"""The Fig. 3 panel CSVs (six methods) from the per-problem outputs of Table 1.

Buckets:
  (a) object count -- the 8 plain cells, 20 problems each
  (b) reference plan length -- all 300 problems, banded 1,3,...,15
  (c) arrangement -- random (the 12 non-biased cells, 240), K=2 (20), K=3 (40)

REAP rows carry every seed of the reported set (the five seeds of Table 1) so the plotting code can compute a
per-bucket std; single-run baselines are emitted with seed=1. A method is only given a row
for a bucket it was actually evaluated on -- "attempted and failed" and "never
evaluated" must not collapse into the same 0.

Per-problem time cap: a solve whose total wall time is over `--cap` seconds
(default 300, the per-problem cap of Table 1; 0 = off) counts as unsolved, for
every method. Only PLOI+hFF has such solves; every other method stops at
the cap.

Inputs: the outputs of the README commands under --root (default $REAP_OUT, else outputs): tabletop/ (REAP-GBFS,
REAP-greedy, hFF) and baselines/ (PLOI+hFF, SAHS, PDDLStream informed cells), as listed in reap.eval.table1_baselines;
the reference plan lengths of panel (b) from the .plan files of the tabletop evaluation sets ($REAP_DATA). Output:
--out (default $REAP_OUT/figure3_csv).

    python -m reap.eval.figure3 [--root DIR] [--cap 300] [--out DIR]
"""
import argparse, csv, glob, json, sys
from pathlib import Path

import os as _os


from reap.eval import table1_baselines as T

TT = BL = CAP = OUT = None   # set by main()
def cfg(s): return Path(str(s)).name.replace(".pddl", "")
def wall(v): return float(v) if v not in (None, "") else None

PLAIN = [f"obj{n}" for n in (10, 12, 14, 16, 18, 20, 22, 24)]
K2, K3 = {"obj10_plan5_biased2"}, {"obj12_plan7_biased3", "obj14_plan9_biased3"}
PLOI_CELLS = PLAIN + ["obj10_plan3", "obj12_plan3", "obj14_plan5", "obj16_plan5",
                      "obj10_plan5_biased2", "obj12_plan7_biased3", "obj14_plan9_biased3"]
REAP_SEEDS = T.SEEDS

def main(argv=None):
    global TT, BL, CAP, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="outputs of the README commands: <root>/tabletop, <root>/baselines (default: $REAP_OUT, "
                         "else outputs)")
    ap.add_argument("--cap", type=float, default=300.0, help="per-problem time cap in seconds (default 300; 0 = off)")
    ap.add_argument("--out", default=None, help="output directory (default: $REAP_OUT/figure3_csv)")
    a = ap.parse_args(argv)
    CAP = a.cap or None
    TT, BL = T.roots(a.root)
    OUT = Path(a.out) if a.out else Path(_os.environ.get("REAP_OUT") or "outputs") / "figure3_csv"

    def from_reap(root):
        out = []
        for i, s in enumerate(REAP_SEEDS, start=1):
            for r in csv.DictReader(open(glob.glob(f"{TT}/{root}/{s}/per_problem.csv")[0])):
                out.append((i, (r["dataset"], cfg(r["problem"])), r["solved"] == "1", wall(r["wall_time"])))
        return out

    def from_search(sub):
        return [(1, (r["dataset"], cfg(r["problem"])), r["solved"] == "1", wall(r.get("wall_time")))
                for f in glob.glob(f"{TT}/{sub}/*__*.csv")
                for r in csv.DictReader(open(f))]

    def from_sahs():
        return [(1, (r["cell"], cfg(r["problem"])), r["solved"], r["wall_s"])
                for p in sorted((BL / T.SAHS).glob("shard_*.json"))
                for r in json.loads(p.read_text())]

    def from_ploi(sub):
        out = []
        for p in sorted((BL / sub).glob("shard_*.jsonl"),
                        key=lambda q: int(q.stem.split("_")[1])):
            n = int(p.stem.split("_")[1])
            for k, line in enumerate(x for x in p.read_text().splitlines() if x.strip()):
                r = json.loads(line)
                out.append((1, (PLOI_CELLS[(n * 10 + k) // 20], r["problem"]), r["solved"], r["wall_time_s"]))
        return out

    def from_pds(arm):
        return [(1, (r["dataset"], cfg(r["problem"])), r["solved"] == "1", wall(r.get("wall_time")))
                for f in glob.glob(f"{BL}/{arm}/*.csv")
                for r in csv.DictReader(open(f))]

    def capped(name, recs):
        out = []
        for s, k, ok, w in recs:
            if CAP and ok and w is not None and w > CAP:
                ok = False
            out.append((s, k, ok))
        return out

    METHODS = [(name, capped(name, recs)) for name, recs in [
        ("REAP-GBFS",   from_reap(T.REAP_GBFS)),
        ("REAP-greedy", from_reap(T.REAP_GREEDY)),
        ("hFF",         from_search(T.HFF)),
        ("PLOI+hFF",    from_ploi(T.PLOI_HFF)),
        ("SAHS",        from_sahs()),
        ("PDDLStream (informed cells)", from_pds(T.PDDLSTREAM["informed_cells_occlusion"])),
    ]]

    panels = {"panela_objcount_seeds.csv": [], "panelb_planlen_seeds.csv": [],
              "panelc_topology_seeds.csv": []}
    missing = []
    for name, recs in METHODS:
        seeds = sorted({s for s, _, _ in recs})
        for si in seeds:
            rows = [(k, ok) for s, k, ok in recs if s == si]
            seen = {k for k, _ in rows}
            for b in PLAIN:                                   # (a)
                g = [ok for k, ok in rows if k[0] == b]
                if not g: missing.append((name, "a", b)); continue
                panels["panela_objcount_seeds.csv"].append((b.replace("obj",""), name, si, sum(g), len(g)))
            by = {}                                            # (b)
            for k, ok in rows:
                L = T.reference_plan_length(*k)
                by.setdefault(L, [0, 0]); by[L][1] += 1; by[L][0] += int(ok)
            for L in sorted(by):
                panels["panelb_planlen_seeds.csv"].append((L, name, si, by[L][0], by[L][1]))
            for bn, sel in (("random_indist", lambda d: d not in K2 and d not in K3),
                            ("K=2", lambda d: d in K2), ("K=3", lambda d: d in K3)):   # (c)
                g = [ok for k, ok in rows if sel(k[0])]
                if not g: missing.append((name, "c", bn)); continue
                panels["panelc_topology_seeds.csv"].append((bn, name, si, sum(g), len(g)))

    OUT.mkdir(parents=True, exist_ok=True)
    for fn, rows in panels.items():
        with (OUT / fn).open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["bucket", "method", "seed", "solved", "total"]); w.writerows(rows)
        print(f"wrote {OUT / fn}")
    if missing:
        print("buckets with no data (row omitted, not zero-filled):", file=sys.stderr)
        for m in missing: print(f"  {m[0]} panel {m[1]} bucket {m[2]}", file=sys.stderr)


if __name__ == "__main__":
    main()
