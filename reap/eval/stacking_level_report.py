"""Per-set, per-level report of a grip-aware 3D-stacking evaluation (reap.eval.stacking).

For each evaluation directory (<out> of reap.eval.stacking):
  protocol  <dir>/protocol.json                  (GBFS 300 s / 16384, MLB_CHECKER=honest_grip)
  verify    <dir>/verify/per_plan/<set>/*.json   (per-action walk + continuous verification of the plans of
                                                  reap.eval.stacking_plans --plan_only)
Per set and level: solved/n; verified = plans that pass the continuous verification (with the smallest arm clearance
of the verified plans and the release heights of the upright puts). The search statistics of Table 2 are printed by
reap.eval.stacking_report.

    python -m reap.eval.stacking_level_report <dir> [<dir> ...] [--out FILE]
"""
import argparse, collections, glob, json, os, re, statistics as S

SETNAME = {"validation": "ID", "eval_ood": "OOD"}
VLABEL = {"validation": "eval", "eval_ood": "eval_ood"}
# the 6 evaluation problems not counted in the main ID row (key overlap)
_RULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stacking_reporting_rules.json")
OUTSIDE = json.load(open(_RULES)) if os.path.exists(_RULES) else None


def level(set_name, name):
    if set_name == "eval_ood":
        return "L5"
    m = re.match(r"(L\d)_", name)
    return m.group(1) if m else "?"


def verify_of(d, set_name):
    out = {}
    for f in glob.glob("%s/verify/per_plan/%s/*.json" % (d, VLABEL[set_name])):
        r = json.load(open(f))
        n = os.path.basename(f)[:-5]
        if r.get("no_plan"):
            out[n] = {"plan": False}; continue
        out[n] = {"plan": True, "pass": bool(r.get("pass")), "arm": (r.get("arm") or {}).get("min_mm"),
                  "rel": [x[2] for x in r.get("release_mm", []) if x[1] in ("put-upright", "put-long-upright")],
                  "err": r.get("error") or (r.get("not_reproducible") or {}).get("error")}
    return out


def fmt(rows):
    if not rows:
        return "-"
    return "%d/%d" % (sum(1 for p in rows if p["solved"]), len(rows))


def report(evdir):
    label = os.path.relpath(evdir)
    f = os.path.join(evdir, "protocol.json")
    if not os.path.exists(f):
        return "%s: protocol result missing (%s)\n" % (label, f)
    d = json.load(open(f))
    lines = ["== %s  checkpoint=%s  checker=%s  data=%s" % (
        label, d.get("checkpoint"), (d.get("checkers") or {}).get("validation", {}).get("label", "?"), d.get("eval_data"))]
    for s in d["results"]:
        ver = verify_of(evdir, s["set"])
        by = collections.defaultdict(list)
        for p in s["problems"]:
            by[level(s["set"], p["name"][:-5])].append(p)
        vall = [v for v in ver.values() if v["plan"]]
        lines.append("  %-10s %-3s all   %-8s verified %d/%d plans" % (s["set"], SETNAME.get(s["set"], ""), fmt(s["problems"]),
                     sum(v["pass"] for v in vall), len(vall)))
        if s["set"] == "validation" and OUTSIDE:   # main ID row: the ID set without those 6 problems
            inn = [p for p in s["problems"] if p["name"][:-5] not in OUTSIDE["overlap"]]
            vin = [v for n, v in ver.items() if v["plan"] and n not in OUTSIDE["overlap"]]
            lines.append("  %-10s %-3s main row   %-8s verified %d/%d plans" % ("", "ID", fmt(inn),
                         sum(v["pass"] for v in vin), len(vin)))
        for lv in sorted(by):
            names = [p["name"][:-5] for p in by[lv]]
            v = [ver[n] for n in names if n in ver and ver[n]["plan"]]
            arms = [x["arm"] for x in v if x["pass"] and x["arm"] is not None]
            rel = collections.Counter(r for x in v for r in x["rel"])
            vstr = "verified %d/%d plans%s%s" % (sum(x["pass"] for x in v), len(v),
                                                  ", arm min %.2f mm" % min(arms) if arms else "",
                                                  ", upright releases %s" % dict(sorted(rel.items())) if rel else "")
            lines.append("  %-10s     %-4s  %-8s %s" % ("", lv, fmt(by[lv]), vstr))
        bad = [(n, x["err"]) for n, x in ver.items() if x["plan"] and not x["pass"]]
        if bad:
            lines.append("    not verified: %s" % bad[:8])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="evaluation directories of reap.eval.stacking")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    txt = "".join(report(d) for d in a.dirs)
    print(txt, end="")
    if a.out:
        open(a.out, "w").write(txt)
