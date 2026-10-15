#!/usr/bin/env python
"""Tables and figure data of the paper from evaluation outputs.

    python scripts/report.py <report> [arguments of the report ...]
    python scripts/report.py <report> --help

When the outputs a report reads are missing, it names them and the commands that create them, and exits with code 1.

Reports (module of reap.eval in brackets):
"""
from __future__ import annotations

import glob
import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REPORTS = {
    "table1": ("reap.eval.table1_report", "Table 1, REAP-GBFS / REAP-greedy and the hFF / BrFS rows; the five runs"),
    "table1_baselines": ("reap.eval.table1_baselines", "Table 1, every baseline row (SAHS, PLOI, BT-h, PDDLStream, ...)"),
    "figure3": ("reap.eval.figure3", "Fig. 3 panel CSVs (object count, plan length, topology)"),
    "table2": ("reap.eval.table2_shelves", "Table 2 rows of the confined and multi-level shelves"),
    "access19": ("reap.eval.access19_report", "Access-19 rows (single policy and two-phase pipeline)"),
    "access19_composed": ("reap.eval.access19_two_phase_report", "Access-19 two-phase pipeline, in distribution"),
    "stacking": ("reap.eval.stacking_report", "3D stacking rows per seed and level"),
    "joint": ("reap.eval.joint_report", "joint training rows"),
    "base_rate": ("reap.eval.base_rate_report", "base rate of infeasible actions and rank-1 infeasibility"),
    "noise": ("reap.eval.noise.report", "execution under perturbed observations"),
    "pddlstream_appendix": ("reap.eval.pddlstream_appendix", "PDDLStream appendix tables (solve rates, common solves)"),
}
__doc__ += "".join(f"  {k:20s} {d} [{m.rsplit('.', 1)[1]}]\n" for k, (m, d) in REPORTS.items())

# ------------------------------------------------------------------------------------- outputs each report reads
TT = "python scripts/eval.py --config configs/tabletop.yaml"
T1_CMD = {
    "reap_gbfs": f"{TT} --seed <k>   (k = 1 to 5; writes outputs/tabletop/reap_gbfs/seed_<k>)",
    "reap_greedy": f"{TT} --seed <k> --protocol greedy   (k = 1 to 5)",
    "hff": f"{TT} --protocol hff",
    "brfs": f"{TT} --protocol brfs",
    "sahs": "python baselines/sahs/run.py eval   (writes outputs/baselines/sahs)",
    "ploi_hff": "python baselines/ploi/run.py eval --mode ploi_hff",
    "ploi_brfs": "python baselines/ploi/run.py eval --mode ploi_bfs",
    "bt_h": "python baselines/bt_h/run.py eval",
    "pddlstream": "python baselines/pddlstream/bench/run.py <CONFIG> outputs/baselines/pddlstream/<directory>"
                  "   (baselines/pddlstream/README.md)",
}
PDDLSTREAM_CONFIG = {"ffastar2": "test_ffastar2", "ffeager": "test_ffeager", "gen_ffeager": "gen_ffeager",
                     "informed_cells_occlusion": "informed_cells_occlusion"}
T1_REPORTS = ("table1", "table1_baselines", "figure3", "pddlstream_appendix")


def _opt(argv, name, default=None):
    """The value of option ``name`` in ``argv`` (``--name X`` or ``--name=X``)."""
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def _positional(argv, with_value=()):
    """The first positional argument of ``argv`` (options in ``with_value`` take a value)."""
    i = 0
    while i < len(argv):
        if argv[i] in with_value:
            i += 2
        elif argv[i].startswith("-"):
            i += 1
        else:
            return argv[i]
    return None


def _module(name):
    """reap/eval/<name>.py, which imports nothing from reap, loaded without importing the package (and torch)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("report_" + name, REPO / "reap" / "eval" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m   # dataclasses look their module up here
    spec.loader.exec_module(m)
    return m


def needs(report, argv):
    """(mode, [(path or glob pattern, command that creates it)], argv for the report). mode "all": every pattern must
    match a file; "any": at least one must (the report prints the rows it finds)."""
    out = Path(os.environ.get("REAP_OUT") or "outputs")
    if report in T1_REPORTS:
        T = _module("table1_baselines")
        tt, bl = T.roots(_opt(argv, "--root"))
        reap = [(tt / m / "seed_*" / "*.csv", T1_CMD[m]) for m in (T.REAP_GBFS, T.REAP_GREEDY)]
        search = [(tt / m / f"{m}__*.csv", T1_CMD[m]) for m in (T.HFF, T.BRFS)]
        if report == "table1":
            return "any", reap + search, argv
        if report == "table1_baselines":
            base = [(bl / T.SAHS / "shard_*.json", T1_CMD["sahs"]),
                    (bl / T.PLOI_HFF / "shard_*.jsonl", T1_CMD["ploi_hff"]),
                    (bl / T.PLOI_BRFS / "shard_*.jsonl", T1_CMD["ploi_brfs"]),
                    (bl / T.BT_H, T1_CMD["bt_h"]),
                    (bl / "pddlstream" / "*" / "[0-9][0-9][0-9].csv", T1_CMD["pddlstream"])]
            return "any", reap + search + base, argv
        per_problem = [(tt / m / s / "per_problem.csv", T1_CMD[m])
                       for m in (T.REAP_GBFS, T.REAP_GREEDY) for s in T.SEEDS]
        if report == "figure3":
            return "all", per_problem, argv
        return "all", [(bl / T.PDDLSTREAM[t] / "[0-9][0-9][0-9].csv",
                        "python baselines/pddlstream/bench/run.py %s outputs/baselines/%s" % (PDDLSTREAM_CONFIG[t],
                                                                                             T.PDDLSTREAM[t]))
                       for t in PDDLSTREAM_CONFIG] + per_problem, argv
    if report == "table2":
        o = Path(_opt(argv, "--root", str(out)))
        return "any", [(o / "confined" / "seed_*" / "*.csv",
                        "python scripts/eval.py --config configs/confined.yaml --seed <k>   (k = 1 to 3)"),
                       (o / "multilevel_shelf" / "seed_*" / "*.csv",
                        "python scripts/eval.py --config configs/multilevel_shelf.yaml --seed <k>")], argv
    if report in ("access19", "access19_composed"):
        a = Path(_opt(argv, "--root", str(out))) / "access19"
        cmd = "python scripts/eval.py --config configs/access19.yaml --seed <k>   (k = 1 to 3)"
        kinds = ("single", "composed") if report == "access19" else ("composed",)
        return "any", [(a / k / "*" / "seed_*" / "*__shard*.csv", cmd) for k in kinds], argv
    if report == "stacking":
        root = Path(_positional(argv) or out / "stacking")
        return "all", [(root / f"seed_{s}" / kind / "protocol.json",
                        f"python scripts/eval.py --config configs/stacking.yaml --seed {s} --kind {kind}")
                       for s in (1, 2, 3) for kind in ("best_val", "final")], argv
    if report == "joint":
        SETS = _module("joint_report").SETS
        o = Path(_opt(argv, "--root", str(out)))
        seeds = [k for k in _opt(argv, "--seeds", "1,2").split(",") if k]
        return "all", [(o / "joint" / f"seed_{k}" / sub / f"seed_{k}__{s}.csv",
                        f"python scripts/eval.py --config configs/joint.yaml --seed {k} --checkpoint <run directory>")
                       for k in seeds for _, sub, sets in SETS for s in sets], argv
    if report == "base_rate":
        d = _positional(argv)
        if d is None:
            d = str(out / "tabletop" / "base_rate")
            argv = [d] + argv
        cmd = f"{TT} --protocol base_rate"
        return "all", [(Path(d) / "rows_*.csv", cmd), (Path(d) / "meta_*.json", cmd)], argv
    if report == "noise":
        d = Path(_positional(argv, with_value=("--out",)) or out / "noise" / "sweep")
        return "any", [(d / "*.json", f"{TT} --protocol noise --checkpoint <checkpoint>   (writes outputs/noise; then: "
                                      "python scripts/report.py noise outputs/noise)")], argv
    return "any", [], argv


def missing_outputs(report, mode, items):
    """Print the missing outputs of ``report`` and the commands that create them, and exit with code 1."""
    miss = [(p, c) for p, c in items if not glob.glob(str(p))]
    if not miss or (mode == "any" and len(miss) < len(items)):
        return
    lines = [f"report.py {report}: missing outputs" + (" (none of these exists)" if mode == "any" else "") + ":"]
    lines += [f"  {p}" for p, _ in miss[:8]]
    if len(miss) > 8:
        lines.append(f"  ... and {len(miss) - 8} more")
    lines.append("Created by:")
    lines += [f"  {c}" for c in dict.fromkeys(c for _, c in miss)]
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(1)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in REPORTS:
        print(__doc__)
        raise SystemExit(0 if not argv or argv[0] in ("-h", "--help") else f"unknown report {argv[0]!r}")
    report, rest = argv[0], argv[1:]
    module = REPORTS[report][0]
    items = []
    if not {"-h", "--help", "--list"} & set(rest):
        mode, items, rest = needs(report, rest)
        missing_outputs(report, mode, items)
    sys.argv = [module] + rest
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except FileNotFoundError as e:
        lines = [f"report.py {report}: missing file {e.filename}"]
        data = Path(os.environ.get("REAP_DATA") or "data").resolve()
        if e.filename and data in Path(str(e.filename)).resolve().parents:
            lines.append("It is part of the datasets: bash scripts/download_assets.sh datasets")
        else:
            cmds = list(dict.fromkeys(c for _, c in items))
            if cmds:
                lines += ["The outputs of this report are created by:"] + [f"  {c}" for c in cmds]
        print("\n".join(lines), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
