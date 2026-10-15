#!/usr/bin/env python
"""Run one PDDLStream configuration on a range of manifest entries, one new Python process per problem (as in the
Table 1 runs).

    python baselines/pddlstream/bench/run.py CONFIG OUTDIR [FIRST [LAST]]

CONFIG       test_ffastar2 | test_ffeager | gen_ffeager | informed_cells_occlusion
OUTDIR       result directory; problem i of bench/manifest.tsv writes OUTDIR/NNN.csv
FIRST, LAST  0-based manifest line range, inclusive (default: every line)

Environment:
  REAP_DATA       the REAP dataset asset's data/ directory, which contains tabletop/eval/<set>/
                  (default: data/ of the repository)
  PDDLSTREAM_DIR  pddlstream checkout with a built Fast Downward (default: baselines/pddlstream/pddlstream, the
                  git submodule)

The per-problem limits of the Table 1 runs: 300 s (--max-time), one inner worker (plain RRT), seed 0.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

CONFIGS = {   # CONFIG -> (script, planner)
    "test_ffastar2": ("bench_problem_test_stream.py", "ff-astar2"),
    "test_ffeager": ("bench_problem_test_stream.py", "ff-eager"),
    "gen_ffeager": ("bench_problem_cell_generator.py", "ff-eager"),
    "informed_cells_occlusion": ("bench_problem_informed_cells_occlusion.py", "ff-eager"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", choices=sorted(CONFIGS))
    ap.add_argument("outdir")
    ap.add_argument("first", nargs="?", type=int, default=0)
    ap.add_argument("last", nargs="?", type=int, default=None)
    ap.add_argument("--max-time", type=float, default=300, help="seconds per problem (Table 1: 300)")
    a = ap.parse_args(argv)

    manifest = [l.rstrip("\n").split("\t") for l in (HERE / "manifest.tsv").read_text().splitlines() if l.strip()]
    last = len(manifest) - 1 if a.last is None else a.last
    # absolute paths: each per-problem script changes into a temporary directory
    data = Path(os.environ.get("REAP_DATA") or REPO / "data").resolve()
    if not (data / "tabletop").is_dir():
        sys.exit(f"REAP_DATA={data} has no tabletop/ (extract the dataset asset, or point REAP_DATA at its data/)")
    pds = Path(os.environ.get("PDDLSTREAM_DIR") or HERE.parent / "pddlstream").resolve()
    if not list((pds / "downward" / "builds").glob("release*/bin")):
        sys.exit(f"no Fast Downward build under {pds}/downward/builds/ (build it first, see the README)")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(pds) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MUJOCO_GL", "egl")
    script, planner = CONFIGS[a.config]
    out = Path(a.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fail = 0
    for i in range(a.first, last + 1):
        if i >= len(manifest):
            sys.exit(f"manifest has no line {i}")
        dataset, problem = manifest[i][:2]
        cmd = [sys.executable, str(HERE / script), "--dataset", dataset, "--problem", str(data / problem),
               "--algorithm", "adaptive", "--planner", planner, "--seed", "0", "--inner-workers", "1",
               "--max-time", str(a.max_time), "--out", str(out / f"{i:03d}.csv")]
        if subprocess.run(cmd, env=env).returncode != 0:
            print(f"task {i} ({dataset}/{Path(problem).name}) exited non-zero", file=sys.stderr)
            fail += 1
    if fail:
        sys.exit(f"{fail} task(s) failed")


if __name__ == "__main__":
    main()
