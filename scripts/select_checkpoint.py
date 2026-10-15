#!/usr/bin/env python
"""Pick the checkpoint to report from a training run, with the rules behind the released Table 2 rows.

    python scripts/select_checkpoint.py best_val --log <training log> [--log <next link's log> ...] [--run_dir <dir> ...]
    python scripts/select_checkpoint.py selection_set --run_dir <dir> [--nproc N] [--dry-run | --score-only]

best_val (3D stacking; Access-19 reload policy). Each time the validator writes checkpoint/checkpoint_<N> it prints
"saving best|periodic model (solved=k/n)"; the i-th such line of a training log belongs to the i-th checkpoint
directory in order of N. The log is the output of scripts/train.py (for example `... 2>&1 | tee train.log`); the run
directory is $REAP_OUT/model/<SAVE_PATH of the log> unless --run_dir is given.
  One log (3D stacking): the pick is the last "best" save of the run.
  Several logs (a warm-start chain, oldest link first; the reload policy): the pick is the save with the most solved
  validation problems over the whole chain, best and periodic saves alike; ties go to the later link, then to the
  later checkpoint.

selection_set (Access-19 single policy). The candidates of a run are checkpoint/checkpoint_<N> for N >= 20 and the
final weights in the run directory (the trainer writes them every 10 iterations, so they are the last write before the
run stopped). Each candidate is evaluated on the selection set
(data/access19/selection, 93 problems) with reap.eval.access19_eval --mode single, 100 s per problem, budget 16384.
The pick has the most replay-verified solves; ties go to the lower mean MP/plan over the solved problems,
then to the later candidate (the final weights count as later than every checkpoint). Shards whose CSV exists are
skipped, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SAVE = re.compile(r"saving (best|periodic) model \(solved=(\d+)/(\d+)")


def ckpt_dirs(run_dir: Path):
    return sorted((p for p in (run_dir / "checkpoint").glob("checkpoint_*") if p.is_dir()),
                  key=lambda p: int(p.name.rsplit("_", 1)[1]))


def saves(log: Path, run_dir: Path | None):
    """The validator's save lines of one training log, each paired with its checkpoint directory."""
    lines, save_path = [], None
    with open(log, errors="replace") as fh:
        for line in fh:
            if save_path is None and line.startswith("SAVE_PATH="):
                save_path = line.split("=", 1)[1].strip()
            m = SAVE.search(line)
            if m:
                lines.append({"tag": m.group(1), "solved": int(m.group(2)), "n": int(m.group(3))})
    if run_dir is None:
        if save_path is None:
            sys.exit(f"{log}: no SAVE_PATH= line; pass --run_dir")
        from reap.paths import out_root
        run_dir = (out_root() / "model" / save_path).resolve()
    ck = ckpt_dirs(run_dir)
    # a checkpoint directory without its line yet is a save in progress: drop it; fewer directories cannot be mapped
    if not len(lines) <= len(ck) <= len(lines) + 1:
        sys.exit(f"{log}: {len(lines)} save lines but {len(ck)} checkpoint directories in {run_dir}")
    ck = ck[:len(lines)]
    mt = [(c / "q_gnn.pt").stat().st_mtime for c in ck if (c / "q_gnn.pt").exists()]
    if mt != sorted(mt):
        sys.exit(f"{run_dir}: checkpoint order by N differs from order by modification time")
    for s, c in zip(lines, ck):
        s.update(path=c, ckpt=int(c.name.rsplit("_", 1)[1]))
    return run_dir, lines


def cmd_best_val(a):
    if a.run_dir and len(a.run_dir) != len(a.log):
        sys.exit("give one --run_dir per --log")
    cands = []
    for i, log in enumerate(a.log):
        run_dir, s = saves(log, a.run_dir[i] if a.run_dir else None)
        print(f"link {i}: {log} -> {run_dir}: {len(s)} saves, max solved "
              f"{max((x['solved'] for x in s), default='-')}")
        cands += [dict(x, link=i) for x in s]
    if not cands:
        sys.exit("no validation saves")
    if len({x["n"] for x in cands}) != 1:
        sys.exit("the validation-set size differs between save lines: %s" % sorted({x["n"] for x in cands}))
    if len(a.log) == 1:
        best = [x for x in cands if x["tag"] == "best"]
        if not best:
            sys.exit("no 'best' save")
        pick = best[-1]
        rule = "last 'best' save"
    else:
        pick = max(cands, key=lambda x: (x["solved"], x["link"], x["ckpt"]))
        rule = "most solved over the chain; ties -> later link, later checkpoint"
    ties = [f"link{x['link']}/checkpoint_{x['ckpt']}" for x in cands if x["solved"] == pick["solved"]]
    print(f"rule: {rule}")
    print(f"saves with {pick['solved']}/{pick['n']} solved: {' '.join(ties)}")
    print(f"pick: {pick['path']}  (link {pick['link']}, {pick['tag']} save, solved {pick['solved']}/{pick['n']})")


def candidates(run_dir: Path, min_ckpt: int):
    """(sort key, label, path): validator checkpoints N >= min_ckpt, then the final weights of the run."""
    out = []
    for c in ckpt_dirs(run_dir):
        n = int(c.name.rsplit("_", 1)[1])
        if n >= min_ckpt and (c / "q_gnn.pt").exists():
            out.append(((0, n), "c%03d" % n, c))
    if (run_dir / "q_gnn.pt").exists():
        out.append(((1, 0), "final", run_dir))
    return out


def score(o: Path, label: str, nshards: int):
    """(solved, mean MP calls over solved, problems) of one candidate, or None while a shard is missing."""
    files = [o / f"{label}__shard{k:02d}.csv" for k in range(nshards)]
    if not all(f.exists() and f.stat().st_size for f in files):
        return None
    rows = [r for f in files for r in csv.DictReader(open(f))]
    sol = [r for r in rows if r["solved"] == "1" and r.get("verify", "ok") == "ok"]
    mp = statistics.mean(int(r["mp_calls_dedup"]) for r in sol) if sol else float("inf")
    return len(sol), mp, len(rows)


def cmd_selection_set(a):
    from reap.paths import data_root, out_root
    root = a.data or (data_root() / "access19").resolve()
    env = {"A19_ROOT": str(root), "A19_SETS": f"{a.set}={a.set}", "OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6"}
    for run_dir in a.run_dir:
        out = (a.out or out_root() / "access19" / "selection").resolve() / run_dir.name
        cands = candidates(run_dir, a.min_ckpt)
        print(f"{run_dir}: {len(cands)} candidates: {' '.join(c[1] for c in cands)}")
        args = lambda path, label, k: ["--mode", "single", "--a", path, "--label", label, "--sets", a.set, "--tl", a.tl,
                                       "--budget", a.budget, "--shard", k, "--nshards", a.nshards, "--out", out / label]
        if a.dry_run:
            print("dry run; each candidate and shard k runs:  A19_ROOT=%s A19_SETS=%s python -m reap.eval.access19_eval %s"
                  % (env["A19_ROOT"], env["A19_SETS"], " ".join(map(str, args("<candidate>", "<label>", "k")))))
            continue
        if not a.score_only:
            from reap.eval.runners import parallel, run_module
            jobs = [lambda path=path, label=label, k=k: run_module(
                        "reap.eval.access19_eval", args(path, label, k), env=env,
                        log=out / label / "logs" / f"shard{k:02d}.log")
                    for _, label, path in cands for k in range(a.nshards)
                    if not (out / label / f"{label}__shard{k:02d}.csv").exists()]
            if parallel(jobs, a.nproc):
                print(f"WARN: a shard failed (logs under {out}/<candidate>/logs)", file=sys.stderr)
        scored = []
        for key, label, path in cands:
            s = score(out / label, label, a.nshards)
            if s is None:
                print(f"  {label}: incomplete")
                continue
            print(f"  {label}: {s[0]}/{s[2]} solved, mean MP calls {s[1]:.1f}")
            scored.append((s[0], -s[1], key, label, path))
        if not scored:
            print("no complete candidate")
            continue
        k, negmp, _, label, path = max(scored)
        print(f"pick: {path}  ({label}, {k} solved, mean MP calls {-negmp:.1f})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="rule", required=True)
    b = sub.add_parser("best_val", help="last best validation save (stacking); chains: most solved (reload)")
    b.add_argument("--log", action="append", type=Path, required=True,
                   help="training log; repeat for the links of a warm-start chain, oldest first")
    b.add_argument("--run_dir", action="append", type=Path, default=[],
                   help="run directory of each log (default: $REAP_OUT/model/<SAVE_PATH of the log>)")
    s = sub.add_parser("selection_set", help="most verified solves on the selection set (Access-19 single policy)")
    s.add_argument("--run_dir", action="append", type=Path, required=True, help="training run directory; repeatable")
    s.add_argument("--data", type=Path, default=None, help="Access-19 data directory (default $REAP_DATA/access19)")
    s.add_argument("--set", default="selection", help="problem set inside the data directory (default selection)")
    s.add_argument("--tl", type=float, default=100.0, help="seconds per problem (default 100)")
    s.add_argument("--budget", type=int, default=16384, help="search budget (default 16384)")
    s.add_argument("--min_ckpt", type=int, default=20, help="first validator checkpoint considered (default 20)")
    s.add_argument("--nshards", type=int, default=6, help="shards per candidate (default 6)")
    s.add_argument("--nproc", type=int, default=1, help="shards run at the same time (default 1)")
    s.add_argument("--out", type=Path, default=None,
                   help="output root; each run writes to <out>/<run>/<candidate>/ (default $REAP_OUT/access19/selection)")
    s.add_argument("--dry-run", action="store_true", help="list the candidates and the evaluation command, then exit")
    s.add_argument("--score-only", action="store_true", help="evaluate nothing; pick among the complete candidates")
    a = ap.parse_args(argv)
    a.run_dir = [p.resolve() for p in a.run_dir]
    if a.rule == "best_val":
        a.log = [p.resolve() for p in a.log]
    else:
        a.data, a.out = (p.resolve() if p else None for p in (a.data, a.out))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.chdir(REPO)
    (cmd_best_val if a.rule == "best_val" else cmd_selection_set)(a)


if __name__ == "__main__":
    main()
