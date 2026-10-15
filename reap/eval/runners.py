"""Evaluation pipelines behind the reported numbers, for one machine.

``scripts/eval.py`` calls one runner per protocol of a config file (``configs/<env>.yaml``, section ``eval``). Each
runner starts the evaluation modules of this package as subprocesses (``python -m reap.eval.<module>``), with the
environment variables those modules read, and runs sharded steps ``nproc`` at a time. Output layout:

    tabletop suite        <out>/<label>/{per_problem.csv, summary.csv, <label>__<set>.csv}
    protocol (Table 2)    <out>/<label>__<set>__<domain>__<ID|OOD>.csv
    stacking              <out>/{protocol.json, plans/, verify/, REPORT.txt, logs/}
    access19              <out>/<label>__shardNN.csv (+ plans/)
    base rate             <out>/{rows_<set>.csv, meta_<set>.json, sanity.json, base_rate_rank1_rows.csv}
    noise                 <out>/<mode>_<noise>_<value>.json, <out>/REPORT.md
"""
from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from reap.paths import data_root

REPO = Path(__file__).resolve().parents[2]


def _env(env: dict | None = None) -> dict:
    """The current environment plus ``env`` (values override), with the repository on PYTHONPATH."""
    e = dict(os.environ)
    e.update({k: str(v) for k, v in (env or {}).items()})
    e["PYTHONPATH"] = str(REPO) + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e["PYTHONUNBUFFERED"] = "1"
    return e


def run_module(module: str, args, env: dict | None = None, log: str | Path | None = None, show: str | None = None,
               check: bool = True) -> int:
    """``python -u -m <module> <args>``; output goes to ``log`` (if given) and to stdout (only the lines matching the
    regular expression ``show`` when a log is written)."""
    return run_cmd([sys.executable, "-u", "-m", module, *[str(a) for a in args]], env, log, show, check, module)


def run_script(script, args, env: dict | None = None, log: str | Path | None = None, show: str | None = None,
               check: bool = True) -> int:
    """``python -u <script> <args>`` (the baselines' scripts), as run_module."""
    return run_cmd([sys.executable, "-u", str(script), *[str(a) for a in args]], env, log, show, check, str(script))


def run_cmd(cmd, env, log, show, check, name) -> int:
    print("+ " + " ".join(cmd) + (f"   (log: {log})" if log else ""), flush=True)
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        fh = stack.enter_context(open(log, "a")) if log else None
        p = stack.enter_context(subprocess.Popen(cmd, cwd=REPO, env=_env(env), stdout=subprocess.PIPE,
                                                 stderr=subprocess.STDOUT, text=True, errors="replace"))
        for line in p.stdout:
            if fh:
                fh.write(line)
            if fh is None or (show and re.search(show, line)):
                sys.stdout.write(line)
                sys.stdout.flush()
        rc = p.wait()
    if check and rc:
        raise RuntimeError(f"{name} exited with status {rc}" + (f" (log: {log})" if log else ""))
    return rc


def parallel(jobs, nproc: int) -> int:
    """Run the callables ``jobs`` at most ``nproc`` at a time; returns the number that raised."""
    def call(job):
        try:
            job()
            return 0
        except Exception as e:  # noqa: BLE001 -- one failed shard must not stop the others
            print(f"ERROR: {e}", file=sys.stderr, flush=True)
            return 1
    with ThreadPoolExecutor(max(1, int(nproc))) as ex:
        return sum(ex.map(call, jobs))


# ------------------------------------------------------------------------------------------------ open tabletop suite
def suite(*, checkpoint, out, label, search_mode="gbfs", filter_actions=False, val_retries=3, max_budget=16384,
          time_limit_s=300, domain_file="domain.pddl", domain="tabletop", only_datasets=None, snapshot="live",
          env=None):
    """One model (or no model: hFF / BrFS) on the 15-set open-tabletop suite (Table 1, sample efficiency)."""
    args = ["--save_path", checkpoint, "--snapshot", snapshot, "--domain_file", domain_file, "--label", label,
            "--output_dir", out, "--search_mode", search_mode, "--val_retries", val_retries,
            "--max_budget", max_budget, "--time_limit_s", time_limit_s, "--domain", domain]
    if filter_actions:
        args.append("--filter_actions")
    if only_datasets:
        args += ["--only_datasets", only_datasets if isinstance(only_datasets, str) else ",".join(only_datasets)]
    run_module("reap.eval.eval_single_model", args, env=env, log=Path(out) / "logs" / f"{label}.log",
               show=r"^\s+\S+\s+\d+/\d+|label:|checkpoint:|wrote|FAILED|Error")


# ------------------------------------------------------------------------- uniform GBFS protocol (Table 2 environments)
def protocol(*, checkpoint, out, label, runs, workers=8, max_budget=16384, env=None):
    """The uniform GBFS protocol, one subprocess per evaluation set (confined shelf, multi-level shelf, joint training).

    ``runs``: list of {script: eval_protocol | joint_protocol, domain, data, subdir (optional), specs: [KIND:SET:TL]}.
    ``data`` is the dataset directory under $REAP_DATA (EVAL_DATA for eval_protocol, U_EVAL_DATA for joint_protocol);
    each spec is KIND:SET:TL[:NAME]: KIND is ID or OOD, SET the set directory inside ``data``, TL the per-problem time
    limit in seconds, NAME the set name in the output file names (default: SET with / replaced by _; the report
    scripts expect the names the configs give). Existing result files are skipped.
    """
    for run in runs:
        script = run.get("script", "eval_protocol")
        o = Path(out) / run["subdir"] if run.get("subdir") else Path(out)
        o.mkdir(parents=True, exist_ok=True)
        e = dict(env or {})
        e["U_EVAL_DATA" if script == "joint_protocol" else "EVAL_DATA"] = run["data"]
        for spec in run["specs"]:
            kind, subdir, tl, *name = spec.split(":")
            set_name = name[0] if name else subdir.replace("/", "_")
            csv = o / f"{label}__{set_name}__{run['domain']}__{kind}.csv"
            if csv.exists() and csv.stat().st_size:
                print(f"skip {csv}")
                continue
            run_module(f"reap.eval.{script}",
                       ["--domain", run["domain"], "--set", kind, "--subdir", subdir, "--model_path", checkpoint,
                        "--snapshot", "live", "--label", f"{label}__{set_name}", "--output_dir", o,
                        "--time_limit_s", tl, "--max_budget", max_budget, "--workers", run.get("workers", workers)],
                       env=e, log=o / "logs" / f"{label}__{set_name}.log",
                       show=r"^(RESULT|domain=|\[checker\])|Error|error")


# --------------------------------------------------------------------------------------------------------- 3D stacking
def stacking(*, checkpoint, out, data="stacking", nproc=4, verify_shards=40, env=None):
    """3D stacking: GBFS protocol, plans, continuous verification of every plan, report (reap.eval.stacking)."""
    from reap.eval import stacking as ST
    ST.run(checkpoint=checkpoint, out=out, data=data, nproc=nproc, verify_shards=verify_shards, env=env)


# ----------------------------------------------------------------------------------------------------------- Access-19
def access19(*, out, seed, sets, single=None, extraction=None, reload=None, nproc=1, shards=None,
             env=None):
    """Access-19 main rows: the single goal-conditioned policy and the two-phase pipeline (extraction policy, then
    reload policy), sharded over problems (reap.eval.access19_eval).

    ``sets``: {name: {root, sets, a19_sets, tl, nshards, modes}}: ``root`` the dataset directory (A19_ROOT, default
    $REAP_DATA/access19),
    ``sets`` the comma-separated set directories (SETS), ``a19_sets`` optional name=dir pairs (A19_SETS), ``tl`` the
    per-problem time limit, ``nshards`` the number of shards, ``modes`` the modes evaluated on this set.
    """
    label = f"seed_{seed}"
    for name, s in sets.items():
        for mode in s.get("modes", ("single", "composed")):
            if mode == "single" and not single or mode == "composed" and not (extraction and reload):
                continue
            o = Path(out) / mode / name / label
            o.mkdir(parents=True, exist_ok=True)
            n = int(s["nshards"])
            e = dict(env or {})
            e.update({"A19_ROOT": str(s.get("root") or data_root() / "access19"),
                      "A19_SETS": s.get("a19_sets", "")})
            ck = ["--a", single] if mode == "single" else ["--a", extraction, "--b", reload]
            jobs = []
            for k in (shards if shards is not None else range(n)):
                csv = o / f"{label}__shard{int(k):02d}.csv"
                if csv.exists() and csv.stat().st_size:
                    print(f"skip {csv}")
                    continue
                jobs.append(lambda k=k, ck=ck, e=e, o=o: run_module(
                    "reap.eval.access19_eval",
                    ["--mode", mode, *ck, "--label", label, "--shard", k, "--nshards", n, "--tl", s["tl"],
                     "--budget", 16384, "--sets", s["sets"], "--dump-plans", o / "plans", "--out", o],
                    env=e, log=o / "logs" / f"shard{int(k):02d}.log"))
            if parallel(jobs, nproc):
                print(f"WARN: a shard of {mode}/{name}/{label} failed (logs in {o / 'logs'})", file=sys.stderr)


# ---------------------------------------------------------------------------------------- base rate, rank-1 infeasible
def base_rate(*, out, sets, checkpoints=None, env=None):
    """Base rate of infeasible actions and rank-1 infeasibility under Q, uniform and hFF (all tabletop seeds at once),
    the --sanity check (verdict flips of repeated checks), then the aggregation. ``checkpoints``: {label: dir}
    (default: seed_1 ... seed_5 under $REAP_CKPT/tabletop)."""
    ck = ["--checkpoints", *[f"{k}={v}" for k, v in checkpoints.items()]] if checkpoints else []
    e = {"MUJOCO_GL": "disabled", **(env or {})}
    for s in sets:
        run_module("reap.eval.base_rate_rank1", ["--shard", s, "--out", out, *ck], env=e,
                   log=Path(out) / "logs" / f"{s}.log", show=r"^(SHARD|DONE|ERROR)|Error")
    run_module("reap.eval.base_rate_rank1", ["--sanity", "--out", out, *ck], env=e,
               log=Path(out) / "logs" / "sanity.log", show=r"^SANITY|Error")
    run_module("reap.eval.base_rate_report", [out], env=env)


# ------------------------------------------------------------------------------------------- perturbed observations
def noise(*, checkpoint, out, cells, n=0, env=None):
    """Execution under perturbed observations: one sweep cell per (mode, noise model, value), then the tables."""
    for mode, model, value in cells:
        f = Path(out) / f"{mode}_{model}_{value}.json"
        if f.exists():
            print(f"skip {f}")
            continue
        run_module("reap.eval.noise.sweep", ["--checkpoint", checkpoint, "--mode", mode, "--noise", model,
                                              "--value", value, "--n", n, "--out", f],
                   env=env, log=Path(out) / "logs" / f"{mode}_{model}_{value}.log", show=r"^(===|baseline|completed|"
                                                                                        r"replanned|wrote)")
    run_module("reap.eval.noise.report", [out, "--out", Path(out) / "REPORT.md"], env=env)


RUNNERS = {"suite": suite, "protocol": protocol, "stacking": stacking, "access19": access19, "base_rate": base_rate,
           "noise": noise}
