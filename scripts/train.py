#!/usr/bin/env python
"""Train a REAP policy (Q-value GNN with motion-planner feedback) for one environment and seed.

    python scripts/train.py --config configs/<env>.yaml --seed <k> [--<parameter> <value> ...]

The ``train`` section of the config gives the keyword arguments of ``reap.rl.av_star.training.main``; every one of
them can be overridden on the command line (listed below). The ``env`` section (training and evaluation) and the
``train_env`` section (training only) set environment variables the code reads (for example the feasibility checker of
an environment). Several seeds: run the command once per seed.
"""
from __future__ import annotations

import argparse
import inspect
import os
import signal
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reap import config as C  # noqa: E402  (light: yaml only)


def parse_duration(text: str) -> float:
    """Seconds of a duration like 24h, 90m, 3600s or 3600."""
    t = str(text).strip().lower()
    mult = {"h": 3600, "m": 60, "s": 1, "d": 86400}.get(t[-1:], None)
    return float(t[:-1]) * mult if mult else float(t)


def start_walltime(seconds: float, grace_s: float = 120.0) -> None:
    """Stop training after ``seconds`` of wall time: SIGTERM to the trainer and its worker processes, then
    SIGKILL after ``grace_s``. The reported runs were cut this way (24 h; 60 h for 3D stacking)."""
    import multiprocessing

    def kill(sig):
        for p in multiprocessing.active_children():
            try:
                os.kill(p.pid, sig)
            except OSError:
                pass
        os.kill(os.getpid(), sig)

    def term():
        print(f"walltime of {seconds:.0f} s reached: stopping training", flush=True)
        threading.Timer(grace_s, kill, args=(signal.SIGKILL,)).start()
        kill(signal.SIGTERM)

    t = threading.Timer(seconds, term)
    t.daemon = True
    t.start()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    cfg = C.load(known.config) if known.config else {}

    # environment first: modules of the package read some variables at import time
    C.apply_env(cfg.get("env"), cfg.get("train_env"))
    os.environ.setdefault("MUJOCO_GL", "egl")

    from reap.rl.av_star.training import main as train_main
    params = inspect.signature(train_main).parameters
    defaults = {k: p.default for k, p in params.items()}
    train_cfg = dict(cfg.get("train") or {})
    unknown = sorted(set(train_cfg) - set(defaults))
    if unknown:
        raise SystemExit(f"unknown training parameters in {known.config}: {unknown}")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="configs/<env>.yaml (sections env, train)")
    ap.add_argument("--walltime", default=None, help="stop after this long, e.g. 24h (default: the config's walltime, else run until stopped)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="set an environment variable (after the config's env section)")
    ap.add_argument("--print-config", action="store_true", help="print the resolved parameters and exit")
    g = ap.add_argument_group("training parameters (reap.rl.av_star.training.main; config value or default shown)")
    for k, d in defaults.items():
        shown = train_cfg.get(k, d)
        g.add_argument("--" + k, type=C.parse_value, default=argparse.SUPPRESS, metavar="V",
                       help=f"(default: {shown!r})")
    a = ap.parse_args(argv)

    for kv in a.env:
        k, v = (kv.split("=", 1) + [""])[:2]
        os.environ[k] = v
    kwargs = {**train_cfg, **{k: getattr(a, k) for k in defaults if hasattr(a, k)}}
    if kwargs.get("seed") is not None and not kwargs.get("wandb_run_name"):
        name = Path(known.config).stem if known.config else "reap"
        kwargs["wandb_run_name"] = f"{name}_s{kwargs['seed']}"
    if a.print_config:
        for k in sorted(kwargs):
            print(f"{k}: {kwargs[k]!r}")
        return
    walltime = a.walltime or cfg.get("walltime")
    if walltime:
        start_walltime(parse_duration(walltime))
    os.chdir(REPO)
    train_main(**kwargs)


if __name__ == "__main__":
    main()
