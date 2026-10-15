#!/usr/bin/env python
"""Evaluate released (or newly trained) checkpoints with the protocols behind the reported numbers.

    python scripts/eval.py --config configs/<env>.yaml --seed <k> [--protocol NAME] [--<arg> <value> ...]
    python scripts/eval.py --config configs/<env>.yaml --list

The ``eval`` section of the config names the protocols of an environment (for the open tabletop: gbfs, greedy, the
hFF and BrFS search baselines, sample efficiency, base rate, noise). Each protocol is a runner of
``reap.eval.runners`` with its arguments; ``--<arg> <value>`` overrides one of them, ``--checkpoint`` and ``--out``
override the checkpoint and output directory, ``--<var> <value>`` sets a template variable of the protocol (its
``vars``, e.g. the training-set size of the sample-efficiency checkpoints). Several seeds: run the command once per
seed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reap import config as C  # noqa: E402  (light: yaml only)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/<env>.yaml (sections env, eval)")
    ap.add_argument("--seed", type=int, default=1, help="seed of the checkpoint to evaluate (default 1)")
    ap.add_argument("--protocol", default=None, help="protocol of the config's eval section (default: its default)")
    ap.add_argument("--checkpoint", default=None, help="checkpoint directory (default: the config's template)")
    ap.add_argument("--out", default=None, help="output directory (default: the config's template)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="set an environment variable")
    ap.add_argument("--list", action="store_true", help="list the protocols of the config and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved runner call and exit")
    a, rest = ap.parse_known_args(argv)
    overrides = C.parse_overrides(rest)

    cfg = C.load(a.config)
    ev = cfg.get("eval") or {}
    protocols = ev.get("protocols") or {}
    if a.list:
        for name, p in protocols.items():
            tv = " ".join(f"--{k} {v}" for k, v in (p.get("vars") or {}).items())
            print(f"{name:20s} runner={p.get('runner')}  {p.get('help', '')}" + (f"  [{tv}]" if tv else ""))
        return
    name = a.protocol or ev.get("default")
    if name not in protocols:
        raise SystemExit(f"unknown protocol {name!r}; {a.config} has: {', '.join(protocols)}")
    proto = protocols[name]

    ctx = {"seed": a.seed, "protocol": name, "env_name": Path(a.config).stem}
    tvars = dict(proto.get("vars") or {})      # template variables (not runner arguments); --<var> overrides them
    for k in list(overrides):
        if k in tvars:
            tvars[k] = overrides.pop(k)
    ctx.update(tvars)
    args = dict(proto.get("args") or {})
    args.update(overrides)
    ctx.update({k: v for k, v in args.items() if isinstance(v, (str, int, float)) and k not in ctx})
    ckpt = a.checkpoint or proto.get("checkpoint", ev.get("checkpoint"))
    if ckpt is not None:
        ctx["checkpoint"] = C.repo_path(C.fill(ckpt, ctx))
    out = a.out or proto.get("out", ev.get("out", "outputs/{env_name}/{protocol}"))
    ctx["out"] = C.repo_path(C.fill(out, ctx))
    args = C.fill(args, ctx)
    if "checkpoint" in ctx and proto.get("pass_checkpoint", True):
        args.setdefault("checkpoint", ctx["checkpoint"])
    args.setdefault("out", ctx["out"])
    for k in list(args):
        v = args[k]
        if v is not None and (k in ("checkpoint", "out") or isinstance(v, str) and v.split("/", 1)[0] in C.ROOT_DIRS):
            args[k] = C.repo_path(v)

    env = {**(cfg.get("env") or {}), **(ev.get("env") or {}), **(proto.get("env") or {})}
    for kv in a.env:
        k, v = (kv.split("=", 1) + [""])[:2]
        env[k] = v
    if a.dry_run:
        print(f"runner {proto['runner']}\nenv {env}")
        for k in sorted(args):
            print(f"  {k}: {args[k]!r}")
        return
    C.apply_env(env)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.chdir(REPO)
    from reap.eval import runners
    runner = runners.RUNNERS[proto["runner"]]
    runner(**args)


if __name__ == "__main__":
    main()
