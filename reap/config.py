"""Config loading shared by scripts/train.py and scripts/eval.py: YAML files, ``--key value`` overrides, templates.

A config file (``configs/<env>.yaml``) has the sections

    env:    environment variables the code reads (set before any module of this package is imported)
    train_env: environment variables for training only (scripts/train.py)
    train:  keyword arguments of ``reap.rl.av_star.training.main`` (see ``python scripts/train.py --help``)
    eval:   ``checkpoint`` (a template, e.g. checkpoints/tabletop/seed_{seed}), ``out`` (output directory template),
            ``default`` (protocol name) and ``protocols``: {name: {runner, env, args, vars, checkpoint, out, help}};
            ``runner`` is a function of ``reap.eval.runners``, ``args`` its keyword arguments, ``vars`` template
            variables with their defaults.
    walltime: optional default of ``scripts/train.py --walltime`` (e.g. 24h)

String values may use the templates {seed}, {out}, {checkpoint} and {<name>} for any scalar argument of the same
protocol. Relative paths are relative to the repository root.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def load(path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO / p
    with open(p) as fh:
        return yaml.safe_load(fh) or {}


def parse_value(text: str):
    """A command-line value as YAML (numbers, booleans, null, lists); plain text otherwise."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def parse_overrides(rest: list[str]) -> dict:
    """``--key value`` / ``--key=value`` / ``--flag`` (true) pairs into a dict; ``-`` and ``_`` are equivalent."""
    out, i = {}, 0
    while i < len(rest):
        tok = rest[i]
        if not tok.startswith("--"):
            raise SystemExit(f"unexpected argument {tok!r} (overrides are --key value)")
        key = tok[2:]
        if "=" in key:
            key, val = key.split("=", 1)
            i += 1
        elif i + 1 < len(rest) and not rest[i + 1].startswith("--"):
            val = rest[i + 1]
            i += 2
        else:
            val = "true"
            i += 1
        out[key.replace("-", "_")] = parse_value(val)
    return out


def apply_env(*envs: dict | None) -> None:
    """Set the environment variables of the given mappings, later ones winning; values are converted to str."""
    for env in envs:
        for k, v in (env or {}).items():
            os.environ[str(k)] = str(v).lower() if isinstance(v, bool) else str(v)


def fill(value, ctx: dict):
    """Expand {name} templates in strings (recursively in lists and dicts)."""
    if isinstance(value, str):
        for _ in range(4):
            new = re.sub(r"\{(\w+)\}", lambda m: str(ctx[m.group(1)]) if m.group(1) in ctx else m.group(0), value)
            if new == value:
                break
            value = new
        return value
    if isinstance(value, list):
        return [fill(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, ctx) for k, v in value.items()}
    return value


_ROOT_VARS = {"data": "REAP_DATA", "checkpoints": "REAP_CKPT", "outputs": "REAP_OUT"}
ROOT_DIRS = tuple(_ROOT_VARS)


def repo_path(p) -> str:
    """``p`` relative to the repository root unless absolute. A relative path under data/, checkpoints/ or outputs/
    moves to $REAP_DATA, $REAP_CKPT or $REAP_OUT when that variable is set (reap.paths)."""
    p = Path(str(p))
    if p.is_absolute():
        return str(p)
    var = _ROOT_VARS.get(p.parts[0]) if p.parts else None
    if var and os.environ.get(var):
        base = Path(os.environ[var])
        return str((base if base.is_absolute() else REPO / base).joinpath(*p.parts[1:]))
    return str(REPO / p)


def baseline_defaults(ap, name: str, argv) -> None:
    """Defaults of a baseline's run.py from ``--config`` (default configs/tabletop.yaml), section ``baselines.<name>``:
    {option: value} with the option names of that run.py (``-`` as ``_``). Command-line options still win."""
    pre = __import__("argparse").ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(REPO / "configs" / "tabletop.yaml"))
    known, _ = pre.parse_known_args(argv)
    cfg = load(known.config) if Path(known.config).exists() or (REPO / known.config).exists() else {}
    section = dict((cfg.get("baselines") or {}).get(name) or {})
    dests = {a.dest for a in ap._actions}
    unknown = sorted(set(section) - dests)
    if unknown:
        raise SystemExit(f"unknown options in {known.config}, baselines.{name}: {unknown}")
    ap.set_defaults(**section)
