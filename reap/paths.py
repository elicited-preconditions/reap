"""Path roots of the REAP code release, set by environment variables (all optional).

    REAP_DATA   the data directory: ``<env>/domain.pddl`` (in git) and the problem sets of the datasets asset,
                ``<env>/{train,validation,eval,eval_<name>}/`` (default: ``data`` in the working directory)
    REAP_CKPT   the checkpoint root: ``<group>/seed_<k>/`` from the checkpoint assets (``tabletop/``, ``confined/``, ...)
                (default: ``checkpoints`` in the working directory)
    REAP_OUT    where training runs and evaluations write (default: ``outputs`` in the working directory)

Relative values resolve against the current working directory. Every script in ``scripts/`` changes to the repository
root before it reads these, so relative defaults mean ``<repo>/data``, ``<repo>/checkpoints`` and ``<repo>/outputs``. The functions read the environment on every call.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or default)


def data_root() -> Path:
    """``$REAP_DATA`` (default ``data``)."""
    return _env_path("REAP_DATA", "data")


def ckpt_root() -> Path:
    """``$REAP_CKPT`` (default ``checkpoints``)."""
    return _env_path("REAP_CKPT", "checkpoints")


def out_root() -> Path:
    """``$REAP_OUT`` (default ``outputs``)."""
    return _env_path("REAP_OUT", "outputs")
