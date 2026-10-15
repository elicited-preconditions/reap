"""Make sure ``xmimir`` and ``tampanda`` are importable.

Both come from the environment: ``xmimir`` is this repository's ``reap/xmimir`` (installed by ``pip install --no-deps -e .``),
and TAMPanda is installed from ``requirements-tampanda.txt`` (README.md, "Install").
"""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Root of the code base that provides ``xmimir`` (the REAP repository)."""
    import reap.xmimir as xmimir
    return Path(xmimir.__file__).resolve().parents[2]


def manipulation_root() -> Path:
    """Directory that contains the installed ``tampanda`` package."""
    import tampanda

    return Path(tampanda.__file__).resolve().parents[1]


def ensure_upstream_importable() -> None:
    """Idempotent. Call before importing ``xmimir``, ``pymimir`` or ``tampanda``."""
    try:
        import reap.xmimir  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "reap.xmimir is not importable: install the REAP repository (pip install --no-deps -e .)"
        ) from exc
    try:
        import tampanda  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'tampanda is not importable: install it from requirements-tampanda.txt (README.md, "Install")'
        ) from exc
