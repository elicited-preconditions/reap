"""Which feasibility checker is active: explicit selection and provenance.

The confined shelf, the multi-level shelf and 3D stacking each name their checker in an environment variable that
must be set explicitly; checker initialisation raises otherwise.

    CONFINED_CHECKER     = honest        (confined-shelf: the honest grasp library)
    SHELF_ACCESS_CHECKER = v2            (shelf-access: tampanda shelf_access_v2, multi-level shelf, grasp library)
    MLB_CHECKER          = honest_grip   (multilevel-blocks / 3D stacking: the grip-aware check; needs the
                                          grip-aware domain with grip-centre / grip-end)

Every training run, validation process and evaluation records the active checker via :func:`describe` /
:func:`describe_data`: checker name, library file and sha256, tampanda commit, domain. Access-19 is told apart by
its domain (``tabletop-access-v2``); tabletop and 3D stacking have a single checker each and are labelled too, so
every number carries its checker.
"""
from __future__ import annotations

import functools
import glob
import hashlib
import os
import re
import subprocess
from pathlib import Path

CONFINED_VAR = "CONFINED_CHECKER"
SHELF_VAR = "SHELF_ACCESS_CHECKER"
MLB_VAR = "MLB_CHECKER"
_ALLOWED = {CONFINED_VAR: ("honest",), SHELF_VAR: ("v2",), MLB_VAR: ("honest_grip",)}
UNIFIED_DOMAIN = "reap-unified"


class CheckerSelectionError(RuntimeError):
    pass


def require(var: str) -> str:
    """The explicit checker choice in ``var``; raises if unset or not one of the allowed values."""
    allowed = _ALLOWED[var]
    raw = os.environ.get(var)
    if raw is None or not raw.strip():
        raise CheckerSelectionError(
            "%s is not set. Choose the feasibility checker explicitly: %s=%s (there is no default)."
            % (var, var, " | ".join(allowed)))
    v = raw.strip().lower()
    if v not in allowed:
        raise CheckerSelectionError("%s=%r is not valid; use one of %s" % (var, raw, allowed))
    return v


def _sha256(path, full=False):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.digest() if full else h.hexdigest()[:16]
    except OSError:
        return None


def _git_state(root, paths=()) -> str:
    """``<short sha>`` of the git checkout at ``root``; if the code under ``paths`` (git pathspecs) has uncommitted
    changes, ``+dirty(<modified>m,<untracked>u)@<sha256 of the diff vs HEAD + untracked file contents, 12 hex>`` --
    so the exact code of a run stays identifiable even before it is committed."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=120).stdout
    try:
        sha = git("rev-parse", "--short", "HEAD").decode().strip()
        st = git("status", "--porcelain", "--", *paths).decode().splitlines()
        if not st:
            return sha or "unknown"
        h = hashlib.sha256(git("diff", "HEAD", "--binary", "--", *paths))
        for f in sorted(git("ls-files", "--others", "--exclude-standard", "--", *paths).decode().splitlines()):
            h.update(f.encode())
            h.update(_sha256(Path(root) / f, full=True) or b"")
        mod = sum(1 for l in st if not l.startswith("??")); unt = sum(1 for l in st if l.startswith("??"))
        return "%s+dirty(%dm,%du)@%s" % (sha or "unknown", mod, unt, h.hexdigest()[:12])
    except Exception:
        return "unknown"


@functools.lru_cache(maxsize=1)
def tampanda_commit() -> str:
    """Git state of the tampanda checkout in use."""
    try:
        import tampanda
        # code only: the scene builders write tmp*.xml into the assets dir at runtime
        state = _git_state(Path(tampanda.__file__).resolve().parent.parent,
                           ("tampanda", ":(exclude)tampanda/environments/assets/franka_emika_panda/tmp*.xml"))
        if state == "unknown":
            # not a git checkout (e.g. installed from requirements.txt): record the installed release instead
            from importlib.metadata import version
            state = "tampanda==" + version("tampanda")
        return state
    except Exception:
        return "unknown"


@functools.lru_cache(maxsize=1)
def code_commit() -> str:
    """Git state of the checkout this module is imported from (reap/domains -> repo root)."""
    # code only (data/, checkpoints/ and out/ are not tracked)
    return _git_state(Path(__file__).resolve().parents[2], ("reap", "scripts"))


def describe(domain_name: str) -> dict:
    """Provenance of the checker the search filter would use for ``domain_name`` (raises for a shelf domain whose
    checker is not selected explicitly)."""
    lib = None
    if domain_name == "confined-shelf":
        require(CONFINED_VAR)
        from tampanda.symbolic.domains.confined_shelf.honest import DEFAULT_LIBRARY
        lib = os.environ.get("CONFINED_LIBRARY") or str(DEFAULT_LIBRARY)
        name = "confined-honest-library"
    elif domain_name == "shelf-access":
        require(SHELF_VAR)
        from tampanda.symbolic.domains.shelf_access_v2 import DEFAULT_LIBRARY
        lib = os.environ.get("SHELF_ACCESS_LIBRARY") or str(DEFAULT_LIBRARY)
        name = "shelf-v2-library"
    elif domain_name == "tabletop-access-v2":
        mode = os.environ.get("A19V2_CHECKER", "library").strip().lower()
        if mode == "library":
            from tampanda.symbolic.domains.access19_v2.checker import DEFAULT_LIBRARY
            lib = os.environ.get("A19V2_LIBRARY") or str(DEFAULT_LIBRARY)
        name = "access19-v2-" + mode
    elif domain_name in ("tabletop", "tabletop-manipulation"):
        name = "tabletop-v1"
    elif domain_name == "multilevel-blocks":
        require(MLB_VAR)
        from tampanda.symbolic.domains.multilevel_blocks.honest_grip import VERSION as _MLB_GRIP_VERSION
        name = "mlblocks-honest-grip-" + _MLB_GRIP_VERSION
    else:
        name = "unknown"
    sha = _sha256(lib) if lib else None
    return {"domain": domain_name, "checker": name, "library": lib, "library_sha256": sha,
            "tampanda_commit": tampanda_commit(), "code_commit": code_commit(),
            "label": name + ("@" + sha if sha else "")}


def domain_name_of(domain_file) -> str:
    m = re.search(r"\(\s*define\s*\(\s*domain\s+([^\s()]+)", Path(domain_file).read_text())
    return m.group(1) if m else "unknown"


def _unified_envs(problem_dirs) -> list:
    envs = set()
    for d in problem_dirs:
        for f in glob.glob(os.path.join(str(d), "*.pddl")):
            with open(f) as fh:
                for line in fh:
                    if line.startswith("; reap-env:"):
                        envs.add(line.split(":", 1)[1].strip()); break
                    if not line.startswith(";"):
                        break
    return sorted(envs)


def describe_data(domain_file, problem_dirs=()) -> dict:
    """Provenance for a dataset: its domain's checker, or -- for the reap-unified (joint) domain -- the checker of
    every environment tagged in ``problem_dirs``. Raises if any shelf environment's checker is not selected."""
    dn = domain_name_of(domain_file)
    if dn != UNIFIED_DOMAIN:
        return describe(dn)
    per = [describe(e) for e in _unified_envs(problem_dirs)]
    if not per:
        raise CheckerSelectionError("reap-unified data without '; reap-env:' tags in %s" % (list(problem_dirs),))
    return {"domain": UNIFIED_DOMAIN, "checker": ";".join("%s=%s" % (p["domain"], p["checker"]) for p in per),
            "library": ";".join(str(p["library"]) for p in per if p["library"]) or None,
            "library_sha256": ";".join(p["library_sha256"] for p in per if p["library_sha256"]) or None,
            "tampanda_commit": tampanda_commit(), "code_commit": code_commit(),
            "label": ";".join("%s=%s" % (p["domain"], p["label"]) for p in per)}


def log_line(info: dict, where: str) -> str:
    return "[checker] %s domain=%s checker=%s library=%s sha256=%s tampanda=%s code=%s" % (
        where, info["domain"], info["checker"], info["library"], info["library_sha256"], info["tampanda_commit"],
        info.get("code_commit"))
