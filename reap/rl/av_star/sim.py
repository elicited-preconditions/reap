"""Thin dispatch over per-domain feasibility checkers.

Domain-specific logic lives in :mod:`reap.domains.<name>` and registers
itself with :mod:`reap.domains.registry` at import time.  This module
imports each known backend in a try/except so that a missing tampanda
backend (e.g. an install without the multilevel_blocks sim code)
doesn't break unrelated domains.

"""
from __future__ import annotations

import importlib
import inspect
import os
import signal
from typing import Dict

import reap.xmimir as xmi

from reap.domains.registry import FeasibilityChecker, get_domain


# Known-domain modules to try at import time.  Add new domain modules here
# (or rely on explicit imports from the caller; registration is global).
_DOMAIN_MODULES = (
    "reap.domains.tabletop",
    "reap.domains.multilevel_blocks",
    "reap.domains.access19",
    "reap.domains.confined_shelf",
    "reap.domains.shelf_access",
)

for _mod in _DOMAIN_MODULES:
    try:
        importlib.import_module(_mod)
    except ImportError:
        # tampanda backend for this domain isn't installed in this
        # environment; fine, other domains can still work.  The error
        # only surfaces when filter_action() is called on a state whose
        # domain wasn't registered.
        pass


# Per-process per-domain checker instances.  Each spawn-started worker
# constructs its own on first filter_action call (the dict is module-level
# but module state isn't shared across spawn processes).
_CHECKERS: Dict[str, FeasibilityChecker] = {}


# UNIFIED DOMAIN (multi-environment training): problems parsed under the maximal
# domain `reap-unified` (data/joint/domain.pddl) all report that domain name,
# so the simulator is chosen from the problem file's first-line tag `; reap-env: <original domain>`
# of the joint-training datasets. Read once per problem file, cached. Any other domain is used as is.
UNIFIED_DOMAIN = "reap-unified"
_ENV_OF_FILE: Dict[str, str] = {}


def _env_of(problem) -> str:
    path = str(problem.filepath)
    env = _ENV_OF_FILE.get(path)
    if env is None:
        with open(path) as fh:
            for line in fh:
                if line.startswith("; reap-env:"):
                    env = line.split(":", 1)[1].strip()
                    break
                if not line.startswith(";"):
                    break
        if not env:
            raise RuntimeError("problem %s is parsed under %s but has no '; reap-env:' tag"
                               % (path, UNIFIED_DOMAIN))
        _ENV_OF_FILE[path] = env
    return env


def _get_checker(state: xmi.XState) -> FeasibilityChecker:
    domain_name = state.problem.domain.name
    if domain_name == UNIFIED_DOMAIN:
        domain_name = _env_of(state.problem)
    if domain_name not in _CHECKERS:
        _CHECKERS[domain_name] = get_domain(domain_name).feasibility_checker()
    return _CHECKERS[domain_name]


def get_checker_for(domain_name: str):
    """Return the per-process checker for ``domain_name`` (or None).

    Used by the eval protocol to read per-problem motion-planner-query
    counts from the checker's result cache (cache size == unique
    check_action invocations).  Not used in training.
    """
    return _CHECKERS.get(domain_name)


# Lightweight per-process filter counters, read by the eval protocol to
# attribute total feasibility checks + infeasible verdicts to each problem
# (reset between problems).  A dict-increment per call is negligible vs the
# checker's ms-scale check_action, so training pays no meaningful cost.
_FILTER_STATS: Dict[str, int] = {"calls": 0, "infeasible": 0, "timeouts": 0}


def reset_filter_stats(clear_cache: bool = False) -> None:
    """Zero the per-process filter counters.

    ``clear_cache=True`` (MP/plan accounting, used by the Table 1 evaluation) also empties the
    verdict cache of every checker built in this process. The next problem then queries the planner afresh, verdicts
    are reused only within a problem, and :func:`get_mp_calls` counts that problem's distinct queries. Default False
    resets the counters only."""
    _FILTER_STATS["calls"] = 0
    _FILTER_STATS["infeasible"] = 0
    _FILTER_STATS["timeouts"] = 0
    if clear_cache:
        for _ck in _CHECKERS.values():
            _cache = getattr(_ck, "_cache", None)
            if _cache is not None:
                _cache.clear()


def get_mp_calls() -> int:
    """MP/plan: distinct (state, action) feasibility queries since the last ``reset_filter_stats(clear_cache=True)``
    = the total size of the verdict caches of the checkers built in this process (a tabletop evaluation builds
    only the tabletop checker)."""
    return sum(len(getattr(_ck, "_cache", None) or ()) for _ck in _CHECKERS.values())


def get_filter_stats() -> Dict[str, int]:
    return dict(_FILTER_STATS)


# Hard wall-clock cap on a single checker.check() call (env FILTER_TIMEOUT_S,
# default 30s, generous vs. the ms-scale median cost).  A pathological joint
# config can make tampanda's linear-lerp collision sweep (plan_joint_lerp ->
# _segment_collision_free_n -> is_collision_free) spin forever with no
# internal timeout of its own.  SIGALRM only interrupts at a Python bytecode
# boundary; that loop runs in pure-Python frames over MuJoCo steps (not one
# opaque blocking C call), so the alarm can preempt it.  0 disables.
FILTER_TIMEOUT_S = float(os.environ.get("FILTER_TIMEOUT_S", "30"))


class _FilterHang(Exception):
    pass


_IN_CHECK = False


def _alarm_handler(signum, frame):
    # Interrupt only a RUNNING checker.check(). An alarm delivered just after the check returned (before
    # signal.alarm(0) in the finally) would raise outside the try and kill the worker process; such a late
    # alarm is ignored.
    if _IN_CHECK:
        raise _FilterHang()


def filter_action(state, action) -> bool:
    """Return True iff ``action`` is physically infeasible in ``state``.

    Dispatches to the registered :class:`FeasibilityChecker` for the
    PDDL domain that owns ``state``.  A hung checker.check() (see
    FILTER_TIMEOUT_S above) is treated as infeasible so it can't wedge the
    calling worker forever; same contract as a legitimately-infeasible
    action, just counted separately in _FILTER_STATS["timeouts"].
    """
    _FILTER_STATS["calls"] += 1
    if FILTER_TIMEOUT_S > 0:
        # Build the checker BEFORE arming the alarm. Checkers initialise lazily on first
        # use (tabletop: FrankaEnvironment + a CollisionWorkerPool, >180 s on a single core); if the
        # alarm fires inside init(), the checker is left half-built (_initialized=True,
        # _checker=None) and every later call raises. init() is idempotent, so this only moves the
        # one-time setup out of the per-call timeout -- no verdict can change.
        # Pass the state where init() takes one: multilevel-blocks infers its scene from the first
        # problem and raises on a stateless init(); every other checker ignores the argument or doesn't take it.
        _ck = _get_checker(state)
        if hasattr(_ck, "init"):
            if "state" in inspect.signature(_ck.init).parameters:
                _ck.init(state=state)
            else:
                _ck.init()
        global _IN_CHECK
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(FILTER_TIMEOUT_S)))
        try:
            _IN_CHECK = True
            try:
                result = _get_checker(state).check(state, action)
            finally:
                _IN_CHECK = False
        except _FilterHang:
            print(f"[filter_action] TIMEOUT after {FILTER_TIMEOUT_S:.0f}s on "
                  f"{action}, treating as infeasible", flush=True)
            _FILTER_STATS["timeouts"] += 1
            result = True
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        result = _get_checker(state).check(state, action)
    if result:
        _FILTER_STATS["infeasible"] += 1
    return result
