"""Domain registry: pluggable feasibility checkers and eval suites.

To add a new learning domain:

1. Write a class implementing :class:`FeasibilityChecker` (lazy init +
   ``check(state, action) -> bool``; True = infeasible).
2. Bundle it with a list of :class:`EvalSuite` entries and a
   ``reference_problem_dir`` into a :class:`DomainSpec`.
3. Call :func:`register_domain` at module-import time.
4. Add the module to the import list in
   :mod:`reap.rl.av_star.sim` so it is loaded at startup.

The PDDL domain name (the symbol in ``(define (domain <name>) ...)``) is
the registry key: :mod:`sim.filter_action` reads ``state.problem.domain.name``
and dispatches.  Convention: keep the registry key identical to the PDDL
domain name (e.g. ``"tabletop"`` or ``"multilevel-blocks"``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Type

import reap.xmimir as xmi


class FeasibilityChecker(Protocol):
    """Per-domain physical-feasibility oracle.

    Called once per (state, action) pair when ``filter_action`` is active.
    Implementations must be safe to instantiate inside spawn-started
    worker processes; expensive resources (MuJoCo env, collision pool,
    motion-planner) should be constructed lazily via :meth:`init`, never
    at module import (otherwise spawn-IPC bootstrap deadlocks on Linux).
    """

    def init(self) -> None:
        """Lazy initialization.  Idempotent.  Construct heavy resources
        here on first :meth:`check` call inside each process."""

    def check(self, state: xmi.XState, action: xmi.XAction) -> bool:
        """Return True if ``action`` is physically INFEASIBLE in ``state``
        (i.e. should be skipped by the search), False if feasible.
        Implementations should cache aggressively."""


@dataclass(frozen=True)
class EvalSuite:
    """One eval cell: a directory of validation problems + a short label.

    The label names the suite in the evaluation outputs (e.g.
    per_problem.csv); the path is relative to the repo root and points at a
    directory containing ``*.pddl`` problem files (no ``domain.pddl``;
    the domain is loaded from :attr:`DomainSpec.reference_problem_dir`).
    """
    label: str
    path: str


@dataclass(frozen=True)
class DomainSpec:
    """Registry record for a learning domain.

    Attributes:
        name: PDDL domain name (the symbol after ``define (domain X)``).
            Used as the registry key and the dispatch key in
            :func:`sim.filter_action`.
        feasibility_checker: Class (not instance) of the per-domain
            :class:`FeasibilityChecker`.  Instances are created lazily
            per-process by :func:`sim._get_checker`.
        reference_problem_dir: Directory (relative to the data root,
            ``$REAP_DATA``, see :mod:`reap.paths`) that ships domain.pddl.
            Used by eval scripts to resolve domain files when an eval
            suite ships only problem instances (no domain.pddl).
        eval_suites: Tuple of eval cells used by post-training
            evaluation; consumed by ``eval_single_model.py`` and
            ``eval_all_datasets.py``.
        domain_files: The domain files in
            ``$REAP_DATA/<reference_problem_dir>/``.
            First is the default.
    """
    name: str
    feasibility_checker: Type[FeasibilityChecker]
    reference_problem_dir: str
    eval_suites: tuple[EvalSuite, ...]
    domain_files: tuple[str, ...] = ("domain.pddl",)


_REGISTRY: dict[str, DomainSpec] = {}


def register_domain(spec: DomainSpec) -> None:
    """Register a :class:`DomainSpec` under its ``name`` key.

    Raises ``ValueError`` on duplicate registration so silent shadowing
    between modules can't happen.
    """
    if spec.name in _REGISTRY:
        existing = _REGISTRY[spec.name]
        if existing is spec:
            return  # idempotent on re-import
        raise ValueError(
            f"domain {spec.name!r} already registered with a different spec "
            f"({existing!r}); current={spec!r}"
        )
    _REGISTRY[spec.name] = spec


def get_domain(name: str) -> DomainSpec:
    """Look up a registered domain by name.  Raises ``KeyError`` if absent."""
    if name not in _REGISTRY:
        raise KeyError(
            f"no domain registered for {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_domains() -> list[str]:
    """Names of all currently-registered domains."""
    return sorted(_REGISTRY)
