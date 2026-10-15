"""Access-19 domain (``tabletop-access-v2``) -- feasibility from the tampanda access19_v2 grasp library.

The shelf (centred bookcase, 5 x 7 identical boxes per level, one frontal grasp) has a precomputed grasp library:
per cell, a few verified motions (home -> approach -> grasp -> retreat -> home, every arm link, the hand and the held
box collision-checked against the exact meshes), each with the set of cells whose box it would touch. An action on
cell c is feasible iff one of c's motions touches no occupied cell, so ``check`` is a pure-Python bitmask lookup
(microseconds): no MuJoCo, IK or planner in the worker, nothing to initialise lazily beyond reading a JSON file,
nothing that can hang (no FILTER_TIMEOUT_S needed). See
``tampanda/symbolic/domains/access19_v2/README.md`` (semantics, validation against online planning, geometry).

``A19V2_LIBRARY`` selects a library other than the shipped one (e.g. a different column pitch); problems must
come from the same geometry (library and datasets are tied by the geometry digest recorded in the library).

``A19V2_CHECKER=online`` replaces the lookup by planning in the loop (``tampanda ... access19_v2.online``): every
distinct query runs the full stack -- DP over all IK branches against the objects present, dense exact checks, VAMP
transits validated against the exact meshes -- in the worker (seconds per query; needs MuJoCo + frantik + vamp, see
the tampanda README; FILTER_TIMEOUT_S applies). Same verdicts as the library on the validation set (700/700).
"""
from __future__ import annotations

import os

import reap.xmimir as xmi

from tampanda.symbolic.domains.access19_v2.checker import DEFAULT_LIBRARY, GraspLibrary

from reap.domains.registry import DomainSpec, register_domain


class Access19V2FeasibilityChecker:
    def __init__(self) -> None:
        self._lib = None          # GraspLibrary (lookup) or OnlineChecker (planning); both have .feasible
        # distinct (occupancy, cell) queries; len(_cache) is the MP call count of a problem (MP/plan)
        self._cache: dict = {}

    def init(self, state: xmi.XState | None = None) -> None:
        if self._lib is not None:
            return
        mode = os.environ.get("A19V2_CHECKER", "library")
        lib = GraspLibrary.load(os.environ.get("A19V2_LIBRARY", str(DEFAULT_LIBRARY)))
        if mode == "library":
            self._lib = lib
        elif mode == "online":
            from tampanda.symbolic.domains.access19_v2.layout import ShelfGeometry
            from tampanda.symbolic.domains.access19_v2.online import OnlineChecker
            # geometry and reference posture from the library file, so both modes describe the same shelf
            self._lib = OnlineChecker(ShelfGeometry(**lib.meta["geometry"]), ref=lib.meta["ref"], cache=False)
            self._lib.init()
        else:
            raise ValueError("A19V2_CHECKER must be 'library' or 'online', got %r" % mode)

    def check(self, state: xmi.XState, action: xmi.XAction) -> bool:
        """True iff ``action`` (``pick``/``put`` ``?obj ?cel``) is physically INFEASIBLE in ``state``."""
        self.init()
        occupied = frozenset(a.objects[0].name for a in state.fluent_atoms if a.predicate.name == "occupied")
        cell = action.objects[1].name
        key = (occupied - {cell}, cell)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._cache[key] = not self._lib.feasible(cell, key[0])
        return hit


register_domain(DomainSpec(
    name="tabletop-access-v2",          # matches (define (domain tabletop-access-v2))
    feasibility_checker=Access19V2FeasibilityChecker,
    reference_problem_dir="access19",
    eval_suites=(),                     # Access-19 is evaluated by reap.eval.access19_eval
    domain_files=("domain.pddl",),
))
