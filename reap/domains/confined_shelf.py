"""confined-shelf domain: feasibility checker and eval suites.

Confined-shelf rearrangement (Wang ICAPS-2022): a closed cubicle with one open face, N identical cylinders in K colour
groups, reach-in pick/put only.

The PDDL domain name is ``confined-shelf`` (hyphen), per the
``(define (domain confined-shelf))`` line in ``confined_shelf/pddl/domain.pddl``.
Same predicate vocabulary as tabletop/access19 (``(occupied cell cyl)``,
``(holding cyl)``), so the state translation mirrors access19.

Feasibility comes from tampanda's honest grasp library for the confined shelf
(:mod:`tampanda.symbolic.domains.confined_shelf.honest`): a lookup that needs no MuJoCo scene in the worker.
"""
from __future__ import annotations

import os

import reap.xmimir as xmi

from reap.domains import checker_info
from reap.domains.registry import DomainSpec, EvalSuite, register_domain


class ConfinedShelfFeasibilityChecker:
    """Feasibility check via the tampanda confined_shelf honest grasp library.

    Results are cached by ``(frozenset(layout), held, action_tuple)``.
    """

    def __init__(self) -> None:
        self._cache: dict = {}
        # CONFINED_CHECKER must be set explicitly to "honest" (checker_info.require): the honest grasp library (real
        # arm collision meshes, exact per-cell conflict sets; tampanda confined_shelf/honest.py; CONFINED_LIBRARY
        # overrides the library file)
        checker_info.require(checker_info.CONFINED_VAR)
        from tampanda.symbolic.domains.confined_shelf.honest import DEFAULT_LIBRARY, HonestLibrary
        self._honest = HonestLibrary(os.environ.get("CONFINED_LIBRARY") or DEFAULT_LIBRARY)

    def init(self, state: xmi.XState | None = None) -> None:
        """No-op for API parity (the honest library needs no MuJoCo scene)."""

    @staticmethod
    def _state_to_layout_held(state: xmi.XState) -> tuple[dict, str | None]:
        """xmimir XState -> ``({cyl: cell_id}, held_cyl)``.

        Only ``(occupied cell cyl)`` and ``(holding cyl)`` truths feed
        the check; only those are extracted (mirrors access19).
        """
        layout: dict[str, str] = {}
        held: str | None = None
        for atom in state.fluent_atoms:
            pname = atom.predicate.name
            objs = atom.objects
            if pname == "occupied" and len(objs) == 2:
                cell_id, cyl = objs[0].name, objs[1].name
                layout[cyl] = cell_id
            elif pname == "holding" and len(objs) == 1:
                held = objs[0].name
        return layout, held

    @staticmethod
    def _action_to_tuple(action: xmi.XAction) -> tuple:
        """``(action_name, cyl, cell_id)``: pick/put both have signature
        ``(?cyl - cylinder ?cel - cell)`` so ``action.objects`` is (cyl, cell)."""
        return (action.name, *(o.name for o in action.objects))

    def check(self, state: xmi.XState, action: xmi.XAction) -> bool:
        """Returns True if the action is INFEASIBLE (filter_action contract)."""
        layout, held = self._state_to_layout_held(state)
        action_tuple = self._action_to_tuple(action)
        key = (frozenset(layout.items()), held, action_tuple)
        # cached: the evaluation counts unique queries (MP calls) by cache size
        if key not in self._cache:
            self._cache[key] = not self._honest.check_action(layout, held, action_tuple)
        return self._cache[key]


# Eval suites.  Dataset layout: data/confined/{domain.pddl, train/, validation/, eval/, eval_nonmonotone_<k>/}
# eval/ is the in-distribution set (the same 50 problems as validation/); eval_nonmonotone_<k>/ are the
# non-monotone rearrangement sets with k objects.
CONFINED_SHELF_EVAL_SUITES: tuple[EvalSuite, ...] = (
    EvalSuite(label="eval", path="data/confined/eval"),
) + tuple(EvalSuite(label=f"eval_nonmonotone_{k}", path=f"data/confined/eval_nonmonotone_{k}") for k in (10, 12, 14, 16))


register_domain(DomainSpec(
    name="confined-shelf",  # matches (define (domain confined-shelf))
    feasibility_checker=ConfinedShelfFeasibilityChecker,
    reference_problem_dir="confined",
    eval_suites=CONFINED_SHELF_EVAL_SUITES,
    domain_files=("domain.pddl",),
))
