"""shelf-access domain: multi-level shelf retrieval (Bouhsain HAL 2025).

Task: retrieve an object-of-interest (``ooi``) from a free-standing 3-tier
shelf past blockers and deliver it to the goal cell.

The PDDL domain name is ``shelf-access``, distinct from tampanda's native
``tabletop-access`` name.  Feasibility comes from tampanda's shelf grasp
library (:mod:`tampanda.symbolic.domains.shelf_access_v2`), which takes
``(kind, obj, cell)`` tuples and does not depend on the domain name.

Same predicate vocab as tabletop/access19 (``(occupied ?cel ?obj)``,
``(holding ?obj)``), plus a structural ``(adjacent ?dir ?c1 ?c2)`` used only
by the GNN encoder.  Plain STRIPS ``pick``/``put``, no derived variant.

Fixed roster (``ooi`` + 10 blockers); per problem only a subset is present
(2-11 movables).
"""
from __future__ import annotations

import os

import reap.xmimir as xmi

from reap.domains import checker_info
from reap.domains.registry import DomainSpec, EvalSuite, register_domain


# Fixed object roster, as in tabletop_access.generate_data
# (_N_BLOCKERS=10, _OBJECT_IDS = ["ooi"] + blocker_0..9).
_N_BLOCKERS = 10
_OBJECT_IDS: list[str] = ["ooi"] + [f"blocker_{i}" for i in range(_N_BLOCKERS)]


class ShelfAccessFeasibilityChecker:
    """Feasibility check via the tampanda shelf_access_v2 grasp library.

    Cached by ``(frozenset(layout), held, action_tuple)``.
    """

    def __init__(self) -> None:
        self._cache: dict = {}
        # SHELF_ACCESS_CHECKER must be set explicitly to "v2" (checker_info.require): the shelf of tampanda
        # shelf_access_v2 (multi-level shelf geometry, exact grasp library with the real arm meshes);
        # SHELF_ACCESS_LIBRARY overrides the library file
        checker_info.require(checker_info.SHELF_VAR)
        from tampanda.symbolic.domains.shelf_access_v2 import DEFAULT_LIBRARY, HonestLibrary
        self._v2 = HonestLibrary(os.environ.get("SHELF_ACCESS_LIBRARY") or DEFAULT_LIBRARY)

    def init(self, state: xmi.XState | None = None) -> None:
        """No-op for API parity (the grasp library needs no MuJoCo scene)."""

    @staticmethod
    def _state_to_layout_held(state: xmi.XState) -> tuple[dict, str | None]:
        """xmimir XState -> ``({obj: cell_id}, held_obj)``.

        Only ``(occupied ?cel ?obj)`` and ``(holding ?obj)`` truths drive
        the check; only those are extracted (mirrors access19).
        """
        layout: dict[str, str] = {}
        held: str | None = None
        for atom in state.fluent_atoms:
            pname = atom.predicate.name
            objs = atom.objects
            if pname == "occupied" and len(objs) == 2:
                cell_id, obj = objs[0].name, objs[1].name
                layout[obj] = cell_id
            elif pname == "holding" and len(objs) == 1:
                held = objs[0].name
        return layout, held

    @staticmethod
    def _action_to_tuple(action: xmi.XAction) -> tuple:
        """``(kind, obj, cell)``: pick/put both ``(?obj ?cel)`` so
        ``action.objects`` is (obj, cell), matching check_action's tuple."""
        return (action.name, *(o.name for o in action.objects))

    def check(self, state: xmi.XState, action: xmi.XAction) -> bool:
        """Returns True if the action is INFEASIBLE (filter_action contract)."""
        layout, held = self._state_to_layout_held(state)
        action_tuple = self._action_to_tuple(action)
        # cached: the evaluation counts unique queries (MP calls) by cache size
        key = (frozenset(layout.items()), held, action_tuple)
        if key not in self._cache:
            self._cache[key] = not self._v2.check_action(layout, held, action_tuple)
        return self._cache[key]


# Eval suites.  Dataset layout: data/multilevel_shelf/{domain.pddl, train/, validation/, eval/, eval_ood/}
# eval/ is the in-distribution set; eval_ood/ the object-count-scaling set.
SHELF_ACCESS_EVAL_SUITES: tuple[EvalSuite, ...] = (
    EvalSuite(label="eval", path="data/multilevel_shelf/eval"),
    EvalSuite(label="eval_ood", path="data/multilevel_shelf/eval_ood"),
)


register_domain(DomainSpec(
    name="shelf-access",  # tampanda's native name is tabletop-access
    feasibility_checker=ShelfAccessFeasibilityChecker,
    reference_problem_dir="multilevel_shelf",
    eval_suites=SHELF_ACCESS_EVAL_SUITES,
    domain_files=("domain.pddl",),
))
