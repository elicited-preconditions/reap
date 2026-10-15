"""Tabletop-with-cylinders domain: feasibility checker and eval suites.

The checker wraps tampanda's :class:`ActionFeasibilityChecker` for the Franka two-arm tabletop env.
"""
from __future__ import annotations

import atexit
import os

import reap.xmimir as xmi

from tampanda import FrankaEnvironment, GraspPlanner, SCENE_SYMBOLIC
from tampanda.planners import CollisionWorkerPool, SpeculativeFeasibilityRRT
from tampanda.symbolic.domains.tabletop import (
    ActionFeasibilityChecker,
    GridDomain,
    StateManager,
)

from reap.domains.registry import DomainSpec, EvalSuite, register_domain
from reap.paths import data_root


class TabletopFeasibilityChecker:
    """Physical-feasibility check for the tabletop-with-cylinders domain.

    Per-worker resources (MuJoCo env, RRT pool, grid abstraction) are
    constructed lazily on first :meth:`check` call to keep spawn-IPC
    bootstrap fast.  Results are cached on a key derived from the
    cylinder layout + action string.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._sim = None
        self._pool = None
        self._feas_planner = None
        self._grid = None
        self._state_manager = None
        self._grasp_planner = None
        self._checker = None
        self._cache: dict = {}

    def init(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._sim = FrankaEnvironment(str(SCENE_SYMBOLIC), rate=200.0)
        self._pool = CollisionWorkerPool(
            str(SCENE_SYMBOLIC),
            n_workers=int(os.environ.get("COLLISION_POOL_WORKERS", "5")),
        )
        self._feas_planner = SpeculativeFeasibilityRRT(
            self._sim, self._pool, batch_size=4, step_size=0.2,
        )
        atexit.register(self._pool.shutdown)
        self._grid = GridDomain(
            model=self._sim.model,
            cell_size=0.04,
            working_area=(0.4, 0.3),
            table_body_name="simple_table",
            table_geom_name="table_surface",
            grid_offset_x=0.05,
            grid_offset_y=0.25,
        )
        self._state_manager = StateManager(self._grid, self._sim)
        self._grasp_planner = GraspPlanner(table_z=self._grid.table_height)
        self._checker = ActionFeasibilityChecker(
            self._sim, self._feas_planner, self._state_manager, self._grasp_planner,
        )

    @staticmethod
    def _state_to_robot_state(state: xmi.XState) -> dict:
        robot_state: dict = {"holding": None, "gripper_empty": True, "cylinders": {}}
        for atom in state.fluent_atoms:
            name = atom.predicate.name
            objs = [o.name for o in atom.objects]
            if name == "occupied":
                cell, cylinder = objs[0], objs[1]
                robot_state["cylinders"][cylinder] = [cell]
            elif name == "holding":
                robot_state["holding"] = objs[0]
                robot_state["gripper_empty"] = False
        return robot_state

    @staticmethod
    def _cache_key(robot_state: dict, action) -> tuple:
        cylinders = frozenset(
            (cyl, tuple(cells)) for cyl, cells in robot_state["cylinders"].items()
        )
        return (cylinders, robot_state["holding"], str(action))

    def check(self, state: xmi.XState, action: xmi.XAction) -> bool:
        self.init()
        robot_state = self._state_to_robot_state(state)
        key = self._cache_key(robot_state, action)
        if key in self._cache:
            return self._cache[key]

        if action.name == "pick":
            feasible, _ = self._checker.check(
                "pick", robot_state, cylinder_name=action.objects[0].name,
            )
            result = not feasible
        elif action.name == "put":
            feasible, _ = self._checker.check(
                "put", robot_state,
                cylinder_name=action.objects[0].name,
                target_cell=action.objects[1].name,
            )
            result = not feasible
        else:
            # Unknown actions in the tabletop domain: filter out conservatively.
            result = True

        self._cache[key] = result
        return result


# The 15-set evaluation suite of the paper ($REAP_DATA/tabletop/eval/<set>, see reap.paths).
TABLETOP_EVAL_SUITES: tuple[EvalSuite, ...] = tuple(
    EvalSuite(label=label, path=str(data_root() / "tabletop" / "eval" / label))
    for label in [
        # object-count scaling
        "obj10",
        "obj12",
        "obj14",
        "obj16",
        "obj18",
        "obj20",
        "obj22",
        "obj24",
        # paired (N, min-plan-len), random placement
        "obj10_plan3",
        "obj12_plan3",
        "obj14_plan5",
        "obj16_plan5",
        # paired (N, min-plan-len), biased with a --target-blockers chain
        "obj10_plan5_biased2",
        "obj12_plan7_biased3",
        "obj14_plan9_biased3",
    ]
)


_TABLETOP_SPEC_KWARGS = dict(
    feasibility_checker=TabletopFeasibilityChecker,
    reference_problem_dir="tabletop",
    eval_suites=TABLETOP_EVAL_SUITES,
    domain_files=("domain.pddl",),
)
# Canonical key = PDDL domain name (what state.problem.domain.name returns).
register_domain(DomainSpec(name="tabletop-manipulation", **_TABLETOP_SPEC_KWARGS))
# Alias = short name used by CLI tooling (eval_single_model --domain,
# data/tabletop/ paths, save_path label prefixes).
register_domain(DomainSpec(name="tabletop", **_TABLETOP_SPEC_KWARGS))
