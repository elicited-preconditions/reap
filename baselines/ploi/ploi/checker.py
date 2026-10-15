"""REAP's feasibility check (``reap.domains.tabletop.TabletopFeasibilityChecker``) and filter dispatch
(``reap.rl.av_star.sim.filter_action``) for PLOI, so that PLOI's searches use the same check as REAP's (the paper's
appendix).

The grid geometry, the initialisation, the conversion of the state atoms into the scene, the action dispatch, the
SIGALRM timeout and the call statistics are copied from ``reap/domains/tabletop.py`` and ``reap/rl/av_star/sim.py``,
not imported; like the SAHS and BT-h baselines, this module builds on ``tampanda``'s ``ActionFeasibilityChecker``.
"""
from __future__ import annotations

import atexit
import os
import signal
from typing import Dict

import reap.xmimir as xmi

from tampanda import FrankaEnvironment, GraspPlanner, SCENE_SYMBOLIC
from tampanda.planners import CollisionWorkerPool, SpeculativeFeasibilityRRT
from tampanda.symbolic.domains.tabletop import (
    ActionFeasibilityChecker,
    GridDomain,
    StateManager,
)

from ploi import config


class TabletopFeasibilityChecker:
    """Physical-feasibility check for the tabletop-with-cylinders domain.

    Per-worker resources (MuJoCo env, RRT pool, grid abstraction) are
    constructed lazily on first :meth:`check` call to keep spawn-IPC
    bootstrap fast. Results are cached on a key derived from the cylinder
    layout + action string, as in
    ``reap.domains.tabletop.TabletopFeasibilityChecker``.
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
            cell_size=config.GRID_CELL_SIZE,
            working_area=config.GRID_WORKING_AREA,
            table_body_name=config.GRID_TABLE_BODY_NAME,
            table_geom_name=config.GRID_TABLE_GEOM_NAME,
            grid_offset_x=config.GRID_OFFSET_X,
            grid_offset_y=config.GRID_OFFSET_Y,
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
        """True iff `action` is physically INfeasible in `state`."""
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
            result = True  # unknown action in this domain -- filter out conservatively

        self._cache[key] = result
        return result

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)


# Per-process singleton, mirroring reap.rl.av_star.sim._CHECKERS -- one
# real MuJoCo scene per worker process, reused across every problem that
# process touches (both the reduced-problem PLAN search and the full-problem
# IsSolution replay share this same instance, since both parse against the
# same domain name).
_CHECKER: TabletopFeasibilityChecker | None = None


def get_checker() -> TabletopFeasibilityChecker:
    global _CHECKER
    if _CHECKER is None:
        _CHECKER = TabletopFeasibilityChecker()
    return _CHECKER


_FILTER_STATS: Dict[str, int] = {"calls": 0, "infeasible": 0, "timeouts": 0}


def reset_filter_stats() -> None:
    _FILTER_STATS["calls"] = 0
    _FILTER_STATS["infeasible"] = 0
    _FILTER_STATS["timeouts"] = 0


def get_filter_stats() -> Dict[str, int]:
    return dict(_FILTER_STATS)


FILTER_TIMEOUT_S = float(os.environ.get("FILTER_TIMEOUT_S", str(config.CHECKER_TIMEOUT_S)))


class _FilterHang(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _FilterHang()


def filter_action(state, action) -> bool:
    """Return True iff `action` is physically infeasible in `state`, with the
    dispatch, timeout and statistics of reap.rl.av_star.sim.filter_action."""
    if os.environ.get("BYPASS_FILTER_ACTION") == "1":
        return False
    _FILTER_STATS["calls"] += 1
    checker = get_checker()
    if FILTER_TIMEOUT_S > 0:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(max(1, int(FILTER_TIMEOUT_S)))
        try:
            result = checker.check(state, action)
        except _FilterHang:
            print(f"[filter_action] TIMEOUT after {FILTER_TIMEOUT_S:.0f}s on "
                  f"{action} -- treating as infeasible", flush=True)
            _FILTER_STATS["timeouts"] += 1
            result = True
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        result = checker.check(state, action)
    if result:
        _FILTER_STATS["infeasible"] += 1
    return result
