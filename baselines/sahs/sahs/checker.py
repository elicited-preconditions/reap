"""InstrumentedFeasibilityChecker: tampanda's ActionFeasibilityChecker, built lazily in one process (no
CollisionWorkerPool), with a cache, call statistics and a per-call timeout.

The environment, the grid and the planner step (step_size=0.2; tampanda's FeasibilityRRT default is 0.1) are as in
REAP's TabletopFeasibilityChecker.init() (reap/domains/tabletop.py), with a single-process FeasibilityRRT in place of
the speculative parallel RRT. As the paper's appendix states, the SAHS check retries IK from up to three seeds
(ik_retry.patch_ik_seed_retry) and uses more IK and planner iterations than REAP's (ik_max_iters 150 instead of 100,
RRT max_iterations 1200 instead of 1000).
"""
from __future__ import annotations

import time

from sahs import config, state_adapter
from sahs.ik_retry import patch_ik_seed_retry, patched_ik_max_iters, patched_rrt_max_iterations
from sahs.timeout import run_with_timeout
from sahs.timing import CallStats


class InstrumentedFeasibilityChecker:
    def __init__(self, timeout_s: float = config.CHECKER_TIMEOUT_S):
        self._checker = None
        self._cache: dict = {}
        self.stats = CallStats()
        self.timeout_s = timeout_s

    def _ensure_init(self) -> None:
        if self._checker is not None:
            return
        from tampanda import FrankaEnvironment, FeasibilityRRT, GraspPlanner, SCENE_SYMBOLIC
        from tampanda.symbolic.domains.tabletop import (
            ActionFeasibilityChecker,
            GridDomain,
            StateManager,
        )

        env = FrankaEnvironment(str(SCENE_SYMBOLIC), rate=200.0)
        patch_ik_seed_retry(env)
        feas_planner = FeasibilityRRT(env, step_size=0.2)
        grid = GridDomain(
            model=env.model,
            cell_size=config.GRID_CELL_SIZE,
            working_area=config.GRID_WORKING_AREA,
            table_body_name=config.GRID_TABLE_BODY_NAME,
            table_geom_name=config.GRID_TABLE_GEOM_NAME,
            grid_offset_x=config.GRID_OFFSET_X,
            grid_offset_y=config.GRID_OFFSET_Y,
        )
        state_manager = StateManager(grid, env)
        grasp_planner = GraspPlanner(table_z=grid.table_height)
        self._checker = ActionFeasibilityChecker(
            env, feas_planner, state_manager, grasp_planner,
            max_iterations=patched_rrt_max_iterations(), ik_max_iters=patched_ik_max_iters(),
        )

    @staticmethod
    def _cache_key(robot_state: dict, action: str, cyl: str, target_cell: str | None) -> tuple:
        return (state_adapter.state_key(robot_state), action, cyl, target_cell)

    def check(
        self,
        robot_state: dict,
        action: str,
        cyl: str,
        target_cell: str | None = None,
        predicate_label: str = "",
        force_refresh: bool = False,
    ) -> tuple[bool, dict]:
        """force_refresh=True bypasses AND overwrites the cache entry --
        needed by callers that must re-sample tampanda's stochastic RRT*
        (e.g. SAHS retrying a failed edge); a cached-only retry would
        silently return the same stale result."""
        self._ensure_init()
        key = self._cache_key(robot_state, action, cyl, target_cell)
        if not force_refresh and key in self._cache:
            self.stats.record_hit(predicate_label)
            return self._cache[key]

        t0 = time.perf_counter()
        feasible, info = run_with_timeout(
            lambda: self._checker.check(
                action, robot_state, cylinder_name=cyl, target_cell=target_cell,
            ),
            self.timeout_s,
            lambda: (False, {"reason": f"adapter_timeout_{self.timeout_s}s"}),
        )

        wall_s = time.perf_counter() - t0
        self.stats.record_miss(predicate_label, wall_s)
        self._cache[key] = (feasible, info)
        return feasible, info
