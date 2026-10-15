"""N2 (occupancy errors) intervention point, plus cache-hit/stale-verdict
instrumentation for the cache-condition measurement.

Subclasses ``reap.domains.tabletop.TabletopFeasibilityChecker``. Two things
are overridden:

- ``init()``: the construction of ``TabletopFeasibilityChecker.init()``,
  with :class:`reap.eval.noise.noisy_state_manager.NoisyStateManager` in
  place of the plain ``StateManager``. This is the N1 (placement-drift)
  threading point, see that module's docstring. ``ActionFeasibilityChecker``
  itself is tampanda's unmodified class.
- ``_state_to_robot_state``: calls the parent implementation, then, for N2,
  corrupts the returned dict's ``cylinders`` map with probability
  ``occupancy_flip_p`` per occupied cylinder ("occupied reported empty").
  The other N2 direction ("empty reported occupied by a nearby object") is
  not implemented: it needs a spare, symbolically unused ``cylinder_N`` id
  (tampanda's state manager has 30 CYLINDER_SPECS slots vs <= 14 objects
  per problem).

``check()`` is also overridden, but ONLY to add cache-hit/miss counting and
optional stale-verdict sampling around the dispatch logic; the pick/put
dispatch block itself (``_compute``) is the one of
``TabletopFeasibilityChecker.check``.
"""
from __future__ import annotations

import atexit
import os
from typing import Optional

import numpy as np
import reap.xmimir as xmi

from reap.domains.tabletop import TabletopFeasibilityChecker
from tampanda import FrankaEnvironment, GraspPlanner, SCENE_SYMBOLIC
from tampanda.planners import CollisionWorkerPool, SpeculativeFeasibilityRRT
from tampanda.symbolic.domains.tabletop import ActionFeasibilityChecker, GridDomain

from reap.eval.noise import config
from reap.eval.noise.noisy_state_manager import NoisyStateManager
from reap.eval.noise.true_pose import TruePoseTable


class NoisyTabletopFeasibilityChecker(TabletopFeasibilityChecker):
    def __init__(
        self,
        true_pose_table: Optional[TruePoseTable] = None,
        occupancy_flip_p: float = 0.0,
        occupancy_rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__()
        # Only consulted once, by init() below, to construct
        # NoisyStateManager. After init() has run, the state manager holds
        # its OWN true_pose_table attribute -- callers that need to change
        # the live table post-construction (execution.py does, to thread a
        # per-rollout TruePoseTable into an already-init()'d checker) must
        # set self._state_manager.true_pose_table directly, not this field.
        self.true_pose_table = true_pose_table
        self.occupancy_flip_p = occupancy_flip_p
        self.occupancy_rng = occupancy_rng
        if occupancy_flip_p > 0.0 and occupancy_rng is None:
            raise ValueError("occupancy_flip_p > 0 requires occupancy_rng (a local np.random.Generator)")
        self.stats = {"hits": 0, "misses": 0, "stale_checked": 0, "stale_disagree": 0}

    def init(self) -> None:
        # The construction of TabletopFeasibilityChecker.init(), with
        # NoisyStateManager in place of the plain StateManager. Every
        # other object (FrankaEnvironment, CollisionWorkerPool,
        # SpeculativeFeasibilityRRT, GridDomain, GraspPlanner,
        # ActionFeasibilityChecker) is the unmodified class.
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
        self._state_manager = NoisyStateManager(self._grid, self._sim, true_pose_table=self.true_pose_table)
        self._grasp_planner = GraspPlanner(table_z=self._grid.table_height)
        self._checker = ActionFeasibilityChecker(
            self._sim, self._feas_planner, self._state_manager, self._grasp_planner,
        )

    def cell_centers(self) -> dict:
        """Expose the grid's cell-id -> (x, y) mapping, for
        TruePoseTable construction. Requires init() to have run."""
        self.init()
        return {cid: info["center"] for cid, info in self._grid.cells.items()}

    def _state_to_robot_state(self, state: xmi.XState) -> dict:
        robot_state = super()._state_to_robot_state(state)
        if self.occupancy_flip_p > 0.0:
            robot_state = self._corrupt_occupancy(robot_state)
        return robot_state

    def _corrupt_occupancy(self, robot_state: dict) -> dict:
        """N2, 'occupied reported empty' direction only (see module
        docstring for the unimplemented other direction)."""
        cylinders = dict(robot_state["cylinders"])
        for cyl_name in list(cylinders.keys()):
            if self.occupancy_rng.random() < self.occupancy_flip_p:
                del cylinders[cyl_name]
        corrupted = dict(robot_state)
        corrupted["cylinders"] = cylinders
        return corrupted

    def _compute(self, action: xmi.XAction, robot_state: dict) -> bool:
        """The dispatch of TabletopFeasibilityChecker.check, factored out so
        check() can wrap it with cache instrumentation."""
        if action.name == "pick":
            feasible, _ = self._checker.check(
                "pick", robot_state, cylinder_name=action.objects[0].name,
            )
            return not feasible
        elif action.name == "put":
            feasible, _ = self._checker.check(
                "put", robot_state,
                cylinder_name=action.objects[0].name,
                target_cell=action.objects[1].name,
            )
            return not feasible
        else:
            return True

    def check(self, state: xmi.XState, action: xmi.XAction) -> bool:
        result, _was_hit = self.check_verbose(state, action)
        return result

    def check_verbose(self, state: xmi.XState, action: xmi.XAction) -> tuple:
        """Same as check(), but also reports whether this was a cache hit --
        the execution loop uses this to decide when to additionally call
        record_stale_sample() on a sampled subset of hits (config.
        STALE_VERDICT_SAMPLE_RATE), for the cache-condition measurement."""
        self.init()
        robot_state = self._state_to_robot_state(state)
        key = self._cache_key(robot_state, action)
        if key in self._cache:
            self.stats["hits"] += 1
            return self._cache[key], True
        self.stats["misses"] += 1
        result = self._compute(action, robot_state)
        self._cache[key] = result
        return result, False

    def fresh_uncached_check(self, state: xmi.XState, action: xmi.XAction) -> bool:
        """Ground truth for the stale-verdict measurement: recompute WITHOUT
        touching or consulting self._cache. Only meaningful in execution-time
        mode (true_pose_table set) -- at planning time the idealized pose is
        deterministic, so a new recheck isn't testing staleness, just RRT*'s
        own internal non-determinism."""
        self.init()
        robot_state = self._state_to_robot_state(state)
        return self._compute(action, robot_state)

    def record_stale_sample(self, state: xmi.XState, action: xmi.XAction, cached_result: bool) -> bool:
        """Call on a sampled subset of execution-time cache HITS (see
        config.STALE_VERDICT_SAMPLE_RATE). Returns True iff the uncached
        recheck AGREES with the cached verdict."""
        fresh = self.fresh_uncached_check(state, action)
        self.stats["stale_checked"] += 1
        agree = fresh == cached_result
        if not agree:
            self.stats["stale_disagree"] += 1
        return agree

    def clear_cache(self) -> None:
        """TabletopFeasibilityChecker has no clear_cache method; this
        subclass adds one. eval_protocol.py likewise clears a checker's
        _cache between problems to isolate per-problem cache-hit stats."""
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)

    def cache_hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total else 0.0

    def stale_verdict_rate(self) -> float:
        checked = self.stats["stale_checked"]
        return self.stats["stale_disagree"] / checked if checked else 0.0
