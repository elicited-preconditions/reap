"""OccludesPre / OccludesManip as SAHS computes them: plan once while ignoring collisions with the movable objects,
then find the objects the planned path collides with (one planning call per target, instead of one per candidate
blocker as in predicates.py's removal test).

OcclusionChecker builds its own FrankaEnvironment / FeasibilityRRT / GridDomain / StateManager / GraspPlanner,
separate from the ActionFeasibilityChecker of checker.py: ActionFeasibilityChecker._check_pick/_check_put clear the
collision exceptions on several failure branches, which would interfere with the exceptions set here for the pass
that ignores collisions.

_pick_approach_grasp/_put_approach_place follow the stages of tampanda's ActionFeasibilityChecker._check_pick/
_check_put (tampanda/symbolic/domains/tabletop/feasibility.py), whose .check() does not return the planned path, and
_pick_candidate() is a copy of ActionFeasibilityChecker._pick_candidate; they follow tampanda v1.1.0. The transport
stage of a pick (moving to a parking pose after the grasp) is not part of OccludesPre; put has no transport stage.
The IK seed retry and the iteration limits of checker.py (ik_retry.py) apply here too.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from sahs import config, state_adapter
from sahs.ik_retry import patch_ik_seed_retry, patched_ik_max_iters, patched_rrt_max_iterations
from sahs.timeout import run_with_timeout
from sahs.timing import CallStats

# check() failure reasons that can come from an object in the way: the only ones for which the pass that ignores
# collisions is run (no_grasp_candidate, no_place_candidate and *_transport_fail do not depend on other objects)
PICK_OCCLUSION_RELEVANT_REASONS = frozenset(
    {"ik_approach_fail", "rrt_approach_fail", "ik_grasp_fail", "rrt_grasp_fail"}
)
PUT_OCCLUSION_RELEVANT_REASONS = frozenset(
    {"ik_approach_fail", "rrt_approach_fail", "ik_place_fail", "rrt_place_fail"}
)


def _pick_candidate(candidates):
    """As ActionFeasibilityChecker._pick_candidate: the FRONT approach if there is one, else the best-scored
    candidate (GraspPlanner.generate_candidates sorts them best first)."""
    from tampanda.planners.grasp_planner import GraspType

    front = next((c for c in candidates if c.grasp_type == GraspType.FRONT), None)
    return front if front is not None else (candidates[0] if candidates else None)


class OcclusionChecker:
    def __init__(
        self,
        timeout_s: float = config.CHECKER_TIMEOUT_S,
        max_iterations: int = patched_rrt_max_iterations(),
        settle_steps: int = 60,
        ik_max_iters: int = patched_ik_max_iters(),
        ik_pos_threshold: float = 0.005,
    ):
        self._env = None
        self._feas_planner = None
        self._state_manager = None
        self._grasp_planner = None
        self.timeout_s = timeout_s
        self.max_iterations = max_iterations
        self.settle_steps = settle_steps
        self.ik_max_iters = ik_max_iters
        self.ik_pos_threshold = ik_pos_threshold
        self._cache: dict = {}
        self.stats = CallStats()

    def _ensure_init(self) -> None:
        if self._env is not None:
            return
        from tampanda import FrankaEnvironment, FeasibilityRRT, GraspPlanner, SCENE_SYMBOLIC
        from tampanda.symbolic.domains.tabletop import GridDomain, StateManager

        env = FrankaEnvironment(str(SCENE_SYMBOLIC), rate=200.0)
        # Tuned for feasibility checking, matching ActionFeasibilityChecker.__init__.
        env.ik.max_iters = self.ik_max_iters
        env.ik.pos_threshold = self.ik_pos_threshold
        patch_ik_seed_retry(env)
        # Bypass the rate limiter -- headless, matching ActionFeasibilityChecker.__init__.
        _dt = env.model.opt.timestep

        def _fast_step():
            if env._attached is not None:
                env._apply_attachment()
            mujoco.mj_step(env.model, env.data)
            env.sim_time += _dt
            return _dt

        env.step = _fast_step

        self._env = env
        # step_size=0.2, as REAP's SpeculativeFeasibilityRRT (reap/domains/tabletop.py)
        self._feas_planner = FeasibilityRRT(env, step_size=0.2)
        grid = GridDomain(
            model=env.model,
            cell_size=config.GRID_CELL_SIZE,
            working_area=config.GRID_WORKING_AREA,
            table_body_name=config.GRID_TABLE_BODY_NAME,
            table_geom_name=config.GRID_TABLE_GEOM_NAME,
            grid_offset_x=config.GRID_OFFSET_X,
            grid_offset_y=config.GRID_OFFSET_Y,
        )
        self._state_manager = StateManager(grid, env)
        self._grasp_planner = GraspPlanner(table_z=grid.table_height)

    def _init_state(self, state_dict: dict) -> None:
        self._state_manager.set_from_grounded_state(state_dict)
        self._env.controller.stop()
        mujoco.mj_forward(self._env.model, self._env.data)
        self._env.ik.update_configuration(self._env.data.qpos)
        for _ in range(self.settle_steps):
            self._env.step()

    def _wait_idle(self, max_steps: int = 5000) -> bool:
        return self._env.wait_idle(max_steps=max_steps, settle_steps=30)

    def _pick_approach_grasp(self, state_dict: dict, cylinder_name: str, ignore_names):
        """Mirrors ActionFeasibilityChecker._check_pick's stages 1-6 (through
        grasp-contact RRT*) -- approach + grasp only, no transport. Returns
        (feasible, path, reason); path is the concatenated approach+grasp
        waypoint list on success, else None."""
        self._init_state(state_dict)
        self._env.set_collision_exceptions(list(ignore_names))
        dt = self._env.model.opt.timestep

        cyl_pos = self._env.get_object_position(cylinder_name)
        half_size = self._env.get_object_half_size(cylinder_name)
        cyl_quat = self._env.get_object_orientation(cylinder_name)
        candidates = self._grasp_planner.generate_candidates(cyl_pos, half_size, cyl_quat)
        candidate = _pick_candidate(candidates)
        if candidate is None:
            self._env.clear_collision_exceptions()
            return False, None, "no_grasp_candidate"

        self._env.ik.set_target_position(candidate.approach_pos, candidate.grasp_quat)
        if not self._env.ik.converge_ik(dt):
            self._env.clear_collision_exceptions()
            return False, None, "ik_approach_fail"

        approach_path = self._feas_planner.plan_to_pose(
            candidate.approach_pos, candidate.grasp_quat, dt=dt, max_iterations=self.max_iterations,
        )
        if approach_path is None:
            self._env.clear_collision_exceptions()
            return False, None, "rrt_approach_fail"

        self._env.execute_path(approach_path, self._feas_planner)
        self._wait_idle()

        self._env.add_collision_exception(cylinder_name)
        self._env.ik.set_target_position(candidate.grasp_pos, candidate.grasp_quat)
        if not self._env.ik.converge_ik(dt):
            self._env.clear_collision_exceptions()
            return False, None, "ik_grasp_fail"

        grasp_path = self._feas_planner.plan_to_pose(
            candidate.grasp_pos, candidate.grasp_quat, dt=dt, max_iterations=self.max_iterations,
        )
        if grasp_path is None:
            self._env.clear_collision_exceptions()
            return False, None, "rrt_grasp_fail"

        self._env.clear_collision_exceptions()
        return True, list(approach_path) + list(grasp_path), "ok"

    def _put_approach_place(self, state_dict: dict, cylinder_name: str, target_cell: str, ignore_names):
        """Mirrors ActionFeasibilityChecker._check_put's full stages (put has
        no transport stage, so nothing to skip). `state_dict` must already
        have holding=cylinder_name set (same convention as
        checker.check("put", ...) / predicates.py's with_synthetic_holding),
        matching how the real system calls put's checker."""
        from tampanda.planners.grasp_planner import quat_to_rotmat
        from tampanda.symbolic.domains.tabletop.state_manager import StateManager

        cylinders_on_table = {
            k: v for k, v in state_dict.get("cylinders", {}).items() if k != cylinder_name
        }
        put_state = {**state_dict, "cylinders": cylinders_on_table, "holding": cylinder_name}
        self._init_state(put_state)
        self._env.set_collision_exceptions(list(ignore_names))

        def _cleanup():
            self._env.clear_collision_held_body()
            self._env.detach_object()
            self._env.clear_collision_exceptions()

        cyl_idx = int(cylinder_name.split("_")[1])
        radius, half_height = StateManager.CYLINDER_SPECS[cyl_idx]
        half_size = np.array([radius, radius, half_height])
        dt = self._env.model.opt.timestep

        grid = self._state_manager.grid
        cx, cy = grid.cells[target_cell]["center"]
        cyl_z = grid.table_height + half_height + 0.002
        target_pos = np.array([cx, cy, cyl_z])

        candidates = self._grasp_planner.generate_candidates(target_pos, half_size)
        candidate = _pick_candidate(candidates)
        if candidate is None:
            _cleanup()
            return False, None, "no_place_candidate"

        put_quat = candidate.grasp_quat
        R_put = quat_to_rotmat(put_quat)
        rel_pos_ee = self._env._attached["rel_pos"]
        place_ee_pos = target_pos - R_put @ rel_pos_ee
        approach_pos = place_ee_pos + np.array([0.0, 0.0, self._grasp_planner.approach_dist])

        self._env.ik.set_target_position(approach_pos, put_quat)
        if not self._env.ik.converge_ik(dt):
            _cleanup()
            return False, None, "ik_approach_fail"

        approach_path = self._feas_planner.plan_to_pose(
            approach_pos, put_quat, dt=dt, max_iterations=self.max_iterations,
        )
        if approach_path is None:
            _cleanup()
            return False, None, "rrt_approach_fail"

        self._env.execute_path(approach_path, self._feas_planner)
        self._wait_idle()

        self._env.ik.set_target_position(place_ee_pos, put_quat)
        if not self._env.ik.converge_ik(dt):
            _cleanup()
            return False, None, "ik_place_fail"

        place_path = self._feas_planner.plan_to_pose(
            place_ee_pos, put_quat, dt=dt, max_iterations=self.max_iterations,
        )
        if place_path is None:
            _cleanup()
            return False, None, "rrt_place_fail"

        _cleanup()
        return True, list(approach_path) + list(place_path), "ok"

    def _bodies_in_contact(self, candidate_names: set[str]) -> set[str]:
        """Only counts robot-vs-candidate contact pairs (mirrors
        FrankaEnvironment.check_collisions()'s b1_is_robot != b2_is_robot
        filter) -- NOT any pair involving a candidate, which would also
        pick up normal, always-present object-resting-on-table contacts
        that have nothing to do with the arm's path."""
        env = self._env
        robot_ids = env._collision_body_ids
        names: set[str] = set()
        for i in range(env.data.ncon):
            contact = env.data.contact[i]
            body1 = env.model.geom_bodyid[contact.geom1]
            body2 = env.model.geom_bodyid[contact.geom2]
            b1_is_robot = body1 in robot_ids
            b2_is_robot = body2 in robot_ids
            if b1_is_robot == b2_is_robot:
                continue
            other_id = body2 if b1_is_robot else body1
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, other_id)
            if name in candidate_names:
                names.add(name)
        return names

    def _walk_path_for_collisions(self, path, candidate_names: set[str]) -> set[str]:
        """Re-check every path waypoint (subdivided at the same granularity
        RRT* itself used internally) with ALL objects live (no exceptions),
        union which candidate objects show up in contact. Correct regardless
        of whether the target/held cylinder's own legitimate grasp/place
        contact also registers as "a collision" here -- we only ever keep
        names in candidate_names, which never includes the target/held
        object itself."""
        env = self._env
        steps = getattr(self._feas_planner, "collision_check_steps", 5)
        saved_qpos = env.data.qpos.copy()
        saved_qvel = env.data.qvel.copy()
        colliding: set[str] = set()
        try:
            for a, b in zip(path[:-1], path[1:]):
                for frac in np.linspace(0.0, 1.0, steps + 1):
                    waypoint = a + frac * (b - a)
                    if not env.is_collision_free_no_restore(waypoint):
                        colliding |= self._bodies_in_contact(candidate_names)
        finally:
            env.data.qpos[:] = saved_qpos
            env.data.qvel[:] = saved_qvel
            mujoco.mj_forward(env.model, env.data)
        return colliding

    def occludes_pre_row(
        self, base_robot_state: dict, target: str, candidates: list[str], baseline_reason: str,
    ) -> dict[str, bool]:
        if baseline_reason not in PICK_OCCLUSION_RELEVANT_REASONS:
            return {o1: False for o1 in candidates}

        self._ensure_init()
        cache_key = (state_adapter.state_key(base_robot_state), "pick", target, frozenset(candidates))
        if cache_key in self._cache:
            self.stats.record_hit("OccludesPre_plan")
            return self._cache[cache_key]

        t0 = time.perf_counter()
        feasible, path, _reason = run_with_timeout(
            lambda: self._pick_approach_grasp(base_robot_state, target, candidates),
            self.timeout_s,
            lambda: (False, None, f"adapter_timeout_{self.timeout_s}s"),
        )
        self.stats.record_miss("OccludesPre_plan", time.perf_counter() - t0)

        if not feasible:
            result = {o1: False for o1 in candidates}
        else:
            t1 = time.perf_counter()
            colliding = self._walk_path_for_collisions(path, set(candidates))
            self.stats.record_miss("OccludesPre_probe", time.perf_counter() - t1)
            result = {o1: (o1 in colliding) for o1 in candidates}

        self._cache[cache_key] = result
        return result

    def occludes_manip_row(
        self,
        base_robot_state: dict,
        held_object: str,
        target_cell: str,
        candidates: list[str],
        baseline_reason: str,
    ) -> dict[str, bool]:
        if baseline_reason not in PUT_OCCLUSION_RELEVANT_REASONS:
            return {o1: False for o1 in candidates}

        self._ensure_init()
        cache_key = (
            state_adapter.state_key(base_robot_state), "put", held_object, target_cell, frozenset(candidates),
        )
        if cache_key in self._cache:
            self.stats.record_hit("OccludesManip_plan")
            return self._cache[cache_key]

        t0 = time.perf_counter()
        feasible, path, _reason = run_with_timeout(
            lambda: self._put_approach_place(base_robot_state, held_object, target_cell, candidates),
            self.timeout_s,
            lambda: (False, None, f"adapter_timeout_{self.timeout_s}s"),
        )
        self.stats.record_miss("OccludesManip_plan", time.perf_counter() - t0)

        if not feasible:
            result = {o1: False for o1 in candidates}
        else:
            t1 = time.perf_counter()
            colliding = self._walk_path_for_collisions(path, set(candidates))
            self.stats.record_miss("OccludesManip_probe", time.perf_counter() - t1)
            result = {o1: (o1 in colliding) for o1 in candidates}

        self._cache[cache_key] = result
        return result
