"""IK seed retries and iteration limits of the SAHS feasibility check (the paper's appendix: the check retries IK from
up to three seeds and uses more IK and planner iterations than REAP's).

tampanda's `ActionFeasibilityChecker._check_pick`/`_check_put` (and their copy in `occlusion.py`) call
`env.ik.converge_ik(dt)` once before each stage, while `FeasibilityRRT.plan_to_pose` retries IK from up to three seeds
that move toward the middle of the joint range (tampanda/planners/feasibility_rrt.py,
tampanda/symbolic/domains/tabletop/feasibility.py). `patch_ik_seed_retry` gives `env.ik.converge_ik` of one
environment instance the same retry, wherever it is called; tampanda itself is not modified.
"""
from __future__ import annotations

import mujoco
import numpy as np


def patch_ik_seed_retry(env, max_seed_retries: int = 3) -> None:
    """Idempotent: a second call on the same env does nothing (marker attribute); checker.py and occlusion.py each
    build their own env."""
    if getattr(env.ik, "_seed_retry_patched", False):
        return

    original_converge_ik = env.ik.converge_ik
    joint_limits_low = env.model.jnt_range[:7, 0].copy()
    joint_limits_high = env.model.jnt_range[:7, 1].copy()

    def converge_ik_with_seed_retry(dt: float) -> bool:
        start_qpos = env.data.qpos[:7].copy()
        for attempt in range(max_seed_retries):
            if attempt > 0:
                # as FeasibilityRRT.plan_to_pose: a seed moved further toward the middle of the joint range on each
                # retry
                neutral = (joint_limits_low + joint_limits_high) / 2.0
                alpha = 0.3 * attempt
                fallback = (1.0 - alpha) * start_qpos + alpha * neutral
                env.data.qpos[:7] = fallback
                mujoco.mj_forward(env.model, env.data)
                env.ik.update_configuration(env.data.qpos)

            if original_converge_ik(dt):
                return True

            if attempt < max_seed_retries - 1:
                env.data.qpos[:7] = start_qpos
                mujoco.mj_forward(env.model, env.data)
                env.ik.update_configuration(env.data.qpos)

        # all seeds failed: restore the start configuration
        env.data.qpos[:7] = start_qpos
        mujoco.mj_forward(env.model, env.data)
        env.ik.update_configuration(env.data.qpos)
        return False

    env.ik.converge_ik = converge_ik_with_seed_retry
    env.ik._seed_retry_patched = True


def patched_ik_max_iters() -> int:
    # A failing IK attempt uses all its iterations and the seed retry repeats it up to three times, so the increase
    # is small.
    return 150  # ActionFeasibilityChecker's default: 100


def patched_rrt_max_iterations() -> int:
    return 1200  # FeasibilityRRT/ActionFeasibilityChecker default: 1000
