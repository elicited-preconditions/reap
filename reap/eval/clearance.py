"""Exact arm clearance on a twin scene, and the multilevel-blocks executor adapter (3D-stacking verification).

The twin is built by the environment's own builder with the robot compiled from base_panda_v2.xml (all link collision
meshes get hulls). ``Clearance`` measures arm bodies link1..7 + hand + fingers against every other collidable geom with
exact contact distances (reap.eval.robust_contacts: mid-phase off, box-box pairs by the exact oriented-box distance),
excluding the gripper against the object of the current action (contact by design) and arm/scene pairs already closer
than 2 mm with the arm at home in the start state (structural, e.g. the shoulder in its mounting). Used by
reap.eval.stacking_verify and reap.eval.stacking_plans.
"""
from __future__ import annotations

import functools
import tempfile
from pathlib import Path

import mujoco

ARM = ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")
GRIPPER = {"hand", "left_finger", "right_finger"}
MARGIN = 0.002
RANGE = 0.05            # clearance measured up to 5 cm


def _v2_builder(module):
    """Context: the module's ArmSceneBuilder compiles the robot from base_panda_v2.xml."""
    from tampanda.scenes import ArmSceneBuilder
    from tampanda.symbolic.domains.access19_v2.shelf import ROBOT_XML as V2_XML
    orig = module.ArmSceneBuilder

    class _Ctx:
        def __enter__(self):
            module.ArmSceneBuilder = functools.partial(ArmSceneBuilder, base=V2_XML)

        def __exit__(self, *a):
            module.ArmSceneBuilder = orig
    return _Ctx()


class MultilevelBlocksAdapter:
    name = "multilevel-blocks"
    fixed_lookat = (0.0, 0.50, 0.35)
    cam_params = (1.0, 0, -35)
    cam_default = [0.4, 0.0, 0.3]

    def build(self, checker, prob):
        from tampanda.symbolic.domains.multilevel_blocks.feasibility import _make_executor
        self.chk = checker
        self.env, self.ws, self.cfg = checker._env, checker._ws, checker._cfg
        self.executor = _make_executor(self.env, self.ws, self.cfg, fast=False)
        return self.env

    def restore_initial(self, state):
        from tampanda.symbolic.domains.multilevel_blocks import state as mb_state
        pddl = self.chk._state_to_pddl_dict(state)
        mb_state.restore_state(self.env, self.ws, self.cfg, pddl, on_held="attach", executor=self.executor)

    def exec_action(self, a):
        from tampanda.symbolic.domains.multilevel_blocks.feasibility import dispatch_action
        return bool(dispatch_action(self.executor, a.name, tuple(o.name for o in a.objects)))

    def twin(self):
        from tampanda.symbolic.domains.multilevel_blocks import env_builder as EB
        from tampanda.symbolic.domains.multilevel_blocks import make_multilevel_blocks_builder
        self._tw_scratch = tempfile.TemporaryDirectory(prefix="mb_twin_")
        with _v2_builder(EB):
            builder, _, _ = make_multilevel_blocks_builder(scratch_dir=Path(self._tw_scratch.name), config=self.cfg)
            return builder.build_env(rate=10000.0)


class Clearance:
    """Clearance on the twin with EXACT contact distances (robust_contacts.py): mid-phase off
    (runtime-raised margins are ignored by the mid-phase when a body has > 1 geom), meshes / cylinders from the
    pipeline, box-box pairs by the exact oriented-box distance (the primitive box-box collider reports bogus deep
    penetrations for separated boxes; mj_geomDistance returns spurious 0.0)."""

    def __init__(self, tw, home_q):
        from tampanda.symbolic.domains.access19_v2.shelf import _enable_arm_collisions
        from reap.eval import robust_contacts as RC
        self.RC = RC
        self.m, self.d = tw.model, tw.data; m = self.m
        _enable_arm_collisions(m)
        RC.configure(m)
        ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in ARM}; ids.pop(-1, None)
        self.arm = {g: ids[m.geom_bodyid[g]] for g in range(m.ngeom) if m.geom_bodyid[g] in ids and m.geom_conaffinity[g]}
        for g in self.arm:
            m.geom_margin[g] = RANGE
        self.bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        self.gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or self.bname(g)
        self.structural = {k for k, v in self.pairs(home_q).items() if v < MARGIN}

    def pairs(self, q):
        m, d = self.m, self.d
        d.qpos[:] = q; mujoco.mj_forward(m, d); self.RC.fix_boxbox(m, d); out = {}
        for c in d.contact[:d.ncon]:
            a1, a2 = c.geom1 in self.arm, c.geom2 in self.arm
            if a1 == a2:
                continue
            ag, sg = (c.geom1, c.geom2) if a1 else (c.geom2, c.geom1)
            k = (self.arm[ag], sg)
            out[k] = min(out.get(k, RANGE), float(c.dist))
        return out

    def __call__(self, q, obj):
        """(min clearance, (arm link, scene geom name, body name)) over the measured pairs of this configuration."""
        best = (RANGE, None)
        for (link, sg), dist in self.pairs(q).items():
            body = self.bname(sg)
            if (link in GRIPPER and obj is not None and body == obj) or (link, sg) in self.structural:
                continue
            if dist < best[0]:
                best = (dist, (link, self.gname(sg), body))
        return best
