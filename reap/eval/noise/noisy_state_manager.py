"""N1 (placement drift) intervention point: :class:`NoisyStateManager`.

Subclasses tampanda's ``StateManager`` and overrides exactly one method,
``set_from_grounded_state``. The parent method places every on-table
cylinder at its idealized cell centroid via
``self._set_cylinder_position(cyl_idx, x, y, z)``. This override calls the
parent placement logic first (unmodified: hide-all, centroid placement, arm
reset, IK-based holding transport), then re-applies the rollout's true
(drifted) x/y for every present cylinder, keeping the z the parent method
computed (table_height + cylinder height + 0.002, from ``CYLINDER_SPECS``).

Both ``pick``'s feasibility check (reads back the acted-on cylinder's
physical position after ``_init_state``) and ``put``'s RRT* obstacle
avoidance (reads the OTHER cylinders' physical positions during planning)
go through this placement path, so this single override covers every case
noise model N1 needs to touch.

Construct with ``true_pose_table=None`` for REAP's idealized behavior
(every *planning-time* search query); construct with a live
:class:`reap.eval.noise.true_pose.TruePoseTable` for *execution-time*
queries that must see accumulated drift.
"""
from __future__ import annotations

from typing import Optional

from tampanda.symbolic.domains.tabletop import StateManager

from reap.eval.noise.true_pose import TruePoseTable


class NoisyStateManager(StateManager):
    def __init__(self, grid_domain, env, true_pose_table: Optional[TruePoseTable] = None):
        super().__init__(grid_domain, env)
        self.true_pose_table = true_pose_table

    def set_from_grounded_state(self, state: dict) -> None:
        super().set_from_grounded_state(state)
        if self.true_pose_table is None:
            return
        for cyl_name in state.get("cylinders", {}):
            pose = self.true_pose_table.get(cyl_name)
            if pose is None:
                # Not yet seeded in the true-pose table (shouldn't happen if
                # init_from_cylinders was called with the FULL initial
                # cylinder set -- fail loudly rather than silently using the
                # idealized centroid, which would hide a real bug).
                raise KeyError(
                    f"{cyl_name} present in grounded state but missing from "
                    "true_pose_table -- was init_from_cylinders() called with "
                    "the complete initial cylinder set for this rollout?"
                )
            x, y = pose
            cyl_idx = int(cyl_name.split("_")[1])
            _, height = self.CYLINDER_SPECS[cyl_idx]
            z = self.grid.table_height + height + 0.002
            self._set_cylinder_position(cyl_idx, x, y, z)
