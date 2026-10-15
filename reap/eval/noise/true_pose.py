"""N1 (placement drift) per-rollout state.

Mechanism: one true-pose table per rollout maps each object to its actual
(x, y). Each entry starts at its cell centre plus an i.i.d. Gaussian offset
with standard deviation sigma. Every time an action is executed, the placed
object's true pose is redrawn as the target cell centre plus a new offset.

This is a MEMORYLESS per-placement redraw, not a compounding random walk:
each redraw is a new i.i.d. sample around the (deterministic) target cell
centre of that action, discarding any prior offset of that object. Drift
accumulates along the executed trajectory only in the sense that more of
the scene carries a drifted pose as more objects get touched.

Uses a LOCAL ``numpy.random.Generator`` (``default_rng``), never the bare
``numpy.random.*`` module-level functions: those share process-global state
with tampanda's own RRT*/FeasibilityRRT sampling (unseeded, global state in
``tampanda/planners/*.py``). A local ``Generator`` instance is a separate
state object, so these draws cannot perturb or be perturbed by RRT*'s
internal randomness.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


def seed_from_problem_id(problem_id: str) -> int:
    """Deterministic, stable across processes/runs (unlike Python's builtin
    ``hash()``, which is salted per-process unless PYTHONHASHSEED is fixed)."""
    digest = hashlib.sha256(problem_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class TruePoseTable:
    """Maps on-table cylinder name -> its actual (x, y), for one rollout.

    Only tracks cylinders currently placed on the table (i.e. present in a
    grounded state's ``cylinders`` dict) -- a held object has no cell-based
    placement to drift (the transport pose is IK-solved, not cell-based, so
    the drift model does not apply while holding).
    """

    def __init__(self, problem_id: str, sigma_m: float, cell_centers: Mapping[str, Tuple[float, float]]):
        self.problem_id = problem_id
        self.sigma_m = sigma_m
        self.cell_centers = cell_centers
        self.rng = np.random.default_rng(seed_from_problem_id(problem_id))
        self._poses: Dict[str, Tuple[float, float]] = {}

    def _draw(self, cell_id: str) -> Tuple[float, float]:
        cx, cy = self.cell_centers[cell_id]
        if self.sigma_m <= 0.0:
            return (cx, cy)
        dx, dy = self.rng.normal(0.0, self.sigma_m, size=2)
        return (cx + float(dx), cy + float(dy))

    def init_from_cylinders(self, cylinders: Mapping[str, list]) -> None:
        """Seed every on-table cylinder's true pose from its initial cell.
        Must be called once at rollout start with the FULL initial cylinder
        set (not just ones the plan will touch) -- other cylinders' true
        positions affect RRT* collision-checking for the acted-on object."""
        for cyl_name, cells in cylinders.items():
            self._poses[cyl_name] = self._draw(cells[0])

    def redraw(self, cyl_name: str, target_cell_id: str) -> Tuple[float, float]:
        """Call on every EXECUTED (not merely hypothesized) placement."""
        pose = self._draw(target_cell_id)
        self._poses[cyl_name] = pose
        return pose

    def drop_from_table(self, cyl_name: str) -> None:
        """Call when a cylinder is picked up -- it's no longer 'on the
        table' until put back down (redraw will re-add it)."""
        self._poses.pop(cyl_name, None)

    def get(self, cyl_name: str) -> Optional[Tuple[float, float]]:
        return self._poses.get(cyl_name)

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        return dict(self._poses)
