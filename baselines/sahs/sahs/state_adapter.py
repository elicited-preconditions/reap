"""XState <-> tampanda "robot_state" dict conversions.

The dict shape (`{"holding": str|None, "gripper_empty": bool,
"cylinders": {name: [cell]}}`) mirrors
reap/domains/tabletop.py::TabletopFeasibilityChecker
._state_to_robot_state exactly (copied, not imported). tampanda's StateManager
rebuilds the MuJoCo scene from scratch on every check() call, driven
entirely by this dict, which is what makes the two extensions below
possible without touching tampanda: hiding cylinders it doesn't know
about, placing only the ones present.
"""
from __future__ import annotations

import reap.xmimir as xmi


def state_key(robot_state: dict) -> tuple:
    """Canonical hashable key for a robot_state dict -- used for cache keys
    and visited-set dedup (checker.py and sahs.py share this one
    convention)."""
    cylinders = frozenset((c, tuple(cells)) for c, cells in robot_state["cylinders"].items())
    return (cylinders, robot_state["holding"])


def state_to_robot_state(state: xmi.XState) -> dict:
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


def with_synthetic_holding(robot_state: dict, cyl: str) -> dict:
    """As-if `cyl` were already picked up: removed from the table, gripper
    holding it. Needed for ManipFree(o,r)/OccludesManip(o1,o2,r) when o is
    still sitting on the table in the real state (true for every initial
    state in this domain, since goals are single-target `(holding target)`
    retrieval, never region-placement)."""
    return {
        "holding": cyl,
        "gripper_empty": False,
        "cylinders": {k: v for k, v in robot_state["cylinders"].items() if k != cyl},
    }


def without_object(robot_state: dict, obj: str) -> dict:
    """Counterfactual: `obj` entirely absent from the scene -- the
    'removed' half of the OccludesPre/OccludesManip counterfactual pair."""
    return without_objects(robot_state, (obj,))


def empty_cells(robot_state: dict, all_cells: list[str]) -> list[str]:
    """All cells not occupied by a cylinder: the regions of `put` (the cells `generate_actions` grounds `put`'s ?cel
    over). In the SAHS problem definition (IJRR 2022, Section 3: "a set of workspace regions R") the regions are part
    of the environment; here they are the grid cells, as in REAP's domain and search."""
    occupied = {cells[0] for cells in robot_state["cylinders"].values()}
    return [c for c in all_cells if c not in occupied]


def without_objects(robot_state: dict, objs) -> dict:
    """Same as without_object, for a set of objects at once -- for the
    'redundant blocker' case (removing several
    true occluders simultaneously restores feasibility, even though removing
    any single one alone does not)."""
    excluded = set(objs)
    return {
        **robot_state,
        "cylinders": {k: v for k, v in robot_state["cylinders"].items() if k not in excluded},
    }
