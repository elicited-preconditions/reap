"""The real continuous state sigma: a plain robot_state dict, exactly the
shape StateManager.ground_state()/ActionFeasibilityChecker.check() expect --
{"cylinders": {name: [cell]}, "gripper_empty": bool, "holding": name|None}.

Maintained entirely in Python, never round-tripped through
xmimir/PDDL: relocate_domain.pddl's abstract layer
deliberately forgets a relocated cylinder's cell (no `occupied` atom is ever
asserted for it again), but the REAL world still has that cylinder sitting
somewhere -- so a "parked" cylinder stays in this dict's `cylinders` map at
its real, controller-chosen cell forever. That's what makes later
relocate/pick controller calls see the true occupied geometry (including
cylinders the abstract layer no longer tracks), and it's precisely the
mechanism behind the long-horizon dependency BT-h has to handle.
"""
from __future__ import annotations

from bt_h.grid import all_cells as _all_cells


def initial_state(cylinder_cells: dict[str, str]) -> dict:
    """cylinder_cells: cylinder_name -> cell_name, e.g. Instance.cylinder_cells."""
    return {
        "cylinders": {cyl: [cell] for cyl, cell in cylinder_cells.items()},
        "gripper_empty": True,
        "holding": None,
    }


def state_key(robot_state: dict) -> tuple:
    """Canonical hashable key -- used for checker cache keys (checker.py)
    and backtracking visited-set dedup (backtracking.py/bt_h.py). Same
    convention as the SAHS baseline's state_adapter.state_key."""
    cylinders = frozenset((c, tuple(cells)) for c, cells in robot_state["cylinders"].items())
    return (cylinders, robot_state["holding"])


def free_cells(robot_state: dict, cells: list[str] | None = None) -> list[str]:
    """Cells not currently occupied by any on-table OR parked cylinder --
    parked cylinders remain in robot_state['cylinders'] (see module
    docstring), so they are correctly excluded here even though the
    abstract PDDL layer has already forgotten about them."""
    if cells is None:
        cells = _all_cells()
    occupied = {cell for cell_list in robot_state["cylinders"].values() for cell in cell_list}
    return [c for c in cells if c not in occupied]
