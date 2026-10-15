"""Parse REAP's reference `.plan` files into a plan ordering pi for
classic_backtrack/bt_h_search ("given the plan ordering" in Table 1), in
place of plan_task()'s symbolic search. Only the order in which the plan
moves the cylinders is taken; its destination cells are discarded, so the
searches choose their own relocate cells with the relocate_thetas sampler.

The reference plans of data/tabletop/ have this shape, and any other shape
raises ValueError:
  - every plan starts and ends with a bare `pick`
  - pick and put alternate
  - a pick's cell argument follows from the state, so it is discarded
  - the final pick's cylinder is the plan's `; Target:` cylinder
  - a plan can move the same cylinder more than once, so the visit order
    is not deduplicated.
"""
from __future__ import annotations

import re
from pathlib import Path

from bt_h.types import GroundAction

PICK_PUT_RE = re.compile(r"\((pick|put|drop)\s+(\S+)\s+(\S+)\)")


def reap_plan_to_pi(plan_path: Path) -> list[GroundAction]:
    """Parse a REAP `.pddl.plan` file into pi = [pick(cyl), relocate(cyl),
    ..., pick(target)] in the cylinder-visit order of REAP's reference plan.
    Raises ValueError if the plan does not have the shape described above."""
    lines = [
        m for m in (PICK_PUT_RE.match(line.strip()) for line in plan_path.read_text().splitlines())
        if m is not None
    ]
    if not lines:
        raise ValueError(f"{plan_path}: no pick/put/drop lines found")

    verbs = ["put" if m.group(1) == "drop" else m.group(1) for m in lines]
    cyls = [m.group(2) for m in lines]

    if verbs[0] != "pick" or verbs[-1] != "pick":
        raise ValueError(f"{plan_path}: expected plan to start and end with pick, got {verbs}")
    expected = (["pick", "put"] * (len(verbs) // 2 + 1))[: len(verbs)]
    if verbs != expected:
        raise ValueError(f"{plan_path}: expected strict pick/put alternation, got {verbs}")

    picks_in_order = cyls[0::2]  # every pick's cylinder, in plan order -- NOT deduplicated

    pi: list[GroundAction] = []
    for cyl in picks_in_order[:-1]:
        pi.append(GroundAction("pick", cyl))
        pi.append(GroundAction("relocate", cyl))
    pi.append(GroundAction("pick", picks_in_order[-1]))  # bare pick(target), no trailing relocate
    return pi
