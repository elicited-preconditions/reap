"""Concrete Python types for the paper's sigma/pi/theta/omega vocabulary
shared across task_planner.py, controller.py,
backtracking.py, and bt_h.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional


class GroundAction(NamedTuple):
    """One ground abstract action omega in the task plan pi.

    kind: "pick" or "relocate".
    cyl:  the cylinder this action acts on.
    cel:  for "pick", the cylinder's cell as bound by the task planner
          (informational only -- the controller re-derives the cylinder's
          real position from the continuous state, never trusts this for
          execution). None for "relocate", whose target cell is exactly the
          thing left unresolved at the symbolic level (see
          domain/relocate_domain.pddl) -- theta for a relocate step IS the
          missing cell.
    """

    kind: str
    cyl: str
    cel: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.kind}({self.cyl})" if self.cel is None else f"{self.kind}({self.cyl},{self.cel})"


@dataclass
class SearchResult:
    """Output of both classic_backtrack (Algorithm 1) and bt_h_search
    (Algorithm 2), so both are evaluated the same way.

    solved:             whether the full task plan pi was refined.
    sigma:               the resulting state sequence (sigma_0..sigma_n), or
                          None if not solved.
    thetas:               theta chosen at each step, parallel to pi, or None
                          if not solved.
    checker_calls:        real ActionFeasibilityChecker.check() calls
                          incurred by this search (delta against the shared
                          checker's cumulative stats -- see SearchBudget).
    wall_time_s:          wall-clock time for this search only.
    termination_reason:   "solved" | "max_budget" | "timeout" | "exhausted"
                          (REAP's termination-reason convention).
    """

    solved: bool
    sigma: Optional[list] = None
    thetas: Optional[list] = None
    checker_calls: int = 0
    wall_time_s: float = 0.0
    termination_reason: str = "exhausted"
