"""Controllers c_omega: run_pick / run_relocate map (state, theta) ->
(next_state | None, success, info). Both wrap
InstrumentedFeasibilityChecker.check() exactly as-is
-- no tampanda changes are needed; ActionFeasibilityChecker.check
already accepts precisely the two calls needed.

Each internally retries the SAME already-committed call up to
config.MAX_PICK_RETRIES/MAX_PUT_RETRIES times (RRT*/IK have no seeding
hook, so identical calls can genuinely disagree -- see checker.py's
force_refresh) before finally reporting infeasible. This is a different,
orthogonal thing from a SAMPLER's decision not to re-propose a theta a
search node already rejected (sampler.py's `exclude` / without-replacement
semantics) -- one is about noise-robustness for a single committed choice,
the other is about not wasting search budget replaying rejected choices.
"""
from __future__ import annotations

from copy import deepcopy

from bt_h import config
from bt_h.checker import InstrumentedFeasibilityChecker


def _retry_check(
    checker: InstrumentedFeasibilityChecker,
    robot_state: dict,
    action: str,
    cyl: str,
    target_cell: str | None,
    max_retries: int,
    call_label: str,
) -> tuple[bool, dict]:
    feasible, info = checker.check(robot_state, action, cyl, target_cell=target_cell, call_label=call_label)
    attempt = 1
    while not feasible and attempt < max_retries:
        feasible, info = checker.check(
            robot_state, action, cyl, target_cell=target_cell,
            call_label=call_label, force_refresh=True,
        )
        attempt += 1
    return feasible, info


def run_pick(
    s_prev: dict, cyl: str, checker: InstrumentedFeasibilityChecker
) -> tuple[dict | None, bool, dict]:
    feasible, info = _retry_check(checker, s_prev, "pick", cyl, None, config.MAX_PICK_RETRIES, "pick")
    if not feasible:
        return None, False, info
    s_next = deepcopy(s_prev)
    del s_next["cylinders"][cyl]
    s_next["holding"] = cyl
    s_next["gripper_empty"] = False
    return s_next, True, info


def run_relocate(
    s_prev: dict, cyl: str, target_cell: str, checker: InstrumentedFeasibilityChecker
) -> tuple[dict | None, bool, dict]:
    feasible, info = _retry_check(checker, s_prev, "put", cyl, target_cell, config.MAX_PUT_RETRIES, "relocate")
    if not feasible:
        return None, False, info
    s_next = deepcopy(s_prev)
    s_next["cylinders"][cyl] = [target_cell]
    s_next["holding"] = None
    s_next["gripper_empty"] = True
    return s_next, True, info
