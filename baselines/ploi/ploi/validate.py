"""IsSolution (PLOI paper Definition 2): replay a plan produced by PLAN on a
*reduced* problem against the *full*, original problem's true initial state.

This is necessary, not just thorough: TabletopFeasibilityChecker derives the
physical scene purely by scanning whichever xmi.XState's occupied/holding
fluent atoms it's handed (reap/domains/tabletop.py, _state_to_robot_state). If
PLAN searched over a reduced object set, the checker never placed excluded
(but really-there) cylinders in the MuJoCo scene during that search, so a
plan that "worked" there might only have worked because a real blocker was
invisible to it. Replaying against the full problem's real initial state
re-checks every step's physical feasibility with the true, complete cylinder
layout -- catching exactly that failure mode and triggering PLOI's outer-loop
retry with a larger object set.

An XAction produced by the reduced problem's own search is a view into that
problem's own PDDLRepositories and cannot be applied directly to a state from
a different XProblem (xmimir ground objects are not portable across
repositories -- see XAtom's own docstring warning). Each plan step is
therefore re-resolved against the full problem's own action space by
(action name, argument object names) before being applied.

Uses ploi.checker.filter_action / ploi.plan_search.next_real rather than
importing reap.rl.av_star.sim/search directly -- see checker.py's
module docstring for why.
"""
from __future__ import annotations

from dataclasses import dataclass

import reap.xmimir as xmi

from ploi.checker import filter_action
from ploi.plan_search import next_real


@dataclass
class ValidationResult:
    solved: bool
    reason: str
    final_state: object | None = None  # xmi.XState


def _match_action(state, generator, action_name: str, arg_names: tuple[str, ...]):
    for a in generator.action_generator.generate_actions(state):
        if a.name == action_name and tuple(o.name for o in a.objects) == arg_names:
            return a
    return None


def is_solution(plan: list, full_problem, full_generator=None) -> ValidationResult:
    """Replay `plan` (a list of xmi.XAction from a REDUCED problem's search)
    against `full_problem` (the original, unreduced xmi.XProblem). Returns
    ValidationResult(solved=True) iff every step is symbolically applicable,
    physically feasible against the full object layout, and the final state
    satisfies the full problem's goal.
    """
    if full_generator is None:
        full_generator = xmi.XSuccessorGenerator(full_problem)
    state = full_generator.initial_state

    for step_i, reduced_action in enumerate(plan):
        action_name = reduced_action.name
        arg_names = tuple(o.name for o in reduced_action.objects)

        full_action = _match_action(state, full_generator, action_name, arg_names)
        if full_action is None:
            return ValidationResult(
                False,
                f"step {step_i}: ({action_name} {' '.join(arg_names)}) not symbolically "
                "applicable against the full problem's state",
                state,
            )
        if filter_action(state, full_action):
            return ValidationResult(
                False,
                f"step {step_i}: ({action_name} {' '.join(arg_names)}) physically "
                "infeasible against the full object layout",
                state,
            )
        state = next_real(full_action.apply(state), full_generator)

    if not state.is_goal():
        return ValidationResult(False, "final state does not satisfy the full problem's goal", state)
    return ValidationResult(True, "solved", state)
