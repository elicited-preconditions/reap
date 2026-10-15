"""PLAN wrapper: bfs_with_plan / gbfs_hff_with_plan.

REAP's reap/rl/av_star/search.py::bfs() and ::gbfs_hff() are the BrFS and
hFF rows of Table 1; both
call filter_action() inline before applying each candidate action
when filter_actions=True, giving physically-grounded search out of the box.
But neither retains a reconstructable plan: both use a bare deque/heap of
(parent, action, distance) tuples and a plain `closed: set()`, so their
return value is only (expansion_count, plan_length, solved_bool, reason,
init_time).

This module reproduces the ~40-line outer loop of each, adding one
`came_from` dict populated exactly where the original code does
`closed.add(node)`, then reconstructing the plan by walking `came_from`
from the goal back to the initial state on success.

Copied, not imported from `reap.rl.av_star.search`/`sim`: `filter_action`
is `ploi.checker.filter_action` (see checker.py); `PrioritizedItem`,
`_hff_value`, `_termination_reason` and `next_real` are copied from
search.py (`next_real` is a no-op in the tabletop domain and kept so the
loop matches search.py).
"""
from __future__ import annotations

import contextlib
import heapq
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pymimir as pm
import reap.xmimir as xmi

from ploi.checker import filter_action


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


def next_real(s, gen):
    # Copied from reap/rl/av_star/search.py::next_real -- pushworld-specific
    # passthrough, a no-op for the tabletop domain.
    if "pushworld" not in gen.problem.domain.name:
        return s
    actions = list(gen.action_generator.generate_actions(s))
    if len(actions) >= 1 and (actions[0].name in ("ppush", "ppush-done")):
        return next_real(actions[0].apply(s), gen)
    return s


def _hff_value(ff, pm_problem, xstate) -> float:
    # FFHeuristic.compute_value wants a wrapped pm.State, not the raw
    # `advanced` state xmi.XState.base exposes. Returns a large finite value
    # for dead-ends (float('inf')) so heapq comparisons stay well-defined.
    wrapped = pm.State(xstate.base, pm_problem)
    h = ff.compute_value(wrapped)
    if h == float("inf"):
        return 1e18
    return float(h)


def _termination_reason(solved, frontier_empty, counter, max_expansion, start_time, time_limit_s) -> str:
    if solved:
        return "solved"
    if frontier_empty:
        return "dead_end"
    if counter >= max_expansion:
        return "max_budget"
    if time_limit_s > 0 and (time.time() - start_time) >= time_limit_s:
        return "timeout"
    return "max_budget"


@dataclass
class PlanResult:
    solved: bool
    plan: list  # list[xmi.XAction], None unless solved
    expanded: int
    reason: str
    init_time: float
    goal_state: object | None = None  # xmi.XState, for validate.py callers that want it


def _reconstruct(came_from: dict, initial_state, goal_state) -> list:
    plan = []
    cur = goal_state
    while cur != initial_state:
        parent, action = came_from[cur]
        plan.append(action)
        cur = parent
    plan.reverse()
    return plan


def _self_check(generator, initial_state, plan: list, goal_state) -> None:
    """Replaying the reconstructed plan from initial_state must reach
    exactly `goal_state` -- guards the came_from-walk invariant, which
    depends on BFS/GBFS's first-discovery-wins property."""
    s = initial_state
    for a in plan:
        s = next_real(a.apply(s), generator)
    assert s == goal_state, (
        "plan reconstruction did not replay to the recorded goal state -- "
        "came_from walk invariant violated"
    )


def bfs_with_plan(
    generator,
    initial_state,
    max_expansion: int = 1000,
    time_limit_s: float = 0,
    filter_actions: bool = True,
) -> PlanResult:
    """REAP's own BrFS baseline, with the actual plan reconstructed."""
    init_start = time.time()
    closed = {initial_state}
    came_from: dict = {}
    frontier = deque([(initial_state, None, 0)])
    init_time = time.time() - init_start
    start_time = time.time()

    counter = 0
    goal = None
    while (
        frontier
        and counter < max_expansion
        and (time.time() - start_time < time_limit_s or time_limit_s <= 0)
    ):
        parent, action, distance = frontier.popleft()
        node = parent
        if action is not None:
            if filter_actions and filter_action(parent, action):
                continue
            node = next_real(action.apply(parent), generator)
            if node in closed:
                continue
            closed.add(node)
            came_from[node] = (parent, action)
            if node.is_goal():
                goal = node
                break

        actions = list(generator.action_generator.generate_actions(node))
        actions.sort(key=lambda a: str(a))
        counter += 1
        for a in actions:
            frontier.append((node, a, distance + 1))

    reason = _termination_reason(
        goal is not None, not frontier, counter, max_expansion, start_time, time_limit_s
    )
    plan = None
    if goal is not None:
        plan = _reconstruct(came_from, initial_state, goal)
        _self_check(generator, initial_state, plan, goal)
    return PlanResult(goal is not None, plan, counter, reason, init_time, goal)


def gbfs_hff_with_plan(
    generator,
    initial_state,
    max_expansion: int = 1000,
    time_limit_s: float = 0,
    filter_actions: bool = True,
) -> PlanResult:
    """REAP's own hFF baseline, with the actual plan reconstructed."""
    init_start = time.time()
    pm_problem = generator.problem.base
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        ff = pm.FFHeuristic(pm_problem)

    closed = {initial_state}
    came_from: dict = {}
    open_heap = [PrioritizedItem(_hff_value(ff, pm_problem, initial_state), (initial_state, None, 0))]
    init_time = time.time() - init_start
    start_time = time.time()

    counter = 0
    goal = None
    while (
        open_heap
        and counter < max_expansion
        and (time.time() - start_time < time_limit_s or time_limit_s <= 0)
    ):
        parent, action, distance = heapq.heappop(open_heap).item
        node = parent
        if action is not None:
            if filter_actions and filter_action(parent, action):
                continue
            node = next_real(action.apply(parent), generator)
            if node in closed:
                continue
            closed.add(node)
            came_from[node] = (parent, action)
            if node.is_goal():
                goal = node
                break

        actions = list(generator.action_generator.generate_actions(node))
        actions.sort(key=lambda a: str(a))
        counter += 1
        for a in actions:
            child = next_real(a.apply(node), generator)
            h = _hff_value(ff, pm_problem, child)
            heapq.heappush(open_heap, PrioritizedItem(h, (node, a, distance + 1)))

    reason = _termination_reason(
        goal is not None, len(open_heap) == 0, counter, max_expansion, start_time, time_limit_s
    )
    plan = None
    if goal is not None:
        plan = _reconstruct(came_from, initial_state, goal)
        _self_check(generator, initial_state, plan, goal)
    return PlanResult(goal is not None, plan, counter, reason, init_time, goal)


def bfs_then_hff_with_plan(
    generator,
    initial_state,
    max_expansion: int = 1000,
    time_limit_s: float = 0,
    filter_actions: bool = True,
) -> PlanResult:
    """Cascade: bfs_with_plan first, gbfs_hff_with_plan only if BFS doesn't
    solve within budget.

    Needed for labels.py's greedy removal search: BFS
    guarantees the objectively SHORTEST symbolically-reachable plan (breadth-
    first, by construction), which matters a lot here beyond "faster" --
    every extra action in a longer plan is another chance for a `put` to
    target a cell that some OTHER, currently-excluded cylinder really
    occupies, which IsSolution then (correctly) rejects. gbfs_hff_with_plan
    is a greedy, non-optimal search and can find
    needlessly long, convoluted plans (e.g. relocating 3 unrelated cylinders
    through a chain of intermediate cells before finally reaching the goal)
    even when REAP's own reference solution proves a much shorter plan
    exists -- inflating false "insufficient" verdicts during label
    generation well beyond genuine necessity. BFS's own weakness (large
    branching factor on many-cylinder problems can exhaust a modest
    expansion budget -- see labels.py's module docstring) is exactly hFF's
    strength, so falling back to hFF only on BFS's failure gets the best of
    both: correctness-preferring search for the (typical) case it can
    handle, coverage for the harder tail it can't.
    """
    result = bfs_with_plan(generator, initial_state, max_expansion, time_limit_s, filter_actions)
    if result.solved:
        return result
    return gbfs_hff_with_plan(generator, initial_state, max_expansion, time_limit_s, filter_actions)


PLAN_FUNCTIONS = {
    "bfs": bfs_with_plan,
    "hff": gbfs_hff_with_plan,
    "bfs_then_hff": bfs_then_hff_with_plan,
}
