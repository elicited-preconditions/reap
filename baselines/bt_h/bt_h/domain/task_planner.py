"""Purely symbolic task planner over relocate_domain.pddl: plain BFS (always
correct, the safe fallback) and an optional GBFS guided by pymimir's hFF
heuristic. No motion planning is involved here at all -- this produces the
task plan pi (Sec 3.1 of Cieslar et al.), which backtracking.py/bt_h.py then
refine into a controller plan against the real MuJoCo checker.

Written after reap/rl/av_star/search.py::bfs / gbfs_hff, without importing
that module (which loads torch and REAP's GNN encoding).

xmimir/pymimir is the planning substrate; no Fast Downward binary is used.
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

from bt_h.types import GroundAction


class PlanNotFound(Exception):
    pass


@dataclass(order=True)
class _PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


def _to_ground_action(action: xmi.XAction) -> GroundAction:
    if action.name == "pick":
        cyl, cel = (o.name for o in action.objects)
        return GroundAction("pick", cyl, cel)
    elif action.name == "relocate":
        (cyl,) = (o.name for o in action.objects)
        return GroundAction("relocate", cyl, None)
    raise ValueError(f"unexpected action name in task plan: {action.name!r}")


def bfs_plan(
    generator: xmi.XSuccessorGenerator,
    max_expansion: int = 20000,
    time_limit_s: float = 30.0,
) -> list[xmi.XAction]:
    """Plain BFS. Always finds a shortest plan if one exists within budget --
    safe at this scale (K<=6, plan length <=13, small branching
    factor). Frontier entries carry their own action-path (not backpointers)
    for simplicity; fine given how small these searches are."""
    initial = generator.initial_state
    if initial.is_goal():
        return []

    closed = {initial}
    frontier = deque()
    frontier.append((initial, []))
    start = time.time()
    counter = 0

    while frontier and counter < max_expansion and (time.time() - start) < time_limit_s:
        state, path = frontier.popleft()
        actions = list(generator.action_generator.generate_actions(state))
        actions.sort(key=str)
        for a in actions:
            counter += 1
            nxt = a.apply(state)
            if nxt in closed:
                continue
            new_path = path + [a]
            if nxt.is_goal():
                return new_path
            closed.add(nxt)
            frontier.append((nxt, new_path))

    raise PlanNotFound(f"BFS exhausted budget (expansions={counter}) without finding a goal state")


def _hff_value(ff: "pm.FFHeuristic", pm_problem, xstate: xmi.XState) -> float:
    wrapped = pm.State(xstate.base, pm_problem)
    h = ff.compute_value(wrapped)
    return 1e18 if h == float("inf") else float(h)


def gbfs_hff_plan(
    generator: xmi.XSuccessorGenerator,
    max_expansion: int = 20000,
    time_limit_s: float = 30.0,
) -> list[xmi.XAction]:
    """GBFS guided by pymimir's hFF heuristic. Faster than plain BFS on
    larger instances, but only used opportunistically -- see plan_task's
    fallback-to-BFS-on-any-failure policy, since hFF's compatibility with
    this domain's :derived-predicates axioms is not guaranteed a
    priori."""
    initial = generator.initial_state
    if initial.is_goal():
        return []

    pm_problem = generator.problem.base
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        ff = pm.FFHeuristic(pm_problem)

    closed = {initial}
    heap = []
    heapq.heappush(heap, _PrioritizedItem(_hff_value(ff, pm_problem, initial), (initial, [])))
    start = time.time()
    counter = 0

    while heap and counter < max_expansion and (time.time() - start) < time_limit_s:
        item = heapq.heappop(heap)
        state, path = item.item
        actions = list(generator.action_generator.generate_actions(state))
        actions.sort(key=str)
        for a in actions:
            counter += 1
            nxt = a.apply(state)
            if nxt in closed:
                continue
            new_path = path + [a]
            if nxt.is_goal():
                return new_path
            closed.add(nxt)
            heapq.heappush(heap, _PrioritizedItem(_hff_value(ff, pm_problem, nxt), (nxt, new_path)))

    raise PlanNotFound(f"GBFS-hFF exhausted budget (expansions={counter}) without finding a goal state")


def plan_task(
    domain_path,
    problem_path,
    prefer_hff: bool = True,
    max_expansion: int = 20000,
    time_limit_s: float = 30.0,
) -> list[GroundAction]:
    """Parse (domain, problem) and return a task plan pi as a list of
    GroundAction. Tries hFF-guided GBFS first (if prefer_hff), falling back
    to plain BFS on ANY failure (heuristic construction error, no plan found
    within its own budget, etc.) -- BFS is re-attempted with the same
    budget, so a real "no plan exists" failure still raises PlanNotFound."""
    domain, problem = xmi.parse(str(domain_path), str(problem_path))
    generator = xmi.XSuccessorGenerator(problem)

    raw_plan = None
    if prefer_hff:
        try:
            raw_plan = gbfs_hff_plan(generator, max_expansion=max_expansion, time_limit_s=time_limit_s)
        except Exception:
            raw_plan = None  # fall through to BFS below

    if raw_plan is None:
        raw_plan = bfs_plan(generator, max_expansion=max_expansion, time_limit_s=time_limit_s)

    return [_to_ground_action(a) for a in raw_plan]
