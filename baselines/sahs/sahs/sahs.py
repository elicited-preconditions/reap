"""SAHS (sampling-based abstract-edge heuristic search, Algorithm 2 of Kim et al., IJRR 2022), on REAP's open-tabletop
domain: priority queue over abstract
edges (state, action), popped by priority = H_rank(state,action), matching
the journal version's Eq 2:

    H_rank(alpha(s,G), a(delta)) = H(alpha(s,G))
        - exp(F(alpha(s,G),delta;theta)) / sum_{a',delta'} exp(F(alpha(s,G),delta';theta))

i.e. H(state) minus a softmax-normalized Q/rank value among sibling
abstract actions in the SAME state -- lower H_rank pops first (standard
min-heap best-first convention; H_rank is low exactly when H is low
[close to goal] and/or Q ranks this action highly among the alternatives).

H(alpha(s,G)) itself is the paper's Eq 1: H = H_COUNT(alpha(s,G))
- |O_achieved(alpha(s,G),G)| + 1[InRegion(o,r) and (o,r) in G]. The last two
terms are specific to the paper's general region-placement goal structure
(multiple (object,region) goal pairs, with a bonus for objects already in
their designated goal region); REAP's tabletop domain has a single
`(holding target)` goal with no goal REGIONS at all, so there is no
"O_achieved" set or "already in its goal region" bonus to compute here:
H(state) is H_COUNT(state).

H_COUNT itself is the paper's Algorithm 1: starting from the goal object(s)
not yet achieved, recursively chase occluders-of-occluders via BOTH
OccludesPre (blocks picking) and OccludesManip (blocks placing elsewhere)
transitively -- not just a one-hop count of direct OccludesPre occluders of
the target.

Key domain simplification: pick/put preconditions (gripper-empty vs.
holding) are mutually exclusive, so at any single state only one operator
is symbolically applicable -- no within-state pick-vs-put comparison is
ever needed (the heap still mixes edges across different frontier states,
standard best-first search).

"Continuous instantiation" at pop time is free by construction in this
domain: realizing an action literally IS checker.check(...), the same call
already made (and cached) while computing pre_free/manip_free at push time
-- reused here on first pop, zero extra calls. Only a retry (edge failed
once, re-pushed at the same priority) uses force_refresh=True, since
tampanda's RRT* is stochastic and retrying against the cache would be a
silent no-op.

The regions of `put` are the empty cells of the state (`state_adapter.empty_cells(rs, ALL_CELLS)`), as in the SAHS
problem definition (IJRR 2022, Section 3: the "set of workspace regions" is part of the environment) and in REAP's
search, which considers every empty cell. `TripleRelationalQNet` (gnn.py) takes the number of regions as a forward
argument (`num_regions`).
"""
from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from dataclasses import dataclass, field

import torch
import reap.xmimir as xmi

from sahs import state_adapter
from sahs.checker import InstrumentedFeasibilityChecker
from sahs.features import build_edge_tensors, entities_for_pick, entities_for_put
from sahs.gnn import TripleRelationalQNet
from sahs.parallel_pool import DeadlineExceeded
from sahs.occlusion import OcclusionChecker
from sahs.predicates import (
    PredicateSet,
    compute_predicate_set,
    compute_predicate_set_incremental,
    compute_predicate_set_parallel,
)

ALL_CELLS = [f"cell_{ix}_{iy}" for ix in range(10) for iy in range(7)]


def h_count(predicate_set, target: str, cylinders: list[str], held: str | None) -> int:
    """Paper's Algorithm 1, adapted: M starts as {target} if not yet held
    (the paper's version starts M from unachieved (goal_object,goal_region)
    pairs; this domain's only goal is holding `target`). Recursively chases
    occluders-of-occluders via both OccludesPre and OccludesManip."""
    if held == target:
        return 0
    manip_blockers = {(o1, o2) for (o1, o2, _r), v in predicate_set.occludes_manip.items() if v}
    m = {target}
    queue = deque([target])
    while queue:
        om = queue.popleft()
        for o in cylinders:
            if o in m:
                continue
            if predicate_set.occludes_pre.get((o, om), False) or (om, o) in manip_blockers:
                m.add(o)
                queue.append(o)
    return len(m)


@dataclass
class _Edge:
    state: xmi.XState
    action: xmi.XAction
    parent: "_Edge | None"
    retries: int = 0
    # Predicate set / region set computed for `state` (the state this edge
    # is PUSHED FROM, before `action` is applied) -- populated by
    # push_all_abstract_edges right after computing them, so that when this
    # edge is later popped and realized, the resulting child state's own
    # expansion can incrementally diff against these instead of recomputing
    # everything from scratch. See predicates.py's
    # compute_predicate_set_incremental docstring.
    ps: "PredicateSet | None" = None
    regions: "list[str] | None" = None


@dataclass(order=True)
class _QueueItem:
    priority: float
    counter: int
    edge: _Edge = field(compare=False)


def sahs_search(
    problem: xmi.XProblem,
    target: str,
    cylinders: list[str],
    q_pick: TripleRelationalQNet,
    q_put: TripleRelationalQNet,
    checker: InstrumentedFeasibilityChecker,
    occlusion_checker: OcclusionChecker,
    max_pops: int = 500,
    max_retries_per_edge: int = 3,
    pool=None,
    time_limit_s: float | None = None,
    stats_out: dict | None = None,
    on_pop=None,
) -> list[tuple[xmi.XState, xmi.XAction]] | None:
    """`pool`, if given a `parallel_pool.ParallelFeasibilityPool`, computes the predicate set of each expanded state
    (most of the cost) in parallel instead of with the sequential `checker`/`occlusion_checker`. The realization of
    a popped action is one call and always uses `checker`.

    `time_limit_s`, if given, stops the search (Table 1: 300 s per problem); `max_pops` is a backstop. The limit
    bounds the whole search: the pool batches that compute predicates stop at the deadline (DeadlineExceeded), and a
    realization check that returns after it does not count.

    `stats_out`, if given a dict, receives `{"expansions": int}` before every return: the number of states expanded
    (calls of push_all_abstract_edges on a new state), the numerator of Exp./steps in Table 1.

    `on_pop`, if given, is called as `on_pop(edge, feasible)` after each realization check; it does not change the
    search."""
    gen = xmi.XSuccessorGenerator(problem)
    init_state = gen.initial_state

    heap: list[_QueueItem] = []
    counter = itertools.count()
    visited: set[tuple] = set()
    expansions = 0

    def is_goal(s: xmi.XState) -> bool:
        return state_adapter.state_to_robot_state(s)["holding"] == target

    def push_all_abstract_edges(state: xmi.XState, parent: _Edge | None) -> None:
        nonlocal expansions
        rs = state_adapter.state_to_robot_state(state)
        key = state_adapter.state_key(rs)
        if key in visited:
            return
        visited.add(key)
        expansions += 1
        regions = state_adapter.empty_cells(rs, ALL_CELLS)

        if pool is not None and parent is not None and parent.ps is not None:
            moved_object = parent.action.objects[0].name
            ps = compute_predicate_set_incremental(
                state, target, cylinders, ALL_CELLS, pool, regions,
                parent.ps, parent.regions, moved_object, deadline=deadline,
            )
        elif pool is not None:
            ps = compute_predicate_set_parallel(
                state, target, cylinders, ALL_CELLS, pool, regions, deadline=deadline,
            )
        else:
            ps = compute_predicate_set(
                state, target, cylinders, ALL_CELLS, checker, regions,
                occlusion_checker=occlusion_checker,
            )
        h = h_count(ps, target, cylinders, rs["holding"])
        actions = list(gen.action_generator.generate_actions(state))

        if rs["holding"] is None:
            pick_actions = [a for a in actions if a.name == "pick"]
            if not pick_actions:
                return
            obj_e, reg_e = entities_for_pick(cylinders)
            x_obj, edge_t = build_edge_tensors(obj_e, reg_e, ps, target, feature_set=q_pick.edge_features)
            with torch.no_grad():
                f_vals = q_pick(x_obj, edge_t, num_regions=len(reg_e)).squeeze(-1)
            f_by_obj = {o: f_vals[i].item() for i, o in enumerate(cylinders)}
            relevant = [(a, f_by_obj[a.objects[0].name]) for a in pick_actions]
        else:
            # `regions` already equals the domain's own empty-cell set, which
            # is exactly what generate_actions grounds `put`'s ?cel over --
            # this filter is a defensive no-op, not a restriction.
            put_actions = [
                a for a in actions if a.name == "put" and a.objects[1].name in regions
            ]
            if not put_actions:
                return
            held = rs["holding"]
            obj_e, reg_e = entities_for_put(cylinders, regions)
            x_obj, edge_t = build_edge_tensors(obj_e, reg_e, ps, target, feature_set=q_put.edge_features)
            with torch.no_grad():
                f_vals = q_put(x_obj, edge_t, num_regions=len(reg_e))
            row = obj_e.index(held)
            relevant = [(a, f_vals[row, reg_e.index(a.objects[1].name)].item()) for a in put_actions]

        if not relevant:
            return
        fs = torch.tensor([f for _a, f in relevant])
        probs = torch.softmax(fs, dim=0).tolist()

        for (action, _f), prob in zip(relevant, probs):
            priority = h - prob  # Eq 2, min-heap: lower H_rank pops first
            edge = _Edge(state=state, action=action, parent=parent, ps=ps, regions=regions)
            heapq.heappush(heap, _QueueItem(priority, next(counter), edge))

    t0 = time.perf_counter()
    # time_limit_s bounds the whole search, nested calls included: the
    # feature-building pool batches stop at the deadline (DeadlineExceeded,
    # parallel_pool.py), and a realization check that returns after it does
    # not count.
    deadline = None if time_limit_s is None else t0 + time_limit_s
    try:
        push_all_abstract_edges(init_state, None)
    except DeadlineExceeded:
        if stats_out is not None:
            stats_out["expansions"] = expansions
        return None

    pops = 0
    while heap and pops < max_pops:
        if time_limit_s is not None and (time.perf_counter() - t0) >= time_limit_s:
            if stats_out is not None:
                stats_out["expansions"] = expansions
            return None
        item = heapq.heappop(heap)
        edge = item.edge
        pops += 1

        rs = state_adapter.state_to_robot_state(edge.state)
        if edge.action.name == "pick":
            feasible, _info = checker.check(
                rs, "pick", edge.action.objects[0].name,
                predicate_label="SAHS", force_refresh=(edge.retries > 0),
            )
        else:
            feasible, _info = checker.check(
                rs, "put", edge.action.objects[0].name, edge.action.objects[1].name,
                predicate_label="SAHS", force_refresh=(edge.retries > 0),
            )

        if on_pop is not None:
            on_pop(edge, feasible)

        if deadline is not None and time.perf_counter() >= deadline:
            if stats_out is not None:
                stats_out["expansions"] = expansions
            return None

        if not feasible:
            if edge.retries < max_retries_per_edge:
                edge.retries += 1
                heapq.heappush(heap, _QueueItem(item.priority, next(counter), edge))
            continue

        next_state = edge.action.apply(edge.state)
        if is_goal(next_state):
            plan_edges = [edge]
            e = edge.parent
            while e is not None:
                plan_edges.append(e)
                e = e.parent
            plan_edges.reverse()
            if stats_out is not None:
                stats_out["expansions"] = expansions
            return [(e.state, e.action) for e in plan_edges]

        try:
            push_all_abstract_edges(next_state, edge)
        except DeadlineExceeded:
            if stats_out is not None:
                stats_out["expansions"] = expansions
            return None

    if stats_out is not None:
        stats_out["expansions"] = expansions
    return None
