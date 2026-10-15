# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from __future__ import annotations

import contextlib
import heapq
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pymimir as pm
import torch
import tqdm

import reap.xmimir as xmimir
import reap.xmimir.iw as iw
from reap.xmimir import XState

from ...encoding import HorizonHGraphEncoder
from ...encoding.horizon_hetero_encoder import TransitionDAG

from . import Model
from .sim import filter_action, get_filter_stats, get_mp_calls, reset_filter_stats


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


# EXPLORE_DIAG=1: bwqs records exploration statistics of each search (goal atoms still unsatisfied, depth,
# unique states) on the root node, and worker.py prints them.
_EXPLORE_DIAG = os.environ.get("EXPLORE_DIAG", "") not in ("", "0")
_GOAL_OCC = re.compile(r"\+?\(occupied\s+(\S+)\s+([^\s)]+)\s*\)")
_GOAL_EMPTY = re.compile(r"\+?\(empty\s+([^\s)]+)\s*\)")


def _goal_pairs_of(generator):
    """Goal requirements as a set of keys (EXPLORE_DIAG statistics):
         ("occ", cell, obj)  for (occupied cell obj) goal atoms
         ("empty", cell)     for (empty cell) goal atoms
    `literal_goal` is a METHOD on xmimir problems.
    """
    prob = generator.problem
    prob = prob() if callable(prob) else prob
    lits = prob.literal_goal
    if callable(lits):
        lits = lits()
    out = set()
    for lit in lits:
        text = str(lit)
        m = _GOAL_OCC.match(text)
        if m:
            out.add(("occ", m.group(1), m.group(2)))
            continue
        m = _GOAL_EMPTY.match(text)
        if m:
            out.add(("empty", m.group(1)))
    return out


def _unsat(state, pairs):
    """Number of goal requirements NOT met in `state`.

    A cell counts as non-empty iff some (occupied cell *) atom holds, so this
    does not depend on whether the `empty` fluent is maintained by the actions.
    """
    occ = set()
    for a in state.atoms():
        m = _GOAL_OCC.match(str(a))
        if m:
            occ.add((m.group(1), m.group(2)))
    occupied_cells = {c for c, _ in occ}
    n = 0
    for k in pairs:
        if k[0] == "occ":
            if (k[1], k[2]) not in occ:
                n += 1
        elif k[0] == "empty":
            if k[1] in occupied_cells:
                n += 1
    return n


@dataclass
class Node:
    # state: the parent state of this (parent, action) pair (init state for the root node)
    state: xmimir.XState
    # a: the action taken from `state` to reach this node's `next` state.
    #    None on the root init node only.
    a: "xmimir.XAction | None"
    # g: path-length from root (incremented when child node is pushed)
    g: float
    # p: PARENT Node (not state).  None on the root init node only.
    p: "Node | None"
    f: float                                  # heap priority used by the search
    h: float                                  # Q-value estimate stored on the node
    # next: the state produced by applying `a` to `state` (set in bwqs after
    #       filter passes).  None on the root init node and on dead_end / not-yet-
    #       processed nodes.
    next: "xmimir.XState | None"
    successors: list["Node"]
    selected: bool = False
    dead_end: bool = False


def next_real(s, gen):
    if "pushworld" not in gen.problem.domain.name:
        return s
    actions = list(gen.action_generator.generate_actions(s))
    if len(actions) >= 1 and (
        actions[0].name == "ppush" or actions[0].name == "ppush-done"
    ):
        return next_real(actions[0].apply(s), gen)
    return s


# batch weighted q* search -> https://prl-theworkshop.github.io/prl2024-icaps/papers/9.pdf
def bwqs(
    generator,
    initial_state,
    state_embedder,
    max_expansion=2048,
    batch_size=64,
    time_limit_s=300,
    gather_training=True,
    action_cost=0.5,
    max_infeasible_retries=2,
    # Probability of overriding a newly-pushed child's heap priority to 0
    # (high-priority slot), drawn per-child at push time.  Higher value =
    # more exploration of low-Q actions.  Default 0.0 is deterministic
    # Q-greedy search.  Useful >0 during the random-Q bootstrap.
    random_exploration=0.0,
):
    start_time = time.time()

    closed = {}
    done = []
    infeasible_per_state: dict = {}  # parent_state -> count of filtered actions seen
    actions_per_state: dict = {}     # parent_state -> total action count at expansion time
    exhausted_states: set = set()    # states where max_infeasible_retries was reached

    # state, action, distance, pre, f, h, selected, children
    init = Node(initial_state, None, 0, None, 0, 0, None, [])

    _gpd = None
    _diag = None
    if _EXPLORE_DIAG:
        _gpd = _goal_pairs_of(generator) or None
        _u0 = _unsat(initial_state, _gpd) if _gpd is not None else -1
        _diag = {"init_unsat": _u0, "min_unsat": _u0, "max_depth": 0}

    open = []
    heapq.heappush(open, PrioritizedItem(0, init))

    goal = None

    while (
        len(open) > 0
        and len(closed) < max_expansion
        and (time.time() - start_time < time_limit_s or time_limit_s <= 0)
    ):
        generated = []
        s = []

        while len(open) > 0 and len(s) < batch_size:
            node = heapq.heappop(open).item
            done.append(node)

            next_state = node.state
            if node.a is not None:
                next_state = next_real(node.a.apply(node.state), generator)
                if node.state in exhausted_states:
                    continue
                if filter_action(node.state, node.a):
                    count = infeasible_per_state.get(node.state, 0) + 1
                    infeasible_per_state[node.state] = count
                    total = actions_per_state.get(node.state)
                    if count >= max_infeasible_retries or (total is not None and count >= total):
                        node.dead_end = True
                        exhausted_states.add(node.state)
                    continue

            node.next = next_state

            # found a goal -> exit condition
            if next_state.is_goal():
                goal = node
                break

            g = node.g + 1

            if next_state not in closed:
                closed[next_state] = g

                actions = list(generator.action_generator.generate_actions(next_state))
                actions.sort(key=lambda key: str(key))
                actions_per_state[next_state] = len(actions)

                for a in actions:
                    generated.append((next_state, a, g, node))
                    if next_state not in s:
                        s.append(next_state)

                if len(actions) == 0:
                    node.dead_end = True

        if goal is not None:
            break
        if len(s) == 0:
            continue

        values, batch = state_embedder.get_q_values(s)
        c = 0
        for next, h in zip(generated, values):
            s, a, g, p = next

            f = action_cost * g - h
            if _diag is not None:
                if g > _diag["max_depth"]:
                    _diag["max_depth"] = g
                if _gpd is not None:
                    _u = _unsat(a.apply(s), _gpd)
                    if _u < _diag["min_unsat"]:
                        _diag["min_unsat"] = _u
            n = Node(s, a, g, p, f, h, None, [])
            if random.random() < random_exploration:
                # promote to the queue head
                f = 0
            p.successors.append(n)
            heapq.heappush(open, PrioritizedItem(f, n))

    if _diag is not None:
        _diag["uniq"] = len(closed)
        init.explore = _diag
    # rebuild goal path
    next = goal
    c = 0.0
    while next is not None:
        c -= 1
        next.goal_path = True
        next = next.p

    if gather_training:
        # Build training data even when no goal was found: the `dead_end`
        # marks of a failed search are the negative training signal (with
        # retries=0 and a bad Q, most attempts end at counter=1 with a single
        # dead_end node).  build_training handles goal=None (its goal-path
        # walk is a no-op then).
        states, targets, target_map, bound = build_training(init, goal, generator)
    else:
        targets = closed
        states = []
        target_map = []
        bound = []
    # difficulty ratio
    if -c <= 1:
        difficulty = 0
    else:
        difficulty = len(targets) / (-c - 1)

    reason = _termination_reason(
        goal is not None, len(open) == 0,
        len(closed), max_expansion, start_time, time_limit_s,
    )

    # `init` (the root Node of this attempt's search tree) is returned so
    # callers can read its EXPLORE_DIAG statistics (init.explore).
    return states, targets, target_map, bound, difficulty, len(closed), goal, reason, init


def is_dead_end(state: xmi.XState, gen: xmi.XSuccessorGenerator):
    actions = list(gen.action_generator.generate_actions(state))
    if len(actions) == 0:
        return True
    return False


def build_training(init, goal, gen):
    open_states = set()
    bound = []
    target_map = []
    targets = []
    target_ids = {None: 0}
    bound_targets = {}
    states = []

    # rebuild goal path
    open2 = []
    next = goal
    c = 0.0
    while next is not None:
        # walk the goal path back to the root
        open2 += next.successors
        bound_targets[next.next] = c
        c -= 1
        next = next.p

    if goal is None:
        # Failed search (no goal found).  open2 (goal-path successors) is
        # empty, but the search tree still contains dead_end markings that
        # feed the trainer as DEAD_END_REWARD targets.  Walk
        # init.successors directly: those are the (parent, action) nodes
        # that were attempted and either accepted (continued downstream) or
        # rejected (filter+exhaustion → dead_end=True).  In the stuck case
        # (counter=1, all children rejected) this gives 1 dead_end mark
        # per attempt; over many attempts the model learns to avoid those
        # specific infeasible (state, action) pairs and the search can
        # eventually progress past the first-action-rejection trap.
        open = [*init.successors]
    else:
        # Goal found.  Walk init.successors to seed
        # open_states (states considered but not on the goal path),
        # then process open2 (goal-path and adjacent successors).
        open = [*init.successors]
        while len(open) > 0:
            f = open.pop(0)
            open_states.add(f.state)
            open += f.successors
            if len(f.successors) > 0 and f.successors[0] not in open2:
                open2 += f.successors
        open = open2

    # target_ids match
    counter = 0
    while len(open) > 0:
        counter += 1
        f = open.pop(0)
        if f.state not in states:
            states.append(f.state)
        if f.dead_end:
            target_map.append(1)
            bound.append(torch.nan)
            continue
        if f.next not in target_ids:
            # an unopened node may be a dead end; test it explicitly and mark it
            if is_dead_end(f.next, gen):
                target_map.append(1)
                bound.append(torch.nan)
                continue
            if f.next not in open_states and f.next not in bound_targets:
                pass
            target_ids[f.next] = len(targets) + 2
            targets.append(f.next)
        if f.next in bound_targets:
            bound.append(bound_targets[f.next])
        else:
            bound.append(torch.nan)
        target_map.append(target_ids[f.next])
    return states, targets, target_map, bound


def gbfs_value(
    generator,
    initial_state,
    state_embedder,
    max_expansion=40,
    time_limit_s=0,
    weight=0.0,
    weight2=1.0,
    filter_actions=False,
):
    # When ``filter_actions`` is True, each popped (parent, action) pair is
    # checked via filter_action() before the symbolic transition; infeasible
    # children are skipped (the search continues with the next heap item).
    # No per-state retry cap: at inference, feasible branches are explored
    # fully until max_expansion or an empty heap.  Default False runs a
    # symbolic-only search.
    init_start = time.time()
    closed = set({})
    open = []

    heapq.heappush(open, PrioritizedItem(0, (initial_state, None, 0)))

    init_time = time.time() - init_start
    start_time = time.time()

    counter = 0
    goal = None
    goal_path_length = 0
    while (
        len(open) > 0
        and counter < max_expansion
        and (time.time() - start_time < time_limit_s or time_limit_s <= 0)
    ):
        l = heapq.heappop(open)

        parent, action, distance = l.item
        node = parent
        if action is not None:
            if filter_actions and filter_action(parent, action):
                continue
            node = next_real(action.apply(parent), generator)
            if node not in closed:
                closed.add(node)
            else:
                continue
            if node.is_goal():
                goal_path_length = distance
                goal = node
                break

        actions = list(generator.action_generator.generate_actions(node))
        actions.sort(key=lambda key: str(key))

        values, batch = state_embedder.get_q_values([node])
        counter += 1

        for a, v in zip(actions, values):
            heapq.heappush(
                open,
                PrioritizedItem(
                    -weight2 * v + weight * (distance + 1), (node, a, distance + 1)
                ),
            )

    reason = _termination_reason(
        goal is not None, len(open) == 0, counter, max_expansion, start_time, time_limit_s
    )
    return counter, goal_path_length, goal is not None, reason, init_time


def _termination_reason(solved, frontier_empty, counter, max_expansion, start_time, time_limit_s):
    # Diagnose why a search loop exited, in priority order.  "max_budget" and
    # "timeout" matter for distinguishing problems that need a longer budget
    # vs. ones where filter_action / RRT churn is the bottleneck.
    if solved:
        return "solved"
    if frontier_empty:
        return "dead_end"
    if counter >= max_expansion:
        return "max_budget"
    if time_limit_s > 0 and (time.time() - start_time) >= time_limit_s:
        return "timeout"
    return "max_budget"


def bfs(
    generator,
    initial_state,
    state_embedder,
    max_expansion=40,
    time_limit_s=0,
    filter_actions=False,
):
    # Plain breadth-first search baseline.  state_embedder is accepted to
    # match the SEARCH_FUNCTIONS calling convention and ignored.  When
    # ``filter_actions`` is True, infeasible (parent, action) pairs are
    # skipped via filter_action(), same semantics as gbfs_value's flag.
    init_start = time.time()
    closed = set({})
    frontier = deque()

    frontier.append((initial_state, None, 0))

    init_time = time.time() - init_start
    start_time = time.time()

    counter = 0
    goal = None
    goal_path_length = 0
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
            if node.is_goal():
                goal_path_length = distance
                goal = node
                break
        else:
            closed.add(node)

        actions = list(generator.action_generator.generate_actions(node))
        actions.sort(key=lambda key: str(key))
        counter += 1

        for a in actions:
            frontier.append((node, a, distance + 1))

    reason = _termination_reason(
        goal is not None, not frontier, counter, max_expansion, start_time, time_limit_s
    )
    return counter, goal_path_length, goal is not None, reason, init_time


def _hff_value(ff, pm_problem, xstate):
    # FFHeuristic.compute_value wants a wrapped pm.State, not the raw
    # `advanced` state that xmi.XState.base exposes.  Wrap each call.
    # Returns float('inf') for dead-ends; convert to a large finite
    # priority so heapq comparisons are well-defined.
    wrapped = pm.State(xstate.base, pm_problem)
    h = ff.compute_value(wrapped)
    if h == float("inf"):
        return 1e18
    return float(h)


def gbfs_hff(
    generator,
    initial_state,
    state_embedder,
    max_expansion=40,
    time_limit_s=0,
    filter_actions=False,
):
    # GBFS guided by pymimir's hFF heuristic.  state_embedder is accepted
    # to match the SEARCH_FUNCTIONS calling convention and ignored.
    # FFHeuristic init prints LiftedGrounder atom stats to stdout; silence
    # it so per-problem logs stay clean.
    init_start = time.time()
    pm_problem = generator.problem.base
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        ff = pm.FFHeuristic(pm_problem)

    closed = set({})
    open_heap = []

    heapq.heappush(
        open_heap,
        PrioritizedItem(
            _hff_value(ff, pm_problem, initial_state), (initial_state, None, 0)
        ),
    )

    init_time = time.time() - init_start
    start_time = time.time()

    counter = 0
    goal = None
    goal_path_length = 0
    while (
        open_heap
        and counter < max_expansion
        and (time.time() - start_time < time_limit_s or time_limit_s <= 0)
    ):
        l = heapq.heappop(open_heap)
        parent, action, distance = l.item
        node = parent
        if action is not None:
            if filter_actions and filter_action(parent, action):
                continue
            node = next_real(action.apply(parent), generator)
            if node in closed:
                continue
            closed.add(node)
            if node.is_goal():
                goal_path_length = distance
                goal = node
                break
        else:
            closed.add(node)

        actions = list(generator.action_generator.generate_actions(node))
        actions.sort(key=lambda key: str(key))
        counter += 1

        for a in actions:
            child = next_real(a.apply(node), generator)
            h = _hff_value(ff, pm_problem, child)
            heapq.heappush(open_heap, PrioritizedItem(h, (node, a, distance + 1)))

    reason = _termination_reason(
        goal is not None, len(open_heap) == 0, counter, max_expansion, start_time, time_limit_s
    )
    return counter, goal_path_length, goal is not None, reason, init_time


def rollout(
    generator,
    initial_state,
    state_embedder,
    max_expansion=40,
    closed=False,
    stochastic=False,
    time_limit_s=0,
    log_level=0,
    compute_loss=False,
    max_infeasible_retries=0,
):
    epsilon = 0.5
    history = []  # track the number of actions possible, the predicated value

    current = initial_state
    counter = 0
    visited = set()
    start_time = time.time()
    while (
        not current.is_goal()
        and counter < max_expansion
        and (time.time() - start_time < time_limit_s or time_limit_s <= 0)
    ):
        visited.add(current)
        counter += 1

        if isinstance(state_embedder.model.encoder, HorizonHGraphEncoder):
            transitions = list(generator.successors(current))
            transitions.sort(
                key=lambda t: str(t.source.fluent_atoms) + str(t.target.fluent_atoms)
            )

            dag = TransitionDAG(
                current,
                transitions,
                ignore_actions=True,
                actions_only=False,
            )
            target = [t for t, idx in dag if idx != 0]

        elif isinstance(generator, xmi.XSuccessorGenerator) and not isinstance(
            generator, iw.IWSuccessorGenerator
        ):
            actions = list(generator.action_generator.generate_actions(current))
            actions.sort(key=lambda key: str(key))
            target = [next_real(t.apply(current), generator) for t in actions]

            # not allowed mask
            # test if action is possible under constraints
            occluded_objects = [x.objects[0] for x in current.derived_atoms if x.predicate.name == "occluder"]
            not_allowed_mask = [a.objects[0] in occluded_objects for a in actions]

        output = state_embedder.get_q_values([current])
        og_output = output[0].clone()

        # deadend
        if len(output[0]) == 0:
            if compute_loss:
                return counter, current.is_goal(), 0
            return counter, current.is_goal()

        if closed:
            visited_maks = torch.tensor([t not in visited for t in target])

            output[0][~visited_maks] = -torch.inf

        # --- action selection with infeasibility retries ---
        # Mirrors bwqs: try actions in Q-order, skip filtered ones up to
        # max_infeasible_retries times before treating the state as a dead end.
        q_vals = output[0].clone()
        infeasible_count = 0
        i = None
        while True:
            if stochastic:
                best_q = torch.max(q_vals)
                near_optimal_idxs = torch.where((best_q - q_vals) <= epsilon)[0]
                candidate = near_optimal_idxs[torch.randint(len(near_optimal_idxs), (1,))]
            else:
                candidate = torch.argmax(q_vals)

            if torch.isinf(q_vals[candidate]) and q_vals[candidate] < 0:
                # all options exhausted (closed-list or all retries used)
                break

            if filter_action(current, actions[candidate]):
                infeasible_count += 1
                q_vals[candidate] = -torch.inf  # mask out and retry
                if infeasible_count > max_infeasible_retries:
                    break
            else:
                i = candidate
                break

        if i is None:
            # dead end
            if compute_loss:
                return counter, current.is_goal(), 0
            return counter, current.is_goal()

        if log_level > 0:
            print(i, output[0], str(actions[i]))

        history.append((len(output[0]), og_output[i]))
        current = target[i]

    if log_level > 1:
        return counter, current.is_goal(), history

    if compute_loss:
        loss = 0
        if len(history) <= 1:
            return counter, current.is_goal(), 0
        for x in range(len(history))[1:]:
            loss += (history[-x][1] + x - 1) ** 2
        return counter, current.is_goal(), ((loss / len(history)) ** 0.5).item()

    return counter, current.is_goal()


import pandas as pd

import reap.xmimir as xmi

from .utils import StateEmbedder


def rollouts(
    problems,
    agent,
    max_expansion=40,
    closed=False,
    stochastic=False,
    func=None,
    time_limit_s=0,
    compute_rmse=False,
    max_infeasible_retries=0,
    filter_actions=False,
    count_distinct_queries=False,
):
    # count_distinct_queries=True: the Table 1 evaluation path. Per problem,
    # the filter counters AND the feasibility verdict caches are reset before the search (verdicts are reused within a
    # problem only), and three columns are added: mp_calls (MP/plan: distinct (state, action) feasibility queries, the
    # cache size after the problem), total_filter_calls (every filter_action call, cache hits included) and
    # infeasible_steps (infeasible verdicts, cache hits included). Default False (e.g. training-time validation) adds no
    # columns and does not reset the caches.
    gen_class = xmi.XSuccessorGenerator
    se = StateEmbedder(agent, low_memory=True)

    generators = [gen_class(p) for p in problems]

    out = []
    for g in generators:
        o = []
        o.append(g.problem.filepath[-40:])
        o.append(1)
        start_time = time.time()
        expanded = 0
        init_time = 0.0
        termination_reason = ""
        if count_distinct_queries:
            reset_filter_stats(clear_cache=True)
        if func is not None:
            # Forward filter_actions to search-based modes (gbfs_value /
            # gbfs_probes) so they can call filter_action() when the caller
            # opts in.  Greedy's max_infeasible_retries is independent and
            # handled below via rollout().  Default filter_actions=False
            # runs a symbolic-only search.
            expanded, steps, solved, termination_reason, init_time = func(
                g, g.initial_state, se, max_expansion, time_limit_s,
                filter_actions=filter_actions,
            )
        else:
            if compute_rmse:
                steps, solved, rmse = rollout(
                    g,
                    g.initial_state,
                    se,
                    max_expansion,
                    closed,
                    stochastic,
                    time_limit_s,
                    compute_loss=compute_rmse,
                    max_infeasible_retries=max_infeasible_retries,
                )
            else:
                steps, solved = rollout(
                    g,
                    g.initial_state,
                    se,
                    max_expansion,
                    closed,
                    stochastic,
                    time_limit_s,
                    max_infeasible_retries=max_infeasible_retries,
                )
            termination_reason = "solved" if solved else "rollout_failed"
        end_time = time.time()
        elapsed_time = end_time - start_time
        if count_distinct_queries:
            fstats = get_filter_stats()
        o.append(1 if solved else 0)
        o.append(expanded)
        o.append(steps)
        o.append(elapsed_time)
        o.append(termination_reason)
        o.append(init_time)
        if count_distinct_queries:
            o.append(get_mp_calls())
            o.append(fstats["calls"])
            o.append(fstats["infeasible"])
        if compute_rmse:
            o.append(rmse)

        out.append(o)

    base_cols = [
        "problem",
        "total",
        "solved",
        "expanded",
        "steps",
        "wall_time",
        "termination_reason",
        "init_time",
    ]
    if count_distinct_queries:
        base_cols += ["mp_calls", "total_filter_calls", "infeasible_steps"]
    if compute_rmse:
        return pd.DataFrame(out, columns=base_cols + ["rmse"])
    return pd.DataFrame(out, columns=base_cols)


