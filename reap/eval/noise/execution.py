"""Plan-then-execute loop: N1 (placement drift) + cache condition + N3
(grasp failures), and the one-shot replanning mechanism.

## Why "plan-then-execute", and why the execution-phase checker's cache is
## deliberately PRE-SEEDED rather than an empty cache

The cache condition concerns REAP's deployed verdict cache going stale under
drift: execution-time queries see drifted poses while the cache is keyed on
the symbolic state, so a cached verdict can disagree with reality. This is
reproduced faithfully only if the execution-time recheck consults FIRST the
same (symbolic_state, action) cache entries that the planning search already
computed, i.e. the execution-phase checker's cache must already contain
"feasible" for every edge on the plan, as REAP's process-global checker
would after having just searched that path.

REAP's global checker (a plain `reap.domains.tabletop.TabletopFeasibilityChecker`
singleton) has no true-pose-table hook, so it cannot be reused for this.
Instead: every edge on a reconstructed plan is BY CONSTRUCTION one that
`filter_action` confirmed feasible during planning (infeasible edges are
skipped and never appear in the reconstructed plan), so a FRESH
:class:`~reap.eval.noise.noisy_checker.NoisyTabletopFeasibilityChecker`'s cache
is pre-seeded with "feasible" for exactly those (state, action) pairs before
the execution-time rechecks start. This reproduces "the cache already has
these entries from the planning search" without touching any real object's
internals; see `seed_cache_from_plan`.

The pre-seeding is used for N1 (and the cache-condition measurement), since
the concern is drift-driven staleness. N2 (occupancy corruption) and N3
(execution failure) have no cache-staleness interaction and use plain,
unseeded per-rollout checkers instead; see `execute_plan_n2` /
`execute_plan_n3`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import reap.xmimir as xmi

from reap.eval.eval_protocol import gbfs_value_eval

from reap.eval.noise import config
from reap.eval.noise.greedy_capture import rollout_with_capture
from reap.eval.noise.noisy_checker import NoisyTabletopFeasibilityChecker
from reap.eval.noise.true_pose import TruePoseTable, seed_from_problem_id


# ---------------------------------------------------------------------------
# Baseline (sigma=0 / p=0) plan generation, i.e. "planning-time":
# idealized cell-centre geometry throughout, matching REAP's reported
# behavior (GBFS via the unmodified eval_protocol.gbfs_value_eval; greedy via
# the greedy_capture copy with a plain, unparametrized checker, which behaves
# like the global one at true_pose_table=None/occupancy_flip_p=0).
# ---------------------------------------------------------------------------

def plan_gbfs(generator, initial_state, embedder):
    """Real, unmodified REAP search. Returns (expanded, plan, solved, reason, init_time)."""
    return gbfs_value_eval(
        generator, initial_state, embedder,
        max_expansion=config.EVAL_MAX_BUDGET, time_limit_s=config.EVAL_TIME_LIMIT_S,
    )


_planning_checker: Optional[NoisyTabletopFeasibilityChecker] = None


def _get_planning_checker() -> NoisyTabletopFeasibilityChecker:
    """Per-process singleton, like reap.rl.av_star.sim's _CHECKERS pattern
    (one MuJoCo scene per worker process, reused across every problem that
    process touches), NOT a new checker per call. A new
    NoisyTabletopFeasibilityChecker() pays a MuJoCo cold start (~9-10 s) on
    its first check() call; without this singleton, plan_greedy would pay
    that cost on EVERY problem instead of once per process, which makes the
    300-problem sweep intractable."""
    global _planning_checker
    if _planning_checker is None:
        _planning_checker = NoisyTabletopFeasibilityChecker()
        _planning_checker.init()
    return _planning_checker


def plan_greedy(generator, initial_state, embedder):
    """The rollout() copy of greedy_capture, with a plain (no-noise) checker
    for the planning pass. Returns (counter, solved, chosen_actions).

    Reuses the process-global planning checker (see _get_planning_checker),
    clearing its cache first to isolate this call's own cache-hit
    statistics, as eval_protocol.py's _eval_one_problem does for the GBFS
    path."""
    planning_checker = _get_planning_checker()
    planning_checker.clear_cache()
    return rollout_with_capture(
        generator, initial_state, embedder, planning_checker,
        max_expansion=config.GREEDY_MAX_EXPANSION, closed=config.GREEDY_CLOSED,
        stochastic=config.GREEDY_STOCHASTIC, time_limit_s=config.EVAL_TIME_LIMIT_S,
        max_infeasible_retries=config.MAX_INFEASIBLE_RETRIES,
    )


def plan_for_mode(search_mode: str, generator, initial_state, embedder):
    """Returns (solved: bool, plan_actions: list)."""
    if search_mode == "gbfs":
        _expanded, plan, solved, _reason, _init_time = plan_gbfs(generator, initial_state, embedder)
        return solved, plan
    elif search_mode == "greedy":
        _counter, solved, plan = plan_greedy(generator, initial_state, embedder)
        return solved, plan
    else:
        raise ValueError(f"unknown search_mode {search_mode!r}, expected 'gbfs' or 'greedy'")


# ---------------------------------------------------------------------------
# Cache pre-seeding (N1 / cache-condition specific)
# ---------------------------------------------------------------------------

def seed_cache_from_plan(checker: NoisyTabletopFeasibilityChecker, initial_state: xmi.XState, plan_actions: list) -> xmi.XState:
    """Populate checker's cache with 'feasible' for every (state, action)
    edge along an already-planned, already-feasible-verified plan (see
    module docstring). Returns the final (goal) state reached by replaying
    the plan symbolically -- callers that don't already have it can use this
    to sanity-check the plan actually reaches the goal.

    IMPORTANT: uses the checker's UNCORRUPTED _state_to_robot_state (N2
    occupancy_flip_p must be 0 on `checker` when this is called) -- seeding
    represents what a clean planning-phase cache would contain, and drawing
    new corruption here would desync the seed key from later real calls.
    """
    checker.init()
    assert checker.occupancy_flip_p == 0.0, "seed_cache_from_plan requires an uncorrupted checker"
    state = initial_state
    for action in plan_actions:
        robot_state = checker._state_to_robot_state(state)
        key = checker._cache_key(robot_state, action)
        checker._cache[key] = False  # False == feasible, matching TabletopFeasibilityChecker.check's convention
        state = action.apply(state)
    return state


# ---------------------------------------------------------------------------
# N1: placement drift + cache condition
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    problem_name: str
    search_mode: str
    sigma_mm: float
    solved_at_baseline: bool
    plan_len: int = 0
    completed_without_replanning: bool = False
    first_failure_step: Optional[int] = None
    replanned: bool = False
    replan_solved: bool = False
    cache_hits: int = 0
    cache_misses: int = 0
    stale_checked: int = 0
    stale_disagree: int = 0
    cache_enabled_solved: bool = False
    error: Optional[str] = None


def _cylinders_from_state(checker: NoisyTabletopFeasibilityChecker, state: xmi.XState) -> dict:
    return checker._state_to_robot_state(state)["cylinders"]


def _apply_and_advance_true_pose(action: xmi.XAction, state: xmi.XState, true_pose_table: TruePoseTable) -> xmi.XState:
    """Advance the symbolic state by `action`, and redraw the placed
    object's true pose (N1: memoryless per-placement redraw, see the
    true_pose module docstring)."""
    if action.name == "put":
        cyl_name = action.objects[0].name
        target_cell = action.objects[1].name
        true_pose_table.redraw(cyl_name, target_cell)
    elif action.name == "pick":
        cyl_name = action.objects[0].name
        true_pose_table.drop_from_table(cyl_name)
    return action.apply(state)


_execution_checker: Optional[NoisyTabletopFeasibilityChecker] = None


def _get_execution_checker() -> NoisyTabletopFeasibilityChecker:
    """Per-process singleton for execution-time checks, separate from
    _get_planning_checker's (different cache lifecycle: execution calls
    clear+reseed per rollout, planning calls clear+reuse per problem). Same
    cold-start rationale (see _get_planning_checker): a new
    NoisyTabletopFeasibilityChecker per problem/rollout costs ~9-10 s each.
    occupancy_flip_p/occupancy_rng and _state_manager.true_pose_table are
    plain mutable attributes, reconfigured per call by the N1/N2/N3
    functions below rather than rebuilding the object."""
    global _execution_checker
    if _execution_checker is None:
        _execution_checker = NoisyTabletopFeasibilityChecker()
        _execution_checker.init()
    return _execution_checker


def execute_plan_n1(
    problem_name: str,
    search_mode: str,
    generator,
    embedder,
    initial_state: xmi.XState,
    plan_actions: list,
    sigma_m: float,
    seed_suffix: str = "",
    stale_sample_rate: float = None,
) -> ExecutionResult:
    """N1 headline experiment + cache condition, for ONE already-sigma0-solved
    problem.

    Runs TWO passes over the SAME drift realization (two newly constructed
    TruePoseTables sharing problem_id/sigma_m/cell_centers reproduce
    identical draws in the same order -- true_pose.py's seeding is
    deterministic -- so both passes see the identical physical world):

    1. UNCACHED (primary -- drives completed_without_replanning /
       first_failure_step / replanned / replan_solved). Every step's
       feasibility is re-verified against the true-pose table via
       fresh_uncached_check, bypassing the cache entirely. This is "execute
       under drift" in the physically meaningful sense; the uncached run is
       the ground truth for verdict correctness and solve rate.

       The CACHE-ENABLED (pre-seeded) path cannot serve this metric: the
       cache key is the symbolic state (not the physical pose), so every
       pre-seeded plan-edge lookup is a GUARANTEED hit regardless of drift,
       and completed_without_replanning would be 100% at every sigma by
       construction.

    2. CACHE-ENABLED (secondary, cache-condition metrics only:
       cache_hits/misses, stale_checked/disagree via sampling,
       cache_enabled_solved). This pass pre-seeds the cache. Its own
       "solved" outcome is EXPECTED to be ~100% by construction for N1 alone
       (the cache has no way to notice the world drifted); reported next to
       pass 1's degrading completion rate, it quantifies the cache condition
       ("solve rate with the cache enabled versus disabled").
    """
    if stale_sample_rate is None:
        stale_sample_rate = config.STALE_VERDICT_SAMPLE_RATE
    result = ExecutionResult(
        problem_name=problem_name, search_mode=search_mode, sigma_mm=sigma_m * 1000.0,
        solved_at_baseline=True, plan_len=len(plan_actions),
    )

    exec_checker = _get_execution_checker()
    exec_checker.occupancy_flip_p = 0.0  # N1 alone: no occupancy corruption
    cell_centers = exec_checker.cell_centers()
    problem_id = f"{problem_name}{seed_suffix}"

    # --- Pass 1: UNCACHED, drives the headline metric ---
    true_pose_table = TruePoseTable(problem_id, sigma_m, cell_centers)
    true_pose_table.init_from_cylinders(_cylinders_from_state(exec_checker, initial_state))
    exec_checker._state_manager.true_pose_table = true_pose_table

    state = initial_state
    for step, action in enumerate(plan_actions):
        infeasible = exec_checker.fresh_uncached_check(state, action)
        if infeasible:
            result.first_failure_step = step
            break
        state = _apply_and_advance_true_pose(action, state, true_pose_table)
    else:
        result.completed_without_replanning = True

    if not result.completed_without_replanning:
        # Disagreement found: one-shot replan from the current actual
        # (symbolically unchanged by drift) state.
        result.replanned = True
        try:
            replan_solved, replan_actions = plan_for_mode(search_mode, generator, state, embedder)
        except Exception as exc:  # noqa: BLE001 -- a failed replan attempt must not crash the sweep
            result.error = f"replan_failed: {exc}"
            replan_solved, replan_actions = False, []

        if replan_solved:
            inner_state = state
            recovery_ok = True
            for action in replan_actions:
                if exec_checker.fresh_uncached_check(inner_state, action):
                    recovery_ok = False
                    break
                inner_state = _apply_and_advance_true_pose(action, inner_state, true_pose_table)
            result.replan_solved = recovery_ok and inner_state.is_goal()
        else:
            result.replan_solved = False

    # --- Pass 2: CACHE-ENABLED, cache-condition metrics only ---
    cache_true_pose_table = TruePoseTable(problem_id, sigma_m, cell_centers)
    cache_true_pose_table.init_from_cylinders(_cylinders_from_state(exec_checker, initial_state))
    exec_checker.clear_cache()
    exec_checker._state_manager.true_pose_table = cache_true_pose_table
    goal_state_symbolic = seed_cache_from_plan(exec_checker, initial_state, plan_actions)
    assert goal_state_symbolic.is_goal(), (
        f"{problem_name}: sigma=0 plan does not symbolically reach the goal -- baseline plan is broken"
    )

    sample_rng = np.random.default_rng(seed_from_problem_id(problem_id + ":stale_sample"))
    cache_state = initial_state
    cache_enabled_hit_infeasible = False
    for action in plan_actions:
        infeasible, was_hit = exec_checker.check_verbose(cache_state, action)
        if was_hit:
            result.cache_hits += 1
            if sample_rng.random() < stale_sample_rate:
                agree = exec_checker.record_stale_sample(cache_state, action, cached_result=infeasible)
                result.stale_checked += 1
                if not agree:
                    result.stale_disagree += 1
        else:
            result.cache_misses += 1
        if infeasible:
            cache_enabled_hit_infeasible = True
            break
        cache_state = _apply_and_advance_true_pose(action, cache_state, cache_true_pose_table)
    result.cache_enabled_solved = (not cache_enabled_hit_infeasible) and cache_state.is_goal()
    return result


# ---------------------------------------------------------------------------
# N2: occupancy flips -- simpler, no cache pre-seeding (see module docstring)
# ---------------------------------------------------------------------------

def execute_plan_n2(problem_name: str, search_mode: str, generator, embedder,
                     initial_state: xmi.XState, plan_actions: list, occupancy_flip_p: float,
                     seed_suffix: str = "") -> ExecutionResult:
    result = ExecutionResult(
        problem_name=problem_name, search_mode=search_mode, sigma_mm=0.0,
        solved_at_baseline=True, plan_len=len(plan_actions),
    )
    problem_id = f"{problem_name}{seed_suffix}"
    rng = np.random.default_rng(seed_from_problem_id(problem_id + ":n2"))
    exec_checker = _get_execution_checker()
    exec_checker.clear_cache()
    exec_checker.occupancy_flip_p = occupancy_flip_p
    exec_checker.occupancy_rng = rng
    exec_checker._state_manager.true_pose_table = None  # N2 alone: no placement drift

    state = initial_state
    for step, action in enumerate(plan_actions):
        infeasible, was_hit = exec_checker.check_verbose(state, action)
        result.cache_hits += int(was_hit)
        result.cache_misses += int(not was_hit)
        if infeasible:
            result.first_failure_step = step
            break
        state = action.apply(state)
    else:
        result.completed_without_replanning = True
        return result

    result.replanned = True
    try:
        replan_solved, _replan_actions = plan_for_mode(search_mode, generator, state, embedder)
    except Exception as exc:  # noqa: BLE001
        result.error = f"replan_failed: {exc}"
        return result
    result.replan_solved = replan_solved
    return result


# ---------------------------------------------------------------------------
# N3: execution failure -- action substitution, no checker changes
# ---------------------------------------------------------------------------

def _find_adjacent_put(generator, state: xmi.XState, cyl_name: str, occupied_cells: set, grid_adjacency: dict, exclude_cell: str, rng: np.random.Generator):
    """Find an already-enumerated `put(cyl_name, adjacent_empty_cell)`
    action of the state's own action generator (xmimir's action enumeration,
    so no fluents are synthesized by hand)."""
    candidates = [c for c in grid_adjacency.get(exclude_cell, ()) if c not in occupied_cells]
    if not candidates:
        return None
    rng.shuffle(candidates)
    enumerated = list(generator.action_generator.generate_actions(state))
    for target_cell in candidates:
        for a in enumerated:
            if a.name == "put" and a.objects[0].name == cyl_name and a.objects[1].name == target_cell:
                return a
    return None


def execute_plan_n3(problem_name: str, search_mode: str, generator, embedder,
                     initial_state: xmi.XState, plan_actions: list, failure_p: float,
                     seed_suffix: str = "") -> ExecutionResult:
    result = ExecutionResult(
        problem_name=problem_name, search_mode=search_mode, sigma_mm=0.0,
        solved_at_baseline=True, plan_len=len(plan_actions),
    )
    problem_id = f"{problem_name}{seed_suffix}"
    rng = np.random.default_rng(seed_from_problem_id(problem_id + ":n3"))
    exec_checker = _get_execution_checker()
    exec_checker.clear_cache()
    exec_checker.occupancy_flip_p = 0.0
    exec_checker._state_manager.true_pose_table = None  # N3 alone: no placement drift
    grid_adjacency = exec_checker._grid.adjacency

    state = initial_state
    for step, action in enumerate(plan_actions):
        if rng.random() < failure_p:
            robot_state = exec_checker._state_to_robot_state(state)
            occupied_cells = {c for cells in robot_state["cylinders"].values() for c in cells}
            substituted = None
            if action.name == "put":
                cyl_name = action.objects[0].name
                intended_cell = action.objects[1].name
                substituted = _find_adjacent_put(
                    generator, state, cyl_name, occupied_cells, grid_adjacency, intended_cell, rng,
                )
            # else (pick, or no adjacent cell free for put): grasp fails,
            # state unchanged -- do not apply anything this step.
            if substituted is not None:
                state = substituted.apply(state)
            # "state unchanged": just fall through without applying `action`.
            continue

        infeasible, was_hit = exec_checker.check_verbose(state, action)
        result.cache_hits += int(was_hit)
        result.cache_misses += int(not was_hit)
        if infeasible:
            result.first_failure_step = step
            break
        state = action.apply(state)

    if state.is_goal():
        result.completed_without_replanning = True
        return result
    if result.first_failure_step is None:
        # N3 substitutions consumed the whole plan without reaching goal
        # (never a checker-level "infeasible" -- record as a soft failure).
        result.first_failure_step = len(plan_actions)

    result.replanned = True
    try:
        replan_solved, _replan_actions = plan_for_mode(search_mode, generator, state, embedder)
    except Exception as exc:  # noqa: BLE001
        result.error = f"replan_failed: {exc}"
        return result
    result.replan_solved = replan_solved
    return result
