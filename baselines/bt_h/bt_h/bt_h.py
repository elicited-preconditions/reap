"""Algorithm 2: BT-h (backtracking with feasibility-classifier veto and
confidence-guided backjumping), plus a capture-tree variant used by
Algorithm 3's data collection to harvest both positive and negative
training examples from a single bounded search.

Implements the paper's pseudocode.

`conf_min` threads through the recursion as a single
scalar -- the minimum classifier confidence along the path of COMMITTED
(accepted, recursed-into) ancestors, updated via `min(conf_min, conf)` on
each recursive call (Line 10). After a child call fails, Lines 13-14 check
whether THIS attempt's own just-computed confidence exceeded the INHERITED
conf_min (the value passed into this call, i.e. the ancestor path's own
worst point, NOT updated by sibling attempts at this same level): if so,
give up on this entire level immediately (skip remaining retries, return
failure fast); if not (this attempt's confidence was no better than the
known worst link, i.e. it might BE the culprit), keep trying alternative
thetas here instead. Backjumping emerges from these purely LOCAL,
per-attempt early returns cascading up the call stack as each ancestor
performs the identical check -- no explicit multi-level jump signal is
needed.

With a constant-1 classifier (conf always 1.0,
initial conf_min also 1.0), `conf > conf_min` (1.0 > 1.0) is never true, so
the early-return at Lines 13-14 never fires -- every level exhausts its own
r_iter budget before failing upward, which is exactly classic_backtrack's
behavior.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from bt_h.backtracking import (
    BudgetExhausted,
    SearchBudget,
    default_max_tries,
    run_controller,
    sample_thetas,
)
from bt_h.checker import InstrumentedFeasibilityChecker
from bt_h.types import GroundAction, SearchResult

INITIAL_CONF_MIN = 1.0  # Algorithm 3's own call: BT-h-CaptureSearchTree(sigma_pref, pi, 1.0, h)


@dataclass
class SearchTree:
    """Output of bt_h_capture_tree (Algorithm 3's Line 9): every (sigma,
    pi_full-suffix-position) pair reached is classified as either a
    successful leaf (full plan completion) or a stuck leaf (budget for that
    node exhausted without completing) -- see data_collection.py for how
    these become positive/negative training examples."""

    successes: list = field(default_factory=list)   # list[list[dict]] -- each a full sigma sequence
    stuck: list = field(default_factory=list)         # list[list[dict]] -- each a sigma sequence that dead-ended


def _bt_h(
    sigma: list,
    pi_full: list,
    depth: int,             # == len(sigma) - 1: number of actions applied so far
    thetas_so_far: list,
    conf_min: float,         # scalar: min classifier confidence along the committed ancestor path
    checker: InstrumentedFeasibilityChecker,
    classifier,               # callable(sigma, pi_full) -> float in [0, 1]
    rng: random.Random,
    budget: SearchBudget,
    tau: float,
    max_tries: dict,
):
    if depth == len(pi_full):
        return sigma, thetas_so_far  # Line 3: i=n, return <> (base case success)

    omega = pi_full[depth]
    for theta in sample_thetas(omega, sigma[-1], rng, max_tries):  # Lines 4-5
        budget.check(checker)
        s_next, ok, _info = run_controller(sigma[-1], omega, theta, checker)  # Line 6
        if not ok:
            continue  # Line 7: s_{i+1} = bottom, continue -- no classifier query needed

        sigma_next = sigma + [s_next]
        conf = classifier(sigma_next, pi_full)  # Line 8: eta, conf := h(pi, sigma (+) <s_next>)
        if conf < tau:
            continue  # Line 9: if not eta then continue (eta = conf >= tau)

        result = _bt_h(  # Line 10
            sigma_next, pi_full, depth + 1, thetas_so_far + [theta],
            min(conf_min, conf), checker, classifier, rng, budget, tau, max_tries,
        )
        if result is not None:
            return result  # Lines 11-12: if p != bottom, return p (+) <theta_{i+1}>

        # Lines 13-14: child failed. If THIS attempt's own confidence beat
        # the ancestor path's worst point, the culprit isn't here -- give up
        # on this whole level immediately rather than trying more thetas.
        if conf > conf_min:
            return None
        # else: conf <= conf_min -- this might be the culprit; keep trying
        # alternative thetas at this level (fall through to next iteration).

    return None  # Line 15: loop exhausted without success


def bt_h_search(
    sigma_prefix,
    pi_full: list,
    checker: InstrumentedFeasibilityChecker,
    classifier,
    rng: random.Random | None = None,
    max_calls: int = 300,
    time_limit_s: float = 90.0,
    max_tries: dict | None = None,
    tau: float = 0.5,
) -> SearchResult:
    """Algorithm 2. `sigma_prefix` may be a single initial state dict (the
    common case) or an already-partially-refined list of states -- Algorithm
    3's descending-prefix-length data-collection loop (data_collection.py)
    needs to restart search from an arbitrary point l in a real
    demonstration, not always from sigma_0.

    Never raises -- budget exhaustion is caught internally and reported via
    SearchResult.termination_reason instead."""
    if rng is None:
        rng = random.Random()
    if max_tries is None:
        max_tries = default_max_tries()

    sigma0 = sigma_prefix if isinstance(sigma_prefix, list) else [sigma_prefix]
    start_depth = len(sigma0) - 1

    budget = SearchBudget(
        max_calls=max_calls,
        deadline=time.time() + time_limit_s,
        calls_at_start=checker.stats.total_calls,
    )
    start = time.time()
    try:
        result = _bt_h(
            sigma0, pi_full, start_depth, [], INITIAL_CONF_MIN,
            checker, classifier, rng, budget, tau, max_tries,
        )
    except BudgetExhausted as exc:
        return SearchResult(
            solved=False, checker_calls=budget.calls_used(checker),
            wall_time_s=time.time() - start, termination_reason=exc.reason,
        )

    wall_time_s = time.time() - start
    if result is None:
        return SearchResult(
            solved=False, checker_calls=budget.calls_used(checker),
            wall_time_s=wall_time_s, termination_reason="exhausted",
        )
    sigma, thetas = result
    return SearchResult(
        solved=True, sigma=sigma, thetas=thetas,
        checker_calls=budget.calls_used(checker),
        wall_time_s=wall_time_s, termination_reason="solved",
    )


def _capture_tree(
    sigma: list,
    pi_full: list,
    depth: int,
    checker: InstrumentedFeasibilityChecker,
    classifier,
    rng: random.Random,
    budget: SearchBudget,
    tau: float,
    max_tries: dict,
    tree: SearchTree,
) -> bool:
    """Returns whether `sigma` itself lies on at least one successful path
    discovered within this capture-tree call -- the CALLER uses this to
    decide whether to record `sigma` as a positive or negative example
    (Appendix B / Figure B.1 of the paper: "all nodes and corresponding
    prefixes of the plan are positive samples" along a successful path, not
    just the terminal leaf; "the two nodes... exemplify the negative
    samples that lead to refinement failure" -- i.e. true dead-ends, nodes
    with NO viable continuation at all).

    Unlike _bt_h, this does NOT stop at the first success and does NOT
    backjump/early-return -- it keeps sampling every theta at every node up
    to its own max_tries budget (both after a success and after a failure),
    so both successful and stuck nodes get harvested, not just one of each.
    BT-h's early return on a confident failure would skip branches that
    give negative examples."""
    if depth == len(pi_full):
        return True  # the goal state itself is trivially "successful"

    omega = pi_full[depth]
    any_child_succeeded = False
    for theta in sample_thetas(omega, sigma[-1], rng, max_tries):
        try:
            budget.check(checker)
        except BudgetExhausted:
            break
        s_next, ok, _info = run_controller(sigma[-1], omega, theta, checker)
        if not ok:
            continue
        sigma_next = sigma + [s_next]
        conf = classifier(sigma_next, pi_full)
        if conf < tau:
            # Classifier already confident either way -- skip WITHOUT
            # labeling. sigma_next is a real, physically feasible state, but
            # spending budget confirming what the classifier already
            # believes teaches it nothing new; labeling it here would just
            # reinforce a possibly-wrong belief with no new evidence.
            continue
        child_succeeded = _capture_tree(
            sigma_next, pi_full, depth + 1, checker, classifier, rng, budget, tau, max_tries, tree,
        )
        if child_succeeded:
            tree.successes.append(sigma_next)
            any_child_succeeded = True
        else:
            tree.stuck.append(sigma_next)

    return any_child_succeeded


def bt_h_capture_tree(
    sigma_prefix,
    pi_full: list,
    checker: InstrumentedFeasibilityChecker,
    classifier,
    rng: random.Random | None = None,
    max_calls: int = 150,
    time_limit_s: float = 45.0,
    max_tries: dict | None = None,
    tau: float = 0.5,
) -> SearchTree:
    """Algorithm 3, Line 9 (BT-h-CaptureSearchTree). `sigma_prefix` may be a
    single initial state dict or an already-partially-refined list of
    states -- Algorithm 3's descending-prefix-length loop
    (data_collection.py) restarts this from sigma_l, a real prefix cut out
    of a solved demonstration, not always from sigma_0.

    Returns a SearchTree; swallows BudgetExhausted internally (a global cap
    on the whole capture call, separate from any per-node bookkeeping)
    since a partially explored tree is still useful training data, not a
    failure to report."""
    if rng is None:
        rng = random.Random()
    if max_tries is None:
        max_tries = default_max_tries()

    sigma0 = sigma_prefix if isinstance(sigma_prefix, list) else [sigma_prefix]
    start_depth = len(sigma0) - 1

    budget = SearchBudget(
        max_calls=max_calls,
        deadline=time.time() + time_limit_s,
        calls_at_start=checker.stats.total_calls,
    )
    tree = SearchTree()
    try:
        _capture_tree(sigma0, pi_full, start_depth, checker, classifier, rng, budget, tau, max_tries, tree)
    except BudgetExhausted:
        pass
    return tree
