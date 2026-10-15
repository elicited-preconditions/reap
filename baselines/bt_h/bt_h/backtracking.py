"""Algorithm 1: classic backtracking, no learned classifier.

Recursively samples continuous/combinatorial parameters theta for each
ground abstract action in the task plan pi, runs the real controller, and
backtracks (tries the next sampled theta, or fails back to the parent
frame) on failure. Exponential in general (Sec 3.2 of Cieslar et al.).
Used (a) as plain backtracking, which the paper's appendix compares with
BT-h on the same plan orderings, and (b) to generate the seed
demonstrations of Algorithm 3's data collection (data_collection.py).

The budget is read from InstrumentedFeasibilityChecker.stats.total_calls
(a delta against calls_at_start): one checker instance serves many
searches (building tampanda's FrankaEnvironment is expensive), and the
count includes the controllers' re-checks (controller.py's
MAX_PICK_RETRIES/MAX_PUT_RETRIES).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from bt_h import config
from bt_h.checker import InstrumentedFeasibilityChecker
from bt_h.controller import run_pick, run_relocate
from bt_h.sampler import pick_thetas, relocate_thetas
from bt_h.types import GroundAction, SearchResult


class BudgetExhausted(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class SearchBudget:
    max_calls: int
    deadline: float
    calls_at_start: int

    def calls_used(self, checker: InstrumentedFeasibilityChecker) -> int:
        return checker.stats.total_calls - self.calls_at_start

    def check(self, checker: InstrumentedFeasibilityChecker) -> None:
        if self.calls_used(checker) >= self.max_calls:
            raise BudgetExhausted("max_budget")
        if time.time() > self.deadline:
            raise BudgetExhausted("timeout")


def sample_thetas(omega: GroundAction, robot_state: dict, rng: random.Random, max_tries: dict):
    if omega.kind == "pick":
        return pick_thetas()
    if omega.kind == "relocate":
        return relocate_thetas(robot_state, rng, max_tries=max_tries["relocate"])
    raise ValueError(f"unknown action kind: {omega.kind!r}")


def run_controller(robot_state: dict, omega: GroundAction, theta, checker: InstrumentedFeasibilityChecker):
    if omega.kind == "pick":
        return run_pick(robot_state, omega.cyl, checker)
    if omega.kind == "relocate":
        return run_relocate(robot_state, omega.cyl, theta, checker)
    raise ValueError(f"unknown action kind: {omega.kind!r}")


def default_max_tries() -> dict:
    return {"pick": config.MAX_TRIES_PICK, "relocate": config.MAX_TRIES_RELOCATE}


def _backtrack(
    sigma: list,
    pi: list,
    thetas_so_far: list,
    checker: InstrumentedFeasibilityChecker,
    rng: random.Random,
    budget: SearchBudget,
    max_tries: dict,
):
    if not pi:
        return sigma, thetas_so_far

    omega = pi[0]
    for theta in sample_thetas(omega, sigma[-1], rng, max_tries):
        budget.check(checker)
        s_next, ok, _info = run_controller(sigma[-1], omega, theta, checker)
        if not ok:
            continue
        result = _backtrack(sigma + [s_next], pi[1:], thetas_so_far + [theta], checker, rng, budget, max_tries)
        if result is not None:
            return result
    return None


def classic_backtrack(
    sigma0: dict,
    pi: list,
    checker: InstrumentedFeasibilityChecker,
    rng: random.Random | None = None,
    max_calls: int = 300,
    time_limit_s: float = 90.0,
    max_tries: dict | None = None,
) -> SearchResult:
    """Algorithm 1. Never raises -- budget exhaustion is caught internally
    and reported via SearchResult.termination_reason instead."""
    if rng is None:
        rng = random.Random()
    if max_tries is None:
        max_tries = default_max_tries()

    budget = SearchBudget(
        max_calls=max_calls,
        deadline=time.time() + time_limit_s,
        calls_at_start=checker.stats.total_calls,
    )
    start = time.time()
    try:
        result = _backtrack([sigma0], pi, [], checker, rng, budget, max_tries)
    except BudgetExhausted as exc:
        return SearchResult(
            solved=False,
            checker_calls=budget.calls_used(checker),
            wall_time_s=time.time() - start,
            termination_reason=exc.reason,
        )

    wall_time_s = time.time() - start
    if result is None:
        return SearchResult(
            solved=False,
            checker_calls=budget.calls_used(checker),
            wall_time_s=wall_time_s,
            termination_reason="exhausted",
        )
    sigma, thetas = result
    return SearchResult(
        solved=True,
        sigma=sigma,
        thetas=thetas,
        checker_calls=budget.calls_used(checker),
        wall_time_s=wall_time_s,
        termination_reason="solved",
    )
