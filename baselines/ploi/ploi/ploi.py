"""PLOI Algorithm 1 (Silver et al., AAAI 2021): the incremental object-set-growth
loop.

    score(o) = f(o, I, G) for all o in O          # one GNN forward pass
    for N = 1, 2, 3, ...:
        Ô = {o : score(o) >= gamma^N}
        Pi_hat = REDUCEPROBLEM(Pi, Ô)
        pi = PLAN(Pi_hat)
        if IsSolution(pi, Pi) or Ô == O: return pi

**Scored/prunable object universe O = cylinders only**, not cylinders+cells.
REAP's own dataset convention already counts "object count" as the number
of cylinder objects; the 70 grid cells are identical,
fixed structural scaffolding across every problem, not the kind of variable,
mostly-irrelevant population PLOI's pruning targets. Letting cells be
individually prunable would need O(84) sufficiency checks per problem and
produce spurious failures whenever an excluded cell is needed as a `put`
destination (see reduce_problem.py's module docstring). Cells still
appear as GNN input nodes for spatial/adjacency context -- they are just
never candidates for exclusion here. `Ô == O`
below therefore means "every cylinder is included," not "every object."

Two of the paper's own stated optimizations are implemented: (1) PLAN is
only called when Ô actually changed since the last attempt (skips redundant
replanning while the threshold keeps shrinking below already-included
objects' scores); (2) every object named in the goal is always assigned
score 1, guaranteeing it's included from N=1 (implemented by scorer callers
via reduce_problem.goal_object_names, applied here regardless of what the
scorer itself returned for those objects).

Practical completeness note (Lemma 1 in the paper assumes gamma^N -> 0 as
N -> infinity; a finite implementation cannot literally iterate to
infinity): once `max_iterations` is exhausted without Ô ever reaching the
full cylinder set (which would only happen if some cylinder's score
underflows to exactly 0.0 in floating point, or max_iterations is set too
low), a final PLAN attempt is made directly on the full cylinder set as an
explicit fallback -- this is what actually guarantees the completeness
Lemma 1 promises in a finite implementation, not merely a large
max_iterations.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import reap.xmimir as xmi

from ploi import reduce_problem, validate
from ploi.reduce_problem import CYLINDER_TYPE


@dataclass
class SufficiencyCheck:
    sufficient: bool
    plan_result: object  # plan_search.PlanResult -- always present, even when sufficient=False
    validation_reason: str


def test_object_set_sufficiency(
    full_problem_path: Path,
    domain_path: Path,
    keep_cylinders: set[str],
    full_problem,
    full_generator,
    plan_fn,
    plan_max_expansion: int,
    plan_time_limit_s: float,
    scratch_dir: Path,
) -> SufficiencyCheck:
    """IsSolution(PLAN(REDUCEPROBLEM(Pi, keep_cylinders)), Pi) -- Definition 2
    as a single reusable operation. Shared by run_ploi's incremental loop
    (below) and labels.py's greedy backward-removal search, both of which
    are literally repeated applications of this same check. REDUCEPROBLEM
    always keeps every cell regardless of keep_cylinders -- see
    reduce_problem.py."""
    reduced_path = scratch_dir / "reduced.pddl"
    reduce_problem.write_reduced_problem(full_problem_path, keep_cylinders, reduced_path)
    _, reduced_problem = xmi.parse(str(domain_path), str(reduced_path))
    reduced_generator = xmi.XSuccessorGenerator(reduced_problem)

    plan_result = plan_fn(
        reduced_generator, reduced_generator.initial_state,
        max_expansion=plan_max_expansion, time_limit_s=plan_time_limit_s,
    )
    if not plan_result.solved:
        return SufficiencyCheck(False, plan_result, f"PLAN did not solve the reduced problem ({plan_result.reason})")

    outcome = validate.is_solution(plan_result.plan, full_problem, full_generator)
    return SufficiencyCheck(outcome.solved, plan_result, outcome.reason)


@dataclass
class PloiResult:
    solved: bool
    plan: list | None
    reason: str
    n_plan_calls: int
    final_object_count: int  # cylinders kept in the successful/final attempt
    total_object_count: int  # total cylinders in the problem
    plan_results: list = field(default_factory=list)  # list[plan_search.PlanResult], one per PLAN() call


def run_ploi(
    full_problem_path: Path,
    domain_path: Path,
    scorer,
    plan_fn,
    gamma: float = 0.9,
    plan_max_expansion: int = 1000,
    plan_time_limit_s: float = 0,
    max_iterations: int = 60,
    scratch_dir: Path | None = None,
    overall_time_limit_s: float | None = None,
) -> PloiResult:
    """
    scorer: Callable[[xmi.XProblem], dict[str, float]] -- returns an
        importance score in (0, 1) for (at least) every CYLINDER name in the
        problem (scores for cell names, if any, are ignored -- cells are
        never pruned). Goal-named cylinders are force-set to 1.0 regardless
        of what the scorer returns for them (paper's own stated optimization).
    plan_fn: plan_search.bfs_with_plan or plan_search.gbfs_hff_with_plan
        (or any callable with the same (generator, initial_state,
        max_expansion, time_limit_s, filter_actions) -> PlanResult signature).
    overall_time_limit_s: the problem's whole wall-clock budget, counted from
        the start of this call. As in tomsilver/ploi's IncrementalPlanner
        (planning/incremental_planner.py: `time_elapsed = time.time()-start_time`,
        then `self._planner(domain, dummy_state, timeout-time_elapsed)`), every
        PLAN() call gets only the time left of it (at most plan_time_limit_s
        when that is > 0), and the loop stops once it is used up: the
        incremental loop's PLAN() calls (one per growing N) share one budget
        instead of each getting its own. A plain
        time.time() comparison, not a second SIGALRM -- filter_action already
        installs its own per-call SIGALRM timeout, and SIGALRM is a single
        process-wide OS timer, so nesting a second one around this whole
        function would be broken: the inner call's cleanup unconditionally
        cancels/resets the alarm, silently discarding an outer one's remaining
        time. None (default) means no overall budget (bounded only by
        max_iterations and each attempt's own plan_time_limit_s).
    """
    start_time = time.time()
    full_domain, full_problem = xmi.parse(str(domain_path), str(full_problem_path))
    full_generator = xmi.XSuccessorGenerator(full_problem)

    all_cylinders = {o.name for o in full_problem.objects if o.type_.name == CYLINDER_TYPE}
    scores = dict(scorer(full_problem))
    for name in reduce_problem.goal_object_names(full_problem):
        scores[name] = 1.0

    plan_results: list = []

    def _time_left() -> float | None:
        return None if overall_time_limit_s is None else overall_time_limit_s - (time.time() - start_time)

    def _out_of_time() -> bool:
        left = _time_left()
        return left is not None and left <= 0

    def _try_cylinder_set(keep_cylinders: set[str], tmp_dir: Path):
        # PLAN() gets the time left in the problem's budget (tomsilver/ploi:
        # `timeout-time_elapsed`), capped by plan_time_limit_s if that is set.
        # plan_search reads a limit <= 0 as "no limit", so a used-up budget
        # skips the call instead.
        limit, left = plan_time_limit_s, _time_left()
        if left is not None:
            if left <= 0:
                return None
            limit = left if limit <= 0 else min(limit, left)
        check = test_object_set_sufficiency(
            full_problem_path, domain_path, keep_cylinders, full_problem, full_generator,
            plan_fn, plan_max_expansion, limit, tmp_dir,
        )
        plan_results.append(check.plan_result)
        return check.plan_result.plan if check.sufficient else None

    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
        tmp_path = Path(tmp)
        previous_set: set[str] | None = None
        reached_full_set = False

        for n in range(1, max_iterations + 1):
            if _out_of_time():
                break

            threshold = gamma ** n
            keep_cylinders = {name for name in all_cylinders if scores.get(name, 0.0) >= threshold}

            if keep_cylinders == previous_set:
                if keep_cylinders == all_cylinders:
                    reached_full_set = True
                    break
                continue  # paper's optimization: skip PLAN() when Ô hasn't changed
            previous_set = keep_cylinders

            plan = _try_cylinder_set(keep_cylinders, tmp_path)
            if plan is not None:
                return PloiResult(
                    True, plan, "solved", len(plan_results), len(keep_cylinders), len(all_cylinders), plan_results,
                )
            if keep_cylinders == all_cylinders:
                reached_full_set = True
                break

        if not reached_full_set and not _out_of_time():
            # Finite-iteration completeness fallback (see module docstring).
            plan = _try_cylinder_set(all_cylinders, tmp_path)
            if plan is not None:
                return PloiResult(
                    True, plan, "solved_via_full_set_fallback", len(plan_results),
                    len(all_cylinders), len(all_cylinders), plan_results,
                )

    reason = "overall_timeout" if _out_of_time() else "no sufficient object set found within budget"
    return PloiResult(
        False, None, reason,
        len(plan_results), len(previous_set or all_cylinders), len(all_cylinders), plan_results,
    )
