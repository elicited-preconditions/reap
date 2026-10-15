# single_plan_touched_objects() is ported from github.com/tomsilver/ploi (guidance/gnn_guidance.py,
# _collect_training_data). MIT License, Copyright (c) 2020 Tom Silver and Rohan Chitnis; the notice is in
# LICENSES/ploi-MIT.txt.
"""PLOI training labels, two strategies (generate_training_labels.py --strategy):

- single_plan (`single_plan_touched_objects`, the labels of the reported weights): as in the authors' code, a cylinder
  is important if a plan for the full problem touches it; problems without such a plan are dropped.
- greedy (`greedy_backward_removal`): the greedy backward removal of the PLOI paper (Section 4, "Training with
  Supervised Learning").

Only cylinders are labeled: cells are always kept by REDUCEPROBLEM (reduce_problem.py), so `TrainingExample.labels`
only contains cylinders (see ploi.py's module docstring).

Greedy backward removal: starting with the full CYLINDER set, iteratively remove an individual
cylinder, accepting the new set if it is sufficient (i.e.
IsSolution(PLAN(REDUCEPROBLEM(Pi, C')), Pi) still holds), until no more
individual cylinders can be removed without violating sufficiency.

Implemented as a SINGLE pass over objects in a fixed order, each tested for
removal against the CURRENT (already-shrunk) set exactly once -- not a
repeated until-fixpoint pass. This is sound (not merely an optimization)
for this domain: REDUCEPROBLEM only ever removes atoms, so
REDUCEPROBLEM(Pi, O''\\{x}) subset-of REDUCEPROBLEM(Pi, O'\\{x}) whenever
O'' subset-of O' -- a strict object-count reduction can only shrink PLAN's
own search space, never unlock a NEW valid action sequence. So once removing
x has been tested and rejected against some set, testing it again later
against an even-smaller set cannot newly succeed (modulo the physical
checker's run-to-run stochasticity). This turns what could naively be an
O(|O|^2) repeated procedure into O(|O|) sufficiency checks, each one a full
PLAN() call.

Goal-named objects are never attempted for removal (PLOI's own stated
optimization: they always score 1, so a correct label would never exclude
them anyway).

ATTEMPT ORDER: sorted cylinder name order (a free choice; the paper does
not specify one).

Checkpointed every `checkpoint_every` attempts to `<out_dir>/<stem>.partial.json`
(current_set + already_tried) so a single problem's multi-minute search can
resume from its last checkpoint rather than restarting from the full object
set.

REFERENCE-PLAN FALLBACK: if PLAN cannot solve the FULL, unreduced object
set within the per-attempt budget, no removal attempt can succeed (PLAN
cannot solve any subset either), and the loop would label every object as
needed. So `greedy_backward_removal` first tries the full set; if that
fails and the problem's reference `.plan` file exists (shipped with every
training problem), the label is the set of objects that plan touches (the
target and every cylinder/cell of its pick/put actions): a sufficient,
not necessarily minimal set.
"""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import reap.xmimir as xmi

from ploi import reduce_problem
from ploi.dataset import TrainingExample
from ploi.ploi import test_object_set_sufficiency
from ploi.reduce_problem import CYLINDER_TYPE


@dataclass
class LabelSearchStats:
    n_objects: int
    n_attempts: int
    n_accepted_removals: int
    wall_time_s: float
    n_plan_calls: int  # one PLAN() call per sufficiency test, plus the upfront full-set presolve
    used_reference_plan_fallback: bool = False


_ACTION_LINE_TOKENS = ("pick", "put")


def parse_reference_plan_objects(plan_path: Path, target_name: str) -> set[str] | None:
    """Object names touched by a REAP reference .plan file's action lines
    (not its trailing `; comment` metadata block) -- target's own name is
    always included even if, degenerately, the plan is a single `(pick
    <target> <cell>)` line that doesn't otherwise repeat it."""
    if not plan_path.exists():
        return None
    touched = {target_name}
    for line in plan_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("(") or not line.endswith(")"):
            continue
        parts = line.strip("()").split()
        if len(parts) == 3 and parts[0] in _ACTION_LINE_TOKENS:
            touched.add(parts[1])  # cylinder
            touched.add(parts[2])  # cell
    return touched


def _load_checkpoint(path: Path) -> tuple[set[str], set[str]] | None:
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    return set(state["current_set"]), set(state["already_tried"])


def _save_checkpoint(path: Path, current_set: set[str], already_tried: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"current_set": sorted(current_set), "already_tried": sorted(already_tried)}))
    tmp.replace(path)  # atomic-ish: never leaves a half-written checkpoint on disk


def greedy_backward_removal(
    problem_path: Path,
    domain_path: Path,
    plan_fn,
    plan_max_expansion: int,
    plan_time_limit_s: float,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 10,
    scratch_dir: Path | None = None,
    reference_plan_path: Path | None = None,
) -> tuple[dict[str, int], LabelSearchStats]:
    start = time.time()
    _, problem = xmi.parse(str(domain_path), str(problem_path))
    full_generator = xmi.XSuccessorGenerator(problem)

    all_cylinders = sorted(o.name for o in problem.objects if o.type_.name == CYLINDER_TYPE)
    goal_names = reduce_problem.goal_object_names(problem)
    attempt_order = [n for n in all_cylinders if n not in goal_names]

    resumed = checkpoint_path is not None and _load_checkpoint(checkpoint_path) is not None
    if resumed:
        current_set, already_tried = _load_checkpoint(checkpoint_path)
    else:
        current_set, already_tried = set(all_cylinders), set()

    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
        tmp_path = Path(tmp)

        # Upfront presolve (skipped on resume -- a checkpoint only exists if
        # this already passed once before): if PLAN can't even solve the
        # FULL object set within budget, no removal attempt can possibly
        # succeed either -- fall back to the reference plan's own touched
        # objects rather than burning the whole attempt loop on a foregone
        # conclusion. See module docstring "REFERENCE-PLAN FALLBACK".
        if not resumed:
            presolve = test_object_set_sufficiency(
                problem_path, domain_path, current_set, problem, full_generator,
                plan_fn, plan_max_expansion, plan_time_limit_s, tmp_path,
            )
            if not presolve.sufficient:
                target = next(iter(goal_names)) if goal_names else None
                fallback = (
                    parse_reference_plan_objects(reference_plan_path, target)
                    if reference_plan_path is not None and target is not None else None
                )
                if fallback is not None:
                    # parse_reference_plan_objects returns both cylinder AND
                    # cell names touched by the plan (each pick/put action's
                    # two arguments) -- only the cylinder portion is a label
                    # here, cells are never labeled (see module docstring).
                    fallback_cylinders = fallback & set(all_cylinders)
                    stats = LabelSearchStats(
                        n_objects=len(all_cylinders), n_attempts=1, n_accepted_removals=0,
                        wall_time_s=time.time() - start, n_plan_calls=1,
                        used_reference_plan_fallback=True,
                    )
                    labels = {name: int(name in fallback_cylinders) for name in all_cylinders}
                    return labels, stats
                # no reference plan: run the attempt loop (its label keeps
                # every object it cannot remove)

        n_attempts_since_checkpoint = 0
        n_accepted = 0
        for name in attempt_order:
            if name in already_tried or name not in current_set:
                continue

            candidate_set = current_set - {name}
            check = test_object_set_sufficiency(
                problem_path, domain_path, candidate_set, problem, full_generator,
                plan_fn, plan_max_expansion, plan_time_limit_s, tmp_path,
            )
            if check.sufficient:
                current_set = candidate_set
                n_accepted += 1
            already_tried.add(name)

            n_attempts_since_checkpoint += 1
            if checkpoint_path is not None and n_attempts_since_checkpoint >= checkpoint_every:
                _save_checkpoint(checkpoint_path, current_set, already_tried)
                n_attempts_since_checkpoint = 0

    stats = LabelSearchStats(
        n_objects=len(all_cylinders),
        n_attempts=len(already_tried),
        n_accepted_removals=n_accepted,
        wall_time_s=time.time() - start,
        n_plan_calls=len(already_tried),
    )
    labels = {name: int(name in current_set) for name in all_cylinders}
    return labels, stats


def single_plan_touched_objects(
    problem_path: Path,
    domain_path: Path,
    plan_fn,
    plan_max_expansion: int,
    plan_time_limit_s: float,
) -> tuple[dict[str, int], LabelSearchStats] | None:
    """The labelling of the authors' code (github.com/tomsilver/ploi,
    gnn_guidance.py::_collect_training_data; MIT license, Copyright (c) 2020
    Tom Silver and Rohan Chitnis, LICENSES/ploi-MIT.txt), the labels of the
    reported weights: run PLAN once on the full, unreduced problem
    (`self._planner(env.domain, state, timeout=60)`, then
    `objects_in_plan = {o for act in plan for o in act.variables}`) and label
    every object the plan touches as positive, every other one as negative.

    As in their code, a problem that PLAN cannot solve within the budget is
    left out of the training set: this function returns None (no
    reference-plan fallback)."""
    start = time.time()
    _, problem = xmi.parse(str(domain_path), str(problem_path))
    generator = xmi.XSuccessorGenerator(problem)
    all_cylinders = sorted(o.name for o in problem.objects if o.type_.name == CYLINDER_TYPE)

    plan_result = plan_fn(
        generator, generator.initial_state,
        max_expansion=plan_max_expansion, time_limit_s=plan_time_limit_s,
    )
    if not plan_result.solved:
        return None

    touched = {o.name for action in plan_result.plan for o in action.objects}
    labels = {name: int(name in touched) for name in all_cylinders}
    stats = LabelSearchStats(
        n_objects=len(all_cylinders), n_attempts=1, n_accepted_removals=0,
        wall_time_s=time.time() - start, n_plan_calls=1,
    )
    return labels, stats


def label_problem_single_plan(
    problem_path: Path,
    domain_path: Path,
    plan_fn,
    out_dir: Path,
    plan_max_expansion: int = 5000,
    plan_time_limit_s: float = 120.0,
) -> tuple[TrainingExample, LabelSearchStats | None] | None:
    """Same skip-if-exists resumability as `label_problem`, but for
    `single_plan_touched_objects`. Returns None (not written, not cached) if
    PLAN can't solve the full problem within budget (the problem is left out,
    as in the authors' code; see that function's docstring)."""
    stem = Path(problem_path).stem
    out_path = out_dir / f"{stem}.json"

    if out_path.exists():
        example = TrainingExample.from_dict(json.loads(out_path.read_text()))
        return example, None

    _, problem = xmi.parse(str(domain_path), str(problem_path))
    goal_names = reduce_problem.goal_object_names(problem)
    target = next(iter(goal_names)) if len(goal_names) == 1 else sorted(goal_names)[0]

    result = single_plan_touched_objects(
        problem_path, domain_path, plan_fn, plan_max_expansion, plan_time_limit_s,
    )
    if result is None:
        return None
    labels, stats = result

    example = TrainingExample(
        problem_path=str(problem_path), domain_path=str(domain_path),
        labels=labels, target_cylinder=target,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(example.to_dict()))
    tmp.replace(out_path)
    return example, stats


def label_problem(
    problem_path: Path,
    domain_path: Path,
    plan_fn,
    out_dir: Path,
    plan_max_expansion: int = 200,
    plan_time_limit_s: float = 60.0,
    checkpoint_every: int = 10,
    scratch_dir: Path | None = None,
    reference_plan_path: Path | None = "auto",
) -> tuple[TrainingExample, LabelSearchStats] | tuple[TrainingExample, None]:
    """Skips (returns the already-computed example, stats=None) if
    `<out_dir>/<stem>.json` already exists (save-immediately/skip-if-exists
    resumability, with the finer per-checkpoint layer above for the higher
    per-problem cost).

    reference_plan_path: "auto" (default) looks for REAP's own convention of
    a `<problem_path>.plan` file sitting alongside the problem file (present
    for every real REAP training/eval problem); pass an explicit Path to
    override, or None to disable the fallback entirely (hand-built instances
    with no reference solution)."""
    stem = Path(problem_path).stem
    out_path = out_dir / f"{stem}.json"
    checkpoint_path = out_dir / f"{stem}.partial.json"

    if out_path.exists():
        example = TrainingExample.from_dict(json.loads(out_path.read_text()))
        return example, None

    if reference_plan_path == "auto":
        reference_plan_path = Path(str(problem_path) + ".plan")

    _, problem = xmi.parse(str(domain_path), str(problem_path))
    goal_names = reduce_problem.goal_object_names(problem)
    target = next(iter(goal_names)) if len(goal_names) == 1 else sorted(goal_names)[0]

    labels, stats = greedy_backward_removal(
        problem_path, domain_path, plan_fn, plan_max_expansion, plan_time_limit_s,
        checkpoint_path=checkpoint_path, checkpoint_every=checkpoint_every, scratch_dir=scratch_dir,
        reference_plan_path=reference_plan_path,
    )
    example = TrainingExample(
        problem_path=str(problem_path), domain_path=str(domain_path),
        labels=labels, target_cylinder=target,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(example.to_dict()))
    tmp.replace(out_path)
    checkpoint_path.unlink(missing_ok=True)
    return example, stats
