"""REDUCEPROBLEM (PLOI paper Definition 1): given a full PDDL problem file and
a CYLINDER-name subset, produce a reduced problem text containing only that
subset's cylinders -- but ALWAYS all 70 grid cells.

Cells are treated as fixed structural/geometric scaffolding, not PLOI-scored
objects: they are identical across every problem, and REAP's own dataset
convention counts "object count" as cylinders only. Letting cells be
individually pruned (like cylinders) would need an O(84) (70 cells + <=14
cylinders) greedy removal search per training problem and cause wrong
results: excluding a specific empty cell can starve a `put` action of any
valid destination, so a plan that is findable with the full cell set would
spuriously fail with an arbitrary cell missing, inflating attempt costs and
label noise alike. Restricting pruning to cylinders alone cuts attempts
from O(84) to O(<=24) and removes that failure mode entirely (see below).

xmimir's XProblem/XDomain expose a parser (xmi.parse) but no serializer, so
round-tripping through xmimir's parsed representation would still require
hand-writing a PDDL emitter -- this instead does regex-based text surgery
directly on the original file, whose format is extremely regular (one atom
per line, single :objects/:init/:goal blocks, as in REAP's dataset files).
"""
from __future__ import annotations

import re
from pathlib import Path

_OBJ_GROUP_RE = re.compile(r"([\w\s]+?)\s*-\s*(\w+)")
_ATOM_RE = re.compile(r"\(([^()]*)\)")

# The domain's only two `direction` constants (:constants north east -
# direction in domain.pddl) -- these appear as the first token of `adjacent`
# atoms, never as a declared :objects entry.
DOMAIN_CONSTANTS = frozenset({"north", "east"})

CYLINDER_TYPE = "cylinder"
CELL_TYPE = "cell"


def _objects_block_text(text: str) -> str:
    match = re.search(r":objects\s*(.*?)\)\s*\(:init", text, re.S)
    if match is None:
        raise ValueError("could not find :objects block")
    return match.group(1)


def _init_block_text(text: str) -> str:
    match = re.search(r":init\s*(.*?)\)\s*\(:goal", text, re.S)
    if match is None:
        raise ValueError("could not find :init block")
    return match.group(1)


def _goal_block_text(text: str) -> str:
    match = re.search(r"(\(:goal\b.*?\))\s*\)\s*(?:;|\Z)", text, re.S)
    if match is None:
        raise ValueError("could not find :goal block")
    return match.group(1)


def parse_objects(text: str) -> list[tuple[str, str]]:
    """Returns [(object_name, type_name), ...] in declaration order."""
    out: list[tuple[str, str]] = []
    for names_str, type_name in _OBJ_GROUP_RE.findall(_objects_block_text(text)):
        out.extend((n, type_name) for n in names_str.split())
    return out


def parse_init_atoms(text: str) -> list[str]:
    """Returns each :init atom as its own '(...)' string, e.g. '(occupied cell_5_2 cylinder_1)'."""
    return [f"({inner.strip()})" for inner in _ATOM_RE.findall(_init_block_text(text))]


def goal_objects(text: str) -> set[str]:
    """Object names referenced anywhere in the :goal block."""
    goal_text = _goal_block_text(text)
    names: set[str] = set()
    for inner in _ATOM_RE.findall(goal_text):
        tokens = inner.strip().split()
        # tokens[0] is either a predicate name or a logical connective (and/or/not)
        names.update(tok for tok in tokens[1:] if tok not in ("and", "or", "not"))
    return names


def goal_object_names(problem) -> set[str]:
    """Object names referenced in an already-parsed xmimir XProblem's goal
    (via problem.literal_goal()) -- the xmimir-native equivalent of
    goal_objects(text) above, for callers (ploi.py) that already hold a
    parsed XProblem and shouldn't re-read/re-parse the PDDL text."""
    names: set[str] = set()
    for lit in problem.literal_goal():
        names.update(o.name for o in lit.atom.objects)
    return names


def reduce_problem_text(problem_path: Path, keep_cylinders: set[str]) -> str:
    """REDUCEPROBLEM(Pi, keep_cylinders): keep ALL cell objects always, keep
    only the given subset of cylinder objects. Definition 1 is applied
    literally: an `(occupied cell cyl)` atom whose cylinder is excluded is
    simply DROPPED (its arguments aren't all in the kept set), not rewritten
    to `(empty cell)` -- that would assert something false (the excluded
    cylinder is still physically/symbolically really there; REDUCEPROBLEM
    restricts what the planner *sees*, it does not get to invent new facts).
    Synthesizing `(empty cell)` here would let the reduced-problem search
    plan to `put` a kept cylinder onto a cell that a real, excluded cylinder
    still occupies, which IsSolution then rejects when replaying against
    the full problem's true state. Enough empty `put` destinations remain
    anyway, since cells are always kept: every
    cell that's ALREADY empty in the original problem (not dependent on any
    cylinder's inclusion status) remains available in every reduction,
    regardless of which cylinders are excluded.

    The :goal is left untouched -- callers must ensure every goal-named
    cylinder is already in keep_cylinders (PLOI's own stated optimization:
    goal objects always score 1); this function fails loudly rather than
    silently emitting a goal that references an excluded cylinder.
    """
    text = Path(problem_path).read_text()

    name_match = re.search(r"\(define\s*\(problem\s+(\S+)\)", text)
    domain_match = re.search(r"\(:domain\s+(\S+)\)", text)
    if name_match is None or domain_match is None:
        raise ValueError(f"{problem_path}: could not find problem/domain name")

    all_objects = parse_objects(text)
    all_cylinders = {n for n, t in all_objects if t == CYLINDER_TYPE}
    all_cells = {n for n, t in all_objects if t == CELL_TYPE}

    missing = keep_cylinders - all_cylinders
    if missing:
        raise ValueError(f"keep_cylinders references non-cylinder or unknown names in {problem_path}: {sorted(missing)}")

    needed_goal_objects = goal_objects(text)
    unmet = (needed_goal_objects & all_cylinders) - keep_cylinders
    if unmet:
        raise ValueError(
            f"keep_cylinders excludes cylinder(s) referenced in the goal: {sorted(unmet)} "
            "-- PLOI always scores goal-named objects 1, so a correct caller should "
            "never hit this; failing loudly rather than emitting an unsolvable reduction."
        )

    kept_names = all_cells | keep_cylinders
    by_type: dict[str, list[str]] = {CELL_TYPE: sorted(all_cells), CYLINDER_TYPE: sorted(keep_cylinders)}
    objects_block = "\n    ".join(f"{' '.join(names)} - {t}" for t, names in by_type.items() if names)

    kept_atoms: list[str] = []
    for atom_text in parse_init_atoms(text):
        tokens = atom_text.strip("()").split()
        args = tokens[1:]
        if all(tok in kept_names or tok in DOMAIN_CONSTANTS for tok in args):
            kept_atoms.append(atom_text)
        # else: references an excluded cylinder (`occupied` at that
        # cylinder's cell, or in principle `holding` it, though the gripper
        # always starts empty in this dataset's initial states) -- dropped,
        # per Definition 1's "only properties over the kept set." The
        # cylinder's former cell simply carries no occupied/empty atom at
        # all in the reduced problem; see the module docstring for why this
        # is correct and not a gap.

    init_block = "\n    ".join(kept_atoms)
    goal_block = _goal_block_text(text)

    return (
        f"(define (problem {name_match.group(1)}_reduced)\n"
        f"  (:domain {domain_match.group(1)})\n\n"
        f"  (:objects\n    {objects_block}\n  )\n\n"
        f"  (:init\n    {init_block}\n  )\n\n"
        f"  {goal_block}\n"
        f")\n"
    )


def write_reduced_problem(problem_path: Path, keep_cylinders: set[str], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(reduce_problem_text(problem_path, keep_cylinders))
    return out_path
