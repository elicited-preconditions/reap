# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from __future__ import annotations

from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Generator, Iterable, Union

import pymimir.advanced as adv
from multimethod import multimethod
from pymimir import Domain, Problem

import reap.xmimir.pymimir_wrappers as wrapmir

from .wrappers import XAction, XAtom, XDomain, XObject, XProblem, XState


def parse(
    domain: Union[str, Path, Domain, XDomain],
    *problems: Union[str, Path, Problem, XProblem],
    raise_on_error: bool = True,
) -> tuple[XDomain, list[XProblem | Exception]] | tuple[XDomain, XProblem | Exception]:
    """
    Parse the problem into a domain, problem, and repositories.
    """
    match domain:
        case Domain():
            domain = XDomain(domain).filepath
        case XDomain():
            domain = domain.filepath
        case Path():
            domain = str(domain.absolute())
        case str():
            domain = domain
        case _:
            raise ValueError(f"Unknown domain type: {type(domain)}")

    options = adv.formalism.ParserOptions()
    parser = adv.formalism.Parser(domain, options)

    domain = XDomain(parser.get_domain(), parser)

    problem_objs = []
    for problem in problems:
        match problem:
            case Problem():
                problem = XProblem(problem).filepath
            case XProblem():
                problem = problem.filepath
            case Path():
                problem = str(problem.absolute())
            case str():
                problem = problem
            case _:
                raise ValueError(f"Unknown problem type: {type(problem)}")
        try:
            prob = XProblem(wrapmir.Problem(domain.base, problem))
        except RuntimeError as e:
            if raise_on_error:
                raise e
            else:
                prob = e
        problem_objs.append(prob)

    return domain, *problem_objs


def parse_domain(domain: Union[str, Path, Domain, XDomain]) -> XDomain:
    match domain:
        case Domain():
            domain = XDomain(domain).filepath
        case XDomain():
            domain = domain.filepath
        case Path():
            domain = str(domain.absolute())
    _domain = Domain(domain)
    return XDomain(_domain)


@multimethod
def gather_objects(problem: XProblem) -> set[XObject]:
    return set(problem.objects)


@multimethod
def gather_objects(
    atoms: Iterable[XAtom | XAction] | XState | Generator,
) -> set[XObject]:
    return set(chain.from_iterable(atom.objects for atom in atoms))


def parse_atom_string(atom_str: str):
    # Basic parse for form: (predicate arg1 arg2 ...)
    atom_str = atom_str.strip()
    assert atom_str.startswith("(") and atom_str.endswith(")")
    parts = atom_str[1:-1].split()
    predicate = parts[0]
    objects = []
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            objects.append({k: v})
        else:
            objects.append(part)
    return predicate, objects


class StateType(Enum):
    DEFAULT = 0
    GOAL = 1
    INITIAL = 2
    DEADEND = 3


__all__ = ["parse", "gather_objects", "StateType", "parse_atom_string"]
