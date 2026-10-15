"""psi_omega: PickSampler (degenerate) and RelocateSampler (uniform over
currently-free cells in the REAL continuous state, without replacement
across a single search node's sampling attempts).

RelocateSampler samples uniformly among the free cells, as the paper's
appendix states; it has no bias away from cells that create a new occluder,
which is what the classifier has to learn.
"""
from __future__ import annotations

import random
from typing import Iterator

from bt_h import config
from bt_h.grid import all_cells
from bt_h.state import free_cells


def pick_thetas() -> Iterator[None]:
    """Degenerate sampler: pick has no real continuous parameter -- grasp
    candidate selection is resolved internally, unexposed, by tampanda's
    checker (grasping always uses a fixed FRONT approach). Yields
    None exactly once; a pick failure is a real infeasibility, not
    something to search over multiple parameter choices for."""
    yield None


def relocate_thetas(
    robot_state: dict,
    rng: random.Random,
    max_tries: int = config.MAX_TRIES_RELOCATE,
    exclude: frozenset[str] = frozenset(),
) -> Iterator[str]:
    """Uniform, without-replacement sampling of candidate free cells for a
    relocate(cyl) step, up to max_tries candidates. `exclude` lets a caller
    (backtracking.py/bt_h.py) skip cells already rejected earlier at the
    same search node, in case this generator is re-invoked mid-search."""
    candidates = [c for c in free_cells(robot_state, all_cells()) if c not in exclude]
    rng.shuffle(candidates)
    yield from candidates[:max_tries]
