"""Grid-cell enumeration shared by domain/problem_gen.py (PDDL emission),
state.py (free-cell queries), and sampler.py (RelocateSampler) -- kept
dependency-free (only bt_h.config) so it can sit underneath all three
without any risk of circular imports.
"""
from __future__ import annotations

from bt_h.config import GRID_NX, GRID_NY


def all_cells() -> list[str]:
    return [f"cell_{ix}_{iy}" for ix in range(GRID_NX) for iy in range(GRID_NY)]


def cell_coords(cell: str) -> tuple[int, int]:
    _, ix, iy = cell.split("_")
    return int(ix), int(iy)


def adjacency_atoms() -> list[str]:
    atoms = []
    for ix in range(GRID_NX):
        for iy in range(GRID_NY):
            cell = f"cell_{ix}_{iy}"
            if iy < GRID_NY - 1:
                atoms.append(f"(adjacent north {cell} cell_{ix}_{iy + 1})")
            if ix < GRID_NX - 1:
                atoms.append(f"(adjacent east {cell} cell_{ix + 1}_{iy})")
    return atoms
