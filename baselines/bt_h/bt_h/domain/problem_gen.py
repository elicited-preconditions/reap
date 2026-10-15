"""Packed-column K-blocker instance generator.

Target cylinder at (column, target_row); K blockers stacked directly south
of it at rows target_row-1 .. target_row-K (cylinder_1 nearest the robot /
lowest row, cylinder_K farthest) -- this is exactly the geometry that
relocate_domain.pddl's occluder-gated `pick` forces into a strict
"nearest-blocker-first" task-plan order (see that file's header
comment). Optional decoy cylinders are scattered in OTHER columns to raise
object count / contest free-cell space for RelocateSampler, without
affecting the intended K (decoys never occupy the target's column, so they
never become additional occluders of anything in the chain).

Cylinder indices are assigned sequentially starting at 0 (target=cylinder_0,
blockers=cylinder_1..K, decoys=cylinder_(K+1)..) -- per
tampanda's StateManager.CYLINDER_SPECS (see bt_h.config), this keeps
every generated instance within the thin-cylinder pool (indices 0-14,
radius 12.5mm) unless the caller pushes total cylinder count past 14,
matching the SAHS baseline's convention and REAP's own default instance shape.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from bt_h.config import GRID_NX, GRID_NY, THIN_IDX_RANGE
from bt_h.grid import adjacency_atoms, all_cells


@dataclass(frozen=True)
class Instance:
    name: str
    cylinder_cells: dict[str, str]  # cylinder_name -> cell_name
    target: str
    k: int  # intended number of blockers, per construction (before any real
             # controller run -- the real controller-verified K can differ)

    def cylinders(self) -> list[str]:
        return sorted(self.cylinder_cells, key=lambda n: int(n.split("_")[1]))


def generate_instance(
    name: str,
    k: int,
    seed: int,
    n_decoys: int = 0,
    column: int | None = None,
    target_row: int | None = None,
) -> Instance:
    """Packed-column instance. Raises ValueError if the requested geometry
    doesn't fit the grid or would push cylinder count past the thin-index
    pool (0-14) -- callers needing more cylinders than that must pass
    explicit index handling of their own (not needed at this scale: K<=6)."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k + 1 + n_decoys > len(THIN_IDX_RANGE):
        raise ValueError(
            f"k={k} blockers + 1 target + {n_decoys} decoys = "
            f"{k + 1 + n_decoys} cylinders exceeds the thin-cylinder index "
            f"pool size ({len(THIN_IDX_RANGE)})"
        )

    if target_row is None:
        # room for k blockers south (rows target_row-k .. target_row-1) plus
        # at least one free row of headroom north where possible.
        target_row = min(k + 1, GRID_NY - 1)
    if column is None:
        column = GRID_NX // 2
    if target_row < k:
        raise ValueError(f"target_row={target_row} too low for k={k} blockers south of it")

    rng = random.Random(seed)

    cylinder_cells = {"cylinder_0": f"cell_{column}_{target_row}"}
    for i in range(1, k + 1):
        cylinder_cells[f"cylinder_{i}"] = f"cell_{column}_{target_row - i}"

    occupied = set(cylinder_cells.values())
    other_column_cells = [
        f"cell_{cx}_{cy}"
        for cx in range(GRID_NX)
        for cy in range(GRID_NY)
        if cx != column and f"cell_{cx}_{cy}" not in occupied
    ]
    rng.shuffle(other_column_cells)
    for j in range(n_decoys):
        cylinder_cells[f"cylinder_{k + 1 + j}"] = other_column_cells[j]

    return Instance(name=name, cylinder_cells=cylinder_cells, target="cylinder_0", k=k)


def emit_pddl(instance: Instance) -> str:
    cells = all_cells()
    cylinders = instance.cylinders()
    occupied_cells = set(instance.cylinder_cells.values())

    init_lines = list(adjacency_atoms())
    for cyl, cell in instance.cylinder_cells.items():
        init_lines.append(f"(occupied {cell} {cyl})")
    for cell in cells:
        if cell not in occupied_cells:
            init_lines.append(f"(empty {cell})")
    init_lines.append("(gripper-empty)")
    init_block = "\n    ".join(init_lines)

    return f"""(define (problem {instance.name})
  (:domain tabletop-relocate)

  (:objects
    {" ".join(cells)} - cell
    {" ".join(cylinders)} - cylinder
  )

  (:init
    {init_block}
  )

  (:goal (and
    (holding {instance.target})
  ))
)

; --- generation metadata ---
; source: bt_h.domain.problem_gen (packed-column generator)
; instance: {instance.name}
; target_cylinder: {instance.target}
; intended_k: {instance.k}
; num_cylinders: {len(cylinders)}
; grid: {GRID_NX}x{GRID_NY}
"""


def generate_batch(
    prefix: str,
    k_counts: dict[int, int],
    seed_start: int = 0,
    n_decoys: int = 0,
) -> list[Instance]:
    """Generate `count` instances for each K in k_counts, varying column and
    seed per instance for geometric diversity (not one fixed layout per K)."""
    instances = []
    seed = seed_start
    for k in sorted(k_counts):
        count = k_counts[k]
        for i in range(count):
            column = 1 + (seed % (GRID_NX - 2))  # keep off the grid's outer edge columns
            instances.append(
                generate_instance(f"{prefix}_k{k}_{i}", k, seed=seed, n_decoys=n_decoys, column=column)
            )
            seed += 1
    return instances
