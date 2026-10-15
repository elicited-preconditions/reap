"""Aggregation of the sweep cells (one JSON per cell, written by
reap.eval.noise.sweep) into the tables of the appendix "Execution Under Perturbed Observations"."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_sweep_cell(path: Path) -> dict:
    return json.loads(path.read_text())


def load_all_sweep_cells(sweep_dir: Path) -> List[dict]:
    return [load_sweep_cell(p) for p in sorted(sweep_dir.glob("*.json"))]


def summarize_cell(cell: dict) -> Dict:
    rows = cell["rows"]
    n = len(rows)
    completed = sum(1 for r in rows if r["completed_without_replanning"])
    replanned = [r for r in rows if r["replanned"]]
    recovered = [r for r in replanned if r["replan_solved"]]
    return {
        "mode": cell["mode"], "noise": cell["noise"], "value": cell["value"],
        "n_solved_baseline": cell["n_solved_baseline"], "n_executed": n,
        "fraction_completed_without_replanning": completed / n if n else 0.0,
        "n_replanned": len(replanned),
        "n_replan_recovered": len(recovered),
        "replan_recovery_rate": len(recovered) / len(replanned) if replanned else None,
        "mean_plan_len": sum(r["plan_len"] for r in rows) / n if n else 0.0,
    }


def per_noise_model_table(cells: List[dict], noise: str) -> List[Dict]:
    return sorted(
        (summarize_cell(c) for c in cells if c["noise"] == noise),
        key=lambda s: (s["mode"], s["value"]),
    )
