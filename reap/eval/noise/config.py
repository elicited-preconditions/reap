"""Shared constants of the perturbed-observation evaluation (appendix "Execution Under Perturbed Observations"): grid
geometry (must match tampanda's GridDomain config exactly, as used by
reap.domains.tabletop.TabletopFeasibilityChecker.init()), the evaluation-suite paths, the search settings and the
noise grids. The checkpoint is an argument of reap.eval.noise.sweep.
"""
from __future__ import annotations

from pathlib import Path

from reap.paths import data_root


# --- GridDomain config, as in
# reap/domains/tabletop.py::TabletopFeasibilityChecker.init(). ---
GRID_CELL_SIZE = 0.04
GRID_WORKING_AREA = (0.4, 0.3)
GRID_TABLE_BODY_NAME = "simple_table"
GRID_TABLE_GEOM_NAME = "table_surface"
GRID_OFFSET_X = 0.05
GRID_OFFSET_Y = 0.25

# Per-call watchdog (seconds), on top of FILTER_TIMEOUT_S in
# reap/rl/av_star/sim.py.
CHECKER_TIMEOUT_S = 30.0

# --- Data paths: the open-tabletop training domain and the 15-set evaluation suite of Table 1, under $REAP_DATA
# (reap.paths). The checkpoint is an argument of reap.eval.noise.sweep. ---
def domain_path() -> Path:
    return data_root() / "tabletop" / "domain.pddl"


def eval_data_root() -> Path:
    return data_root() / "tabletop" / "eval"


# The 15 evaluation sets (300 problems) of Table 1, the same sets as in
# reap/domains/tabletop.py and baselines/ploi/ploi/config.py EVAL_SETS.
EVAL_CELLS = (
    "obj10",
    "obj12",
    "obj14",
    "obj16",
    "obj18",
    "obj20",
    "obj22",
    "obj24",
    "obj10_plan3",
    "obj12_plan3",
    "obj14_plan5",
    "obj16_plan5",
    "obj10_plan5_biased2",
    "obj12_plan7_biased3",
    "obj14_plan9_biased3",
)


def eval_cell_dir(cell: str) -> Path:
    return eval_data_root() / cell


# --- Search settings: GBFS = gbfs_value_eval(..., max_expansion=EVAL_MAX_BUDGET, time_limit_s=EVAL_TIME_LIMIT_S);
# greedy = rollouts(problems, model, GREEDY_MAX_EXPANSION, closed=True, stochastic=False,
# max_infeasible_retries=MAX_INFEASIBLE_RETRIES) (greedy_capture.py). ---
MAX_INFEASIBLE_RETRIES = 2
EVAL_TIME_LIMIT_S = 90.0
EVAL_MAX_BUDGET = 16384  # GBFS max_expansion
GREEDY_MAX_EXPANSION = 1000
GREEDY_CLOSED = True
GREEDY_STOCHASTIC = False

# The checks are physics-only: run with MUJOCO_GL=disabled (configs/tabletop.yaml sets it for the noise protocol).

# --- Noise sweep grids. N1 in meters (mm/1000); N2/N3 are probabilities. Index 0 in each grid is the shared
# "off" setting. ---
N1_SIGMA_MM = (0, 2, 5, 10)
N2_OCCUPANCY_FLIP_P = (0.0, 0.01, 0.02, 0.05)
N3_EXECUTION_FAILURE_P = (0.0, 0.02, 0.05, 0.10)

# Replanning budget: exactly one new re-search attempt per problem on
# first execution-time disagreement (a per-problem yes/no, not a closed-loop controller).
MAX_REPLAN_ATTEMPTS = 1

# Stale-verdict sampling: fraction of execution-time cache HITS that get a
# forced second, uncached re-query for ground-truth comparison.
STALE_VERDICT_SAMPLE_RATE = 0.1
