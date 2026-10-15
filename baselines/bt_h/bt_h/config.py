"""Shared constants of the BT-h baseline: the grid geometry (tampanda's GridDomain as REAP's
TabletopFeasibilityChecker.init() builds it, reap/domains/tabletop.py), the cylinder-size index pools, the path of the
relocate domain, and the parameters of backtracking, data collection, the classifier veto and the evaluation.
"""
from __future__ import annotations

from pathlib import Path

# --- GridDomain config -------------------------------------------------
GRID_CELL_SIZE = 0.04
GRID_WORKING_AREA = (0.4, 0.3)
GRID_TABLE_BODY_NAME = "simple_table"
GRID_TABLE_GEOM_NAME = "table_surface"
GRID_OFFSET_X = 0.05
GRID_OFFSET_Y = 0.25
GRID_NX = 10  # columns (ix), matches REAP's 10x7 open-tabletop grid
GRID_NY = 7   # rows (iy); iy=0 is south / robot-facing (closest to the arm)

# --- Cylinder index -> (radius_m, half_height_m) pools ------------------
# Copied from tampanda/symbolic/domains/tabletop/state_manager.py::StateManager.CYLINDER_SPECS
# (idx 0-14 thin, 15-24 medium, 25-29 thick). cylinder_<N>'s size
# is derived from N alone, so instance generation must draw indices from the
# right pool, not just name cylinders cylinder_0..cylinder_K arbitrarily.
THIN_IDX_RANGE = range(0, 15)
MEDIUM_IDX_RANGE = range(15, 25)
THICK_IDX_RANGE = range(25, 30)

# --- Checker timeout ------------------------------------------------------
# Timeout of one check (seconds; a typical check takes 50 to 90 ms), as FILTER_TIMEOUT_S in reap/rl/av_star/sim.py.
CHECKER_TIMEOUT_S = 30.0

# --- Retry policy --------------------------------------
# pick/put get same-theta retries (RRT* has no seeding hook -- identical
# calls can disagree); relocate's *cell choice* theta is not retried on its
# own failure (a rejected cell is a real data point, not planner noise).
MAX_PICK_RETRIES = 3
MAX_PUT_RETRIES = 3

# --- Backtracking / sampler budgets ------------------------------------
# r_iter = 20, as in the paper's Appendix A.4: "The SeSamE-based methods
# (including our own) set r_iter = 20 for all domains."
# MAX_TRIES_PICK has no effect: pick_thetas() yields one theta (the
# checker chooses the grasp itself).
MAX_TRIES_PICK = 20
MAX_TRIES_RELOCATE = 20

# --- BT-h classifier veto threshold -----------------------------------------
# Set by hand (the paper's appendix); a candidate whose confidence is below
# tau is vetoed.
BT_H_TAU = 0.2

# --- Instance generation defaults ------------------------------------------
# Synthetic problems of the relocate domain (relocate_domain.pddl), 600 for
# the seed demonstrations and data collection, K = 1 to 6 blockers each.
K_TRAIN_RANGE = range(1, 7)     # K in {1..6} for seed-demo / data-collection instances
N_TRAIN_PER_K = 100              # -> 600 seed-demo-pool instances total

# --- Algorithm 3 data-collection loop defaults ------------------------------
# 30 search trees per level (the paper: r_dp = 4,000) and at most 400
# classifier training steps per round.
DATA_COLLECTION_R_DP = 30
DATA_COLLECTION_TRAIN_MAX_STEPS = 400

# --- Evaluation budget -------------------------------------------------
# The time limit is the budget (scripts/run_bt_h.py: 300 s per problem, as
# for every Table 1 row); EVAL_MAX_CHECKER_CALLS is a backstop that does
# not bind first.
EVAL_MAX_CHECKER_CALLS = 100_000


def relocate_domain_path() -> Path:
    """The domain of this baseline, REAP's open-tabletop domain with a relocate action (bt_h/domain/)."""
    return Path(__file__).parent / "domain" / "relocate_domain.pddl"
