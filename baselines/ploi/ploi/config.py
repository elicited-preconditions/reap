"""Shared constants of the PLOI baseline: the grid geometry (tampanda's GridDomain as REAP's
TabletopFeasibilityChecker.init() builds it), the hyperparameters of the PLOI paper (gamma, loss weight, GNN width,
message-passing rounds), the evaluation budgets and the 15 evaluation sets of Table 1.
"""
from __future__ import annotations

# GridDomain config, as in
# reap/domains/tabletop.py::TabletopFeasibilityChecker.init()
GRID_CELL_SIZE = 0.04
GRID_WORKING_AREA = (0.4, 0.3)
GRID_TABLE_BODY_NAME = "simple_table"
GRID_TABLE_GEOM_NAME = "table_surface"
GRID_OFFSET_X = 0.05
GRID_OFFSET_Y = 0.25

# Default of FILTER_TIMEOUT_S (seconds), the timeout of one feasibility check, as in reap/rl/av_star/sim.py.
CHECKER_TIMEOUT_S = 30.0

# --- PLOI paper hyperparameters (Silver et al., AAAI 2021, Sections 4 and 5, Appendix A) ---
GAMMA = 0.9                 # incremental-planning geometric threshold, "for all experiments"
FALSE_NEGATIVE_WEIGHT = 10  # weighted BCE loss weight on y=1 objects
GNN_HIDDEN_DIM = 16
GNN_NUM_MESSAGE_PASSING_ROUNDS = 3
TRAIN_LR = 1e-3
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 1000

# --- PLAN search budgets ---
# The evaluation budgets of REAP's searches (Table 1: 300 s per problem), used by run_evaluation.py. Label
# generation uses its own, smaller budget (generate_training_labels.py).
EVAL_MAX_EXPANSION = 16384
EVAL_TIME_LIMIT_S = 300.0

# The 15 evaluation sets of Table 1 (20 problems each), in the order of the Table 1 run.
EVAL_SETS = (
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
