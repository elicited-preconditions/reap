"""Shared constants of the SAHS baseline: the grid geometry (tampanda's GridDomain as REAP's
TabletopFeasibilityChecker.init() builds it) and the timeout of a feasibility check.
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

# Timeout of one feasibility check (seconds; a typical check takes 50 to 90 ms), as FILTER_TIMEOUT_S in
# reap/rl/av_star/sim.py.
CHECKER_TIMEOUT_S = 30.0
