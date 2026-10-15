#!/usr/bin/env python
"""Configuration 3 (cell-generator, ff-eager): PDDLStream with put-cells
produced by a stream (not pre-grounded over all 70 cells) -> shrinks FD's
optimistic action space.

pddlstream forbids a FLUENT stream's output from appearing in an action effect,
so we split the put-cell choice into two streams:
  sample-cell(?cyl) -> (?cel)            NON-fluent generator: yields candidate
                                          cells; certifies (Cell ?cel). Its output
                                          ?cel is allowed in the (occupied) effect.
  test-placeable(?cyl ?cel) -> (?t)      FLUENT test (reads occupied/holding):
                                          runs checker.check("put"); ?t witnesses
                                          (Placeable ?cyl ?cel ?t) -- a PRECONDITION
                                          only, so it's legal.
Cells are NOT in init as (Cell ...) facts -> put only grounds on stream-produced
cells (lazy), all cells still legal. graspable/env/checker/cwd-fix/SIGALRM/CSV/
MP-accounting identical to bench_problem_test_stream.py. 1 MP call = 1 checker.check.
"""
import argparse
import contextlib
import os
import random
import re
import signal
import tempfile
import time
import types
from pathlib import Path

import numpy as np

from pddlstream.algorithms.meta import solve
from pddlstream.language.generator import from_fn, from_gen_fn
from pddlstream.language.constants import Output, PDDLProblem
from pddlstream.utils import read, INF
from tampanda.symbolic.domains.tabletop import generate_data as gd

DOMAIN_PDDL = """(define (domain ttppgen)
  (:requirements :strips)
  (:predicates
    (Cylinder ?cyl) (Cell ?cel)
    (occupied ?cel ?cyl) (empty ?cel) (holding ?cyl) (gripper-empty)
    (Graspable ?cyl ?t) (Placeable ?cyl ?cel ?t))
  (:action pick
    :parameters (?cyl ?cel ?t)
    :precondition (and (Cylinder ?cyl) (gripper-empty)
                       (occupied ?cel ?cyl) (Graspable ?cyl ?t))
    :effect (and (not (gripper-empty)) (not (occupied ?cel ?cyl))
                 (empty ?cel) (holding ?cyl)))
  (:action put
    :parameters (?cyl ?cel ?t)
    :precondition (and (Cylinder ?cyl) (Cell ?cel) (holding ?cyl)
                       (Placeable ?cyl ?cel ?t))
    :effect (and (gripper-empty) (not (holding ?cyl))
                 (occupied ?cel ?cyl) (not (empty ?cel)))))
"""
STREAM_PDDL = """(define (stream ttppgen)
  (:stream sample-graspable
    :inputs (?cyl) :domain (Cylinder ?cyl)
    :fluents (occupied) :outputs (?t) :certified (Graspable ?cyl ?t))
  (:stream sample-cell
    :inputs (?cyl) :domain (Cylinder ?cyl)
    :outputs (?cel) :certified (Cell ?cel))
  (:stream test-placeable
    :inputs (?cyl ?cel) :domain (and (Cylinder ?cyl) (Cell ?cel))
    :fluents (occupied holding) :outputs (?t) :certified (Placeable ?cyl ?cel ?t)))
"""


class _Timeout(Exception):
    pass


def _occ_holding(fluents):
    occ, holding = {}, None
    for f in fluents:
        p = str(f[0]).lower()
        if p == "occupied":
            occ.setdefault(f[2], []).append(f[1])
        elif p == "holding":
            holding = f[1]
    return occ, holding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--algorithm", default="adaptive")
    ap.add_argument("--planner", default="ff-eager")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-workers", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=300.0)
    # Search knobs. Defaults match pddlstream's own solve() defaults
    # exactly (meta.py: search_sample_ratio=1, max_skeletons=INF), so omitting
    # these flags gives pddlstream's default behaviour.
    ap.add_argument("--search-sample-ratio", type=float, default=1)
    ap.add_argument("--max-skeletons", type=float, default=INF)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    txt = Path(a.problem).read_text()
    occ0 = re.findall(r"\(occupied\s+(cell_\d+_\d+)\s+(cylinder_\d+)\)", txt)
    placements = {cyl: cel for cel, cyl in occ0}
    target = re.search(r"\(holding\s+(cylinder_\d+)\)", txt.split(":goal")[1]).group(1)
    problem_id = Path(a.problem).name

    eargs = types.SimpleNamespace(
        inner_workers=a.inner_workers, ik_iters=100, ik_pos_thresh=0.005,
        cell_size=0.04, grid_width=0.4, grid_height=0.3,
        grid_offset_x=0.05, grid_offset_y=0.25)
    (env, planner, feas_planner, checker, reachable_cells,
     grid, sm_, gp_, pool, xmlp) = gd._build_env(eargs)
    ALL_CELLS = sorted(grid.cells.keys())

    ctr = {"mp": 0}

    # --- query counting (MP / plan): an observer, not a cache --------------
    # Records the (state, action, args) key of every feasibility query, so the
    # number of distinct (state, action) queries is reported next to the number
    # of calls. It never skips a call: PDDLStream has no feasibility cache. The
    # key is that of REAP's feasibility cache (reap/rl/av_star/sim.py).
    seen = set()

    def _observe(action, robot_state, args):
        cyls = frozenset(
            (c, tuple(sorted(v))) for c, v in robot_state["cylinders"].items()
        )
        seen.add((cyls, robot_state.get("holding"), action, args))
    # ------------------------------------------------------------------------

    def graspable_fn(cyl, fluents=frozenset()):
        occ, _ = _occ_holding(fluents)
        ctr["mp"] += 1
        _rs = {"cylinders": occ, "gripper_empty": True, "holding": None}
        _observe("pick", _rs, (cyl, None))
        ok, _ = checker.check("pick", _rs, cylinder_name=cyl)
        return Output(f"g_{cyl}") if ok else None

    def cell_gen(cyl):                       # non-fluent: yields candidate cells
        cells = list(ALL_CELLS)
        rng.shuffle(cells)
        for cel in cells:
            yield (cel,)

    def placeable_fn(cyl, cel, fluents=frozenset()):
        occ, holding = _occ_holding(fluents)
        occ_cells = {c for cs in occ.values() for c in cs}
        if cel in occ_cells:                 # occupied target: cheap reject, no MP call
            return None
        ctr["mp"] += 1
        _rs = {"cylinders": occ, "gripper_empty": False, "holding": holding or cyl}
        _observe("put", _rs, (cyl, cel))
        ok, _ = checker.check("put", _rs, cylinder_name=cyl, target_cell=cel)
        return Output(f"p_{cyl}_{cel}") if ok else None

    sm = {"sample-graspable": from_fn(graspable_fn),
          "sample-cell": from_gen_fn(cell_gen),
          "test-placeable": from_fn(placeable_fn)}
    occ_cells = set(placements.values())
    init = ([("Cylinder", c) for c in placements]
            + [("occupied", cell, cyl) for cyl, cell in placements.items()]
            + [("empty", c) for c in ALL_CELLS if c not in occ_cells]
            + [("gripper-empty",)])           # NB: no (Cell ...) in init -> cells are lazy
    td = Path(tempfile.mkdtemp())
    (td / "d.pddl").write_text(DOMAIN_PDDL)
    (td / "s.pddl").write_text(STREAM_PDDL)
    prob = PDDLProblem(read(str(td / "d.pddl")), {}, read(str(td / "s.pddl")),
                       sm, init, ("holding", target))

    os.chdir(tempfile.mkdtemp(prefix="pdstream_"))

    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
    signal.alarm(int(a.max_time))
    t0 = time.perf_counter()
    solved, steps = 0, 0
    # max_skeletons must reach solve() as int or INF (float('inf')), matching
    # pddlstream's own default type exactly -- not an arbitrary float.
    max_skeletons = INF if a.max_skeletons == INF else int(a.max_skeletons)
    try:
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            sol = solve(prob, algorithm=a.algorithm, unit_costs=True,
                        max_time=a.max_time, verbose=False, planner=a.planner,
                        search_sample_ratio=a.search_sample_ratio,
                        max_skeletons=max_skeletons)
        if sol.plan is not None:
            solved, steps = 1, len(sol.plan)
    except _Timeout:
        pass
    except Exception as e:
        print(f"EXC {problem_id}: {type(e).__name__}: {e}")
    finally:
        signal.alarm(0)
    wall = time.perf_counter() - t0
    wall = round(min(wall, a.max_time), 2) if solved else round(a.max_time, 2)

    # mp_calls = distinct (state, action) queries (MP / plan); total_filter_calls =
    # every call made. PDDLStream has no cache, so repeated queries are planned
    # again.
    row = (f"PDDLStream,{a.dataset},{problem_id},{solved},"
           f"{len(seen)},{ctr['mp']},{steps},{wall}\n")
    Path(a.out).write_text(
        "model,dataset,problem,solved,mp_calls,total_filter_calls,steps,wall_time\n" + row)
    print(row.strip())
    if pool is not None:
        try:
            pool.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
