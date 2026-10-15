#!/usr/bin/env python
"""Configurations 1 and 2 (test-stream; --planner ff-astar2 / ff-eager).

Run ONE tabletop eval problem with PDDLStream; emit one CSV row:
  model,dataset,problem,solved,mp_calls,total_filter_calls,steps,wall_time
mp_calls = distinct (state, action) feasibility queries (MP / plan in Table 1),
total_filter_calls = every checker call made.

Formulation (every cell is a put target):
  Graspable(?cyl ?t)        -- no cell argument (the checker does not use it)
  Placeable(?cyl ?cel ?t)   -- ?cel over ALL cells

MP calls: one per checker.check(), i.e. per evaluation of the graspable or
placeable stream, feasible or not (as one filter_action call of REAP's
search).

Hard 300 s solve cap via SIGALRM. Motion planner = speculative RRT when
--inner-workers > 1 (matches tampanda training/eval). The Table 1 runs used
--inner-workers 1 (plain RRT); bench/run.py passes it explicitly.
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
from pddlstream.language.generator import from_fn
from pddlstream.language.constants import Output, PDDLProblem
from pddlstream.utils import read
from tampanda.symbolic.domains.tabletop import generate_data as gd

DOMAIN_PDDL = """(define (domain ttpp)
  (:requirements :strips)
  (:predicates
    (Cell ?cel) (Cylinder ?cyl)
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
                       (empty ?cel) (Placeable ?cyl ?cel ?t))
    :effect (and (gripper-empty) (not (holding ?cyl))
                 (occupied ?cel ?cyl) (not (empty ?cel)))))
"""
STREAM_PDDL = """(define (stream ttpp)
  (:stream sample-graspable
    :inputs (?cyl) :domain (Cylinder ?cyl)
    :fluents (occupied) :outputs (?t) :certified (Graspable ?cyl ?t))
  (:stream sample-placeable
    :inputs (?cyl ?cel) :domain (and (Cylinder ?cyl) (Cell ?cel))
    :fluents (occupied holding) :outputs (?t) :certified (Placeable ?cyl ?cel ?t)))
"""


class _Timeout(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--algorithm", default="adaptive")
    ap.add_argument("--planner", default="ff-astar2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-workers", type=int, default=8)
    ap.add_argument("--max-time", type=float, default=300.0)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)

    txt = Path(a.problem).read_text()
    occ = re.findall(r"\(occupied\s+(cell_\d+_\d+)\s+(cylinder_\d+)\)", txt)
    placements = {cyl: cel for cel, cyl in occ}
    target = re.search(r"\(holding\s+(cylinder_\d+)\)", txt.split(":goal")[1]).group(1)
    problem_id = Path(a.problem).name

    # ---- build env / checker (speculative RRT when inner-workers > 1) ----
    eargs = types.SimpleNamespace(
        inner_workers=a.inner_workers, ik_iters=100, ik_pos_thresh=0.005,
        cell_size=0.04, grid_width=0.4, grid_height=0.3,
        grid_offset_x=0.05, grid_offset_y=0.25)
    (env, planner, feas_planner, checker, reachable_cells,
     grid, sm_, gp_, pool, xmlp) = gd._build_env(eargs)
    cells = sorted(grid.cells.keys())

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
        cyls = {}
        for f in fluents:
            if str(f[0]).lower() == "occupied":
                cyls.setdefault(f[2], []).append(f[1])
        ctr["mp"] += 1
        _rs = {"cylinders": cyls, "gripper_empty": True, "holding": None}
        _observe("pick", _rs, (cyl, None))
        ok, _ = checker.check("pick", _rs, cylinder_name=cyl)
        return Output(f"g_{cyl}") if ok else None

    def placeable_fn(cyl, cel, fluents=frozenset()):
        cyls, holding = {}, None
        for f in fluents:
            p = str(f[0]).lower()
            if p == "occupied":
                cyls.setdefault(f[2], []).append(f[1])
            elif p == "holding":
                holding = f[1]
        ctr["mp"] += 1
        _rs = {"cylinders": cyls, "gripper_empty": False, "holding": holding or cyl}
        _observe("put", _rs, (cyl, cel))
        ok, _ = checker.check("put", _rs, cylinder_name=cyl, target_cell=cel)
        return Output(f"p_{cyl}_{cel}") if ok else None

    sm = {"sample-graspable": from_fn(graspable_fn),
          "sample-placeable": from_fn(placeable_fn)}
    occ_cells = set(placements.values())
    init = ([("Cell", c) for c in cells]
            + [("Cylinder", c) for c in placements]
            + [("occupied", cell, cyl) for cyl, cell in placements.items()]
            + [("empty", c) for c in cells if c not in occ_cells]
            + [("gripper-empty",)])
    td = Path(tempfile.mkdtemp())
    (td / "d.pddl").write_text(DOMAIN_PDDL)
    (td / "s.pddl").write_text(STREAM_PDDL)
    prob = PDDLProblem(read(str(td / "d.pddl")), {}, read(str(td / "s.pddl")),
                       sm, init, ("holding", target))

    # pddlstream writes FastDownward files to "<cwd>/temp/" (relative) and
    # clear_dir()s it each call -> concurrent tasks sharing a cwd clobber each
    # other. Give this process a private node-local cwd.
    os.chdir(tempfile.mkdtemp(prefix="pdstream_"))

    # ---- solve with a hard SIGALRM wall ----
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_Timeout()))
    signal.alarm(int(a.max_time))
    t0 = time.perf_counter()
    solved, steps = 0, 0
    try:
        with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
            sol = solve(prob, algorithm=a.algorithm, unit_costs=True,
                        max_time=a.max_time, verbose=False,
                        planner=a.planner, max_planner_time=30)
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

    # Write the row FIRST (so the result survives even if teardown stalls;
    # the caller's time limit is the backstop), then cleanly tear down the
    # speculative pool (terminate+join releases its semaphores).
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
