#!/usr/bin/env python
"""SAHS training data: one example per step of the reference plans of REAP's 600 training problems.

The reference plan of a problem (`<problem>.pddl.plan`, next to every problem in data/tabletop/train, 3 to 14
cylinders) is replayed through the feasibility check. At every step the predicate set of the state is computed (the
network input) and the plan's action is the chosen delta. The regions of `put` are the empty cells of the state
(state_adapter.empty_cells): in the SAHS problem definition (IJRR 2022, Section 3) the regions are part of the
environment, and REAP's search also considers every empty cell. A `put` first tries the plan's own cell, which is
always empty; if the check finds it infeasible, another empty cell that the same predicate set marks ManipFree is used
(no extra checker calls). If no candidate is feasible, the replay stops and the problem is reported as truncated.
Only plans replayed up to the goal (holding the target) give examples, with q = -(steps to the goal); the paper's
training set has 2,019 examples from 573 of the 600 plans.

Sharding: --start/--end index the problem files sorted by name, so several processes can split the 600 problems,
each with its own ParallelFeasibilityPool.

    python baselines/sahs/scripts/generate_training_data.py \\
        --problems-dir $REAP_DATA/tabletop/train \\
        --domain $REAP_DATA/tabletop/domain.pddl \\
        --start 0 --end 20 --workers 16 \\
        --out $REAP_OUT/baselines/sahs_training/training/shards/shard_0.jsonl
(python baselines/sahs/run.py data runs the 30 shards and concatenates them into training/examples.jsonl.)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sahs._bootstrap import ensure_upstream_importable  # noqa: E402

ensure_upstream_importable()

import reap.xmimir as xmi  # noqa: E402

from sahs import state_adapter  # noqa: E402
from sahs.dataset import TrainingExample, predicate_set_to_dict  # noqa: E402
from sahs.parallel_pool import ParallelFeasibilityPool  # noqa: E402
from sahs.predicates import compute_predicate_set_parallel  # noqa: E402

ALL_CELLS = [f"cell_{ix}_{iy}" for ix in range(10) for iy in range(7)]

_ACTION_RE = re.compile(r"\(\s*(\w+)\s+([^)]*)\)")
_TARGET_RE = re.compile(r";\s*Target:\s*(\S+)")


def parse_plan_file(path: Path) -> tuple[list[tuple[str, list[str]]], str | None]:
    actions: list[tuple[str, list[str]]] = []
    target: str | None = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(";"):
            m = _TARGET_RE.match(line)
            if m:
                target = m.group(1)
            continue
        m = _ACTION_RE.match(line)
        if m:
            actions.append((m.group(1), m.group(2).split()))
    return actions, target


def _find_action(actions, name: str, obj_name: str, cell_name: str | None):
    for a in actions:
        if a.name != name or a.objects[0].name != obj_name:
            continue
        if name == "put" and cell_name is not None and a.objects[1].name != cell_name:
            continue
        return a
    return None


def _applicable_deltas(actions, operator: str, regions: list[str]) -> list[dict]:
    deltas = []
    for a in actions:
        if a.name != operator:
            continue
        if operator == "pick":
            deltas.append({"object": a.objects[0].name, "region": None})
        else:
            cell = a.objects[1].name
            if cell not in regions:
                continue
            deltas.append({"object": a.objects[0].name, "region": cell})
    return deltas


def _retry_feasible(pool: ParallelFeasibilityPool, kind: str, args: tuple, max_retries: int = 3) -> bool:
    """Up to max_retries fresh checks (force_refresh=True bypasses the cache) of an action the predicate set marks
    infeasible. The motion planner (RRT*) is stochastic, so a re-check can succeed. One success counts as feasible, as
    for the retried edges of the search (sahs.py)."""
    for _ in range(max_retries):
        ((feasible, _reason),) = pool.dispatch_batch([(kind, (*args, True))])
        if feasible:
            return True
    return False


def replay_problem(
    name: str, domain_path: Path, problem_path: Path, plan_path: Path, pool: ParallelFeasibilityPool,
) -> tuple[list[TrainingExample], dict]:
    plan_actions, target = parse_plan_file(plan_path)
    report: dict = {"instance": name, "reap_plan_length": len(plan_actions), "solved": False}

    if target is None or not plan_actions:
        report["truncated_reason"] = "empty_or_unparseable_plan"
        return [], report

    _domain, problem = xmi.parse(str(domain_path), str(problem_path))
    gen = xmi.XSuccessorGenerator(problem)
    state = gen.initial_state
    cylinders = sorted(
        state_adapter.state_to_robot_state(state)["cylinders"].keys(), key=lambda n: int(n.split("_")[1]),
    )
    report["num_cylinders"] = len(cylinders)

    realized: list[tuple[str, dict, dict, list[dict]]] = []  # (operator, delta, predicates_dict, applicable)

    for step_idx, (op_name, args) in enumerate(plan_actions):
        gen_actions = list(gen.action_generator.generate_actions(state))
        rs = state_adapter.state_to_robot_state(state)
        regions = state_adapter.empty_cells(rs, ALL_CELLS)
        ps = compute_predicate_set_parallel(state, target, cylinders, ALL_CELLS, pool, regions)

        if op_name == "pick":
            cyl = args[0]
            action = _find_action(gen_actions, "pick", cyl, None)
            feasible = action is not None and ps.pre_free.get(cyl, False)
            if action is not None and not feasible:
                feasible = _retry_feasible(pool, "pre_free", (rs, cyl))
            delta = {"object": cyl, "region": None}
        elif op_name == "put":
            cyl = args[0]
            reap_cell = args[1] if len(args) > 1 else None
            put_by_cell = {
                a.objects[1].name: a for a in gen_actions if a.name == "put" and a.objects[0].name == cyl
            }
            held_rs = state_adapter.with_synthetic_holding(rs, cyl)
            action = None
            delta = None
            # the plan's own cell first (always empty here), then another empty cell that the predicate set marks
            # ManipFree
            candidate_order = [reap_cell] if reap_cell in put_by_cell else []
            candidate_order += [c for c in regions if c in put_by_cell and c != reap_cell]
            for cand in candidate_order:
                ok = ps.manip_free.get((cyl, cand), False)
                if cand == reap_cell and not ok:
                    ok = _retry_feasible(pool, "manip_free", (held_rs, cyl, cand))
                if ok:
                    action = put_by_cell[cand]
                    delta = {"object": cyl, "region": cand}
                    break
            feasible = action is not None
        else:
            report["truncated_at_step"] = step_idx
            report["truncated_reason"] = f"unknown_op_{op_name}"
            return [], report

        if not feasible:
            report["truncated_at_step"] = step_idx
            report["truncated_reason"] = "infeasible"
            report["realized_steps"] = len(realized)
            return [], report

        applicable = _applicable_deltas(gen_actions, op_name, regions)
        realized.append((op_name, delta, predicate_set_to_dict(ps), applicable))
        state = action.apply(state)

    final_rs = state_adapter.state_to_robot_state(state)
    if final_rs["holding"] != target:
        report["truncated_reason"] = "plan_exhausted_not_at_goal"
        report["realized_steps"] = len(realized)
        return [], report

    total = len(realized)
    examples = [
        TrainingExample(
            instance=name, step=t, operator=op, target=target, cylinders=cylinders,
            delta=delta, q=-float(total - t), applicable_deltas=applicable, predicates=ps_dict,
        )
        for t, (op, delta, ps_dict, applicable) in enumerate(realized)
    ]
    report["solved"] = True
    report["realized_steps"] = total
    report["num_examples"] = len(examples)
    return examples, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems-dir", required=True, type=Path)
    parser.add_argument("--domain", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None, help="exclusive; default = all")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    pddl_files = sorted(args.problems_dir.glob("*.pddl"), key=lambda p: p.stem)
    shard = pddl_files[args.start:args.end]
    print(f"shard: {len(shard)} problems ({args.start}:{args.end}) of {len(pddl_files)} total")

    import json

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.out.with_suffix(".report.json")

    # written after every problem (examples appended, report.json rewritten), so a stopped shard keeps its finished
    # problems
    all_examples: list[TrainingExample] = []
    reports: list[dict] = []
    pool = ParallelFeasibilityPool(n_workers=args.workers)
    t0 = time.perf_counter()
    try:
        with open(args.out, "w") as jsonl_f:
            for i, pddl_path in enumerate(shard):
                plan_path = pddl_path.with_suffix(".pddl.plan")
                if not plan_path.exists():
                    continue
                name = pddl_path.stem
                examples, report = replay_problem(name, args.domain, pddl_path, plan_path, pool)
                reports.append(report)
                all_examples.extend(examples)
                for ex in examples:
                    from dataclasses import asdict
                    jsonl_f.write(json.dumps(asdict(ex)) + "\n")
                jsonl_f.flush()
                status = "solved" if report["solved"] else f"truncated({report.get('truncated_reason')})"
                print(f"  [{i+1}/{len(shard)}] {name}: {status}, {len(examples)} examples "
                      f"({time.perf_counter() - t0:.0f}s elapsed)")

                solved_so_far = [r for r in reports if r["solved"]]
                report_path.write_text(json.dumps({
                    "shard": [args.start, args.end],
                    "num_problems": len(shard),
                    "num_solved": len(solved_so_far),
                    "num_examples": len(all_examples),
                    "wall_s": round(time.perf_counter() - t0, 2),
                    "in_progress": i + 1 < len(shard),
                    "reports": reports,
                }, indent=2))
    finally:
        pool.shutdown()
    elapsed = time.perf_counter() - t0

    solved = [r for r in reports if r["solved"]]
    print(f"\nsolved {len(solved)}/{len(reports)}, {len(all_examples)} examples, {elapsed:.1f}s")
    print(f"wrote {args.out}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
