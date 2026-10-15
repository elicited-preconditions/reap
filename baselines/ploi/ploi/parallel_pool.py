"""Process pool for label generation, one problem per task.

Unlike SAHS's ParallelFeasibilityPool (which parallelizes individual,
fine-grained feasibility CHECKS within one problem's search, and so has to
handle cross-process call-count accounting -- see its module docstring),
the parallelism here is per PROBLEM: `labels.label_problem` does its own
per-problem checkpointing/skip-if-exists and writes its result straight to
disk, so a worker's unit of work is the labelling of one whole problem.
There is no shared mutable state: each
worker writes to its own, distinct `<out_dir>/<stem>.json` file, and
per-problem stats are returned directly from that worker's own call, not
accumulated across workers.

USES PLAIN multiprocessing.Process, NOT multiprocessing.Pool:
`Pool`'s workers are created with `daemon=True` (hardcoded in CPython's
`multiprocessing.pool`), and a daemonic process is disallowed from spawning
its OWN children (`AssertionError: daemonic processes are not allowed to
have children`). `checker.py`'s lazily-built `TabletopFeasibilityChecker`
constructs a `CollisionWorkerPool`, which itself spawns at least one
subprocess (even with `COLLISION_POOL_WORKERS=1`), so any worker that
later touches the checker must itself be a non-daemonic process. SAHS's
`ParallelFeasibilityPool` avoids `Pool` for the same reason.

Each worker is a long-lived spawn-context process: one real MuJoCo scene is
constructed lazily on that worker's first `filter_action` call and reused
across every problem the worker pool subsequently hands it, not rebuilt per
problem.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path


def _worker_init() -> None:
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root))
    from ploi._bootstrap import ensure_upstream_importable
    ensure_upstream_importable()


@dataclass
class LabelJobResult:
    problem_path: str
    skipped: bool
    wall_time_s: float | None
    n_attempts: int
    n_kept: int
    n_total: int
    error: str | None = None
    excluded: bool = False  # single_plan only: PLAN couldn't solve the full problem, problem dropped entirely


def _process_one(problem_path, domain_path, plan_fn_name, out_dir, plan_max_expansion, plan_time_limit_s, checkpoint_every, strategy) -> LabelJobResult:
    from ploi import labels, plan_search

    try:
        plan_fn = plan_search.PLAN_FUNCTIONS[plan_fn_name]
        if strategy == "single_plan":
            result = labels.label_problem_single_plan(
                Path(problem_path), Path(domain_path), plan_fn, Path(out_dir),
                plan_max_expansion=plan_max_expansion, plan_time_limit_s=plan_time_limit_s,
            )
            if result is None:  # PLAN couldn't solve the full problem -- excluded, matches the actual code
                return LabelJobResult(problem_path, False, None, 0, 0, 0, excluded=True)
            example, stats = result
        else:
            example, stats = labels.label_problem(
                Path(problem_path), Path(domain_path), plan_fn, Path(out_dir),
                plan_max_expansion=plan_max_expansion, plan_time_limit_s=plan_time_limit_s,
                checkpoint_every=checkpoint_every,
            )
        n_kept = sum(example.labels.values())
        n_total = len(example.labels)
        if stats is None:  # already-completed problem, skipped
            return LabelJobResult(problem_path, True, None, 0, n_kept, n_total)
        return LabelJobResult(problem_path, False, stats.wall_time_s, stats.n_attempts, n_kept, n_total)
    except Exception as exc:  # noqa: BLE001 -- one bad problem must not kill the whole shard
        return LabelJobResult(problem_path, False, None, 0, 0, 0, error=repr(exc))


def _worker_loop(task_queue, result_queue, static_args) -> None:
    _worker_init()
    domain_path, plan_fn_name, out_dir, plan_max_expansion, plan_time_limit_s, checkpoint_every, strategy = static_args
    while True:
        item = task_queue.get()
        if item is None:  # poison pill
            return
        idx, problem_path = item
        result = _process_one(
            problem_path, domain_path, plan_fn_name, out_dir,
            plan_max_expansion, plan_time_limit_s, checkpoint_every, strategy,
        )
        result_queue.put((idx, result))


def run_parallel_label_generation(
    problem_paths: list[Path],
    domain_path: Path,
    out_dir: Path,
    plan_fn_name: str = "bfs",
    n_workers: int = 8,
    plan_max_expansion: int = 200,
    plan_time_limit_s: float = 60.0,
    checkpoint_every: int = 10,
    strategy: str = "greedy",  # "greedy" (label_problem) or "single_plan" (label_problem_single_plan)
) -> list[LabelJobResult]:
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    static_args = (str(domain_path), plan_fn_name, str(out_dir), plan_max_expansion, plan_time_limit_s, checkpoint_every, strategy)
    workers = [
        ctx.Process(target=_worker_loop, args=(task_queue, result_queue, static_args), daemon=False)
        for _ in range(n_workers)
    ]
    for w in workers:
        w.start()

    for idx, p in enumerate(problem_paths):
        task_queue.put((idx, str(p)))
    for _ in workers:
        task_queue.put(None)

    results_by_idx: dict[int, LabelJobResult] = {}
    for _ in range(len(problem_paths)):
        idx, result = result_queue.get()
        results_by_idx[idx] = result

    for w in workers:
        w.join(timeout=30)
        if w.is_alive():
            w.terminate()

    return [results_by_idx[i] for i in range(len(problem_paths))]
