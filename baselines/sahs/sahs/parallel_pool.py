"""Multi-process worker pool for the independent feasibility and occlusion checks of one state (the 16 processes per
problem of the paper's appendix).

tampanda's CollisionWorkerPool parallelizes the collision checks within one planning call. Here each request is a
whole PreFree / ManipFree / occlusion-row check with its own IK and RRT* runs, so the pool follows tampanda's
symbolic/domains/tabletop/generate_data.py instead: plain multiprocessing.Process workers, each with its own checker
stack, built once at startup.

The workers serve many problems: the MuJoCo scene (SCENE_SYMBOLIC) has 30 cylinder slots for every problem (REAP's
problems have up to 24 cylinders), and only the state dict sent with each request changes.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
import time

from sahs import config


def _worker_loop(task_queue, result_queue, timeout_s: float) -> None:
    from sahs._bootstrap import ensure_upstream_importable

    ensure_upstream_importable()

    from sahs.checker import InstrumentedFeasibilityChecker
    from sahs.occlusion import OcclusionChecker

    checker = InstrumentedFeasibilityChecker(timeout_s=timeout_s)
    occlusion_checker = OcclusionChecker(timeout_s=timeout_s)

    while True:
        item = task_queue.get()
        if item is None:  # poison pill
            break
        req_id, kind, args = item
        calls_before = checker.stats.total_calls + occlusion_checker.stats.total_calls
        try:
            if kind == "pre_free":
                rs, cyl, *rest = args
                force_refresh = rest[0] if rest else False
                feasible, info = checker.check(
                    rs, "pick", cyl, predicate_label="PreFree", force_refresh=force_refresh,
                )
                result = (feasible, info.get("reason", ""))
            elif kind == "manip_free":
                rs, cyl, cell, *rest = args
                force_refresh = rest[0] if rest else False
                feasible, info = checker.check(
                    rs, "put", cyl, cell, predicate_label="ManipFree", force_refresh=force_refresh,
                )
                result = (feasible, info.get("reason", ""))
            elif kind == "occludes_pre_row":
                rs, target, candidates, baseline_reason = args
                result = occlusion_checker.occludes_pre_row(rs, target, candidates, baseline_reason)
            elif kind == "occludes_manip_row":
                rs, held, cell, candidates, baseline_reason = args
                result = occlusion_checker.occludes_manip_row(rs, held, cell, candidates, baseline_reason)
            else:
                raise ValueError(f"unknown task kind: {kind}")
            calls_after = checker.stats.total_calls + occlusion_checker.stats.total_calls
            # Call-count delta travels back on the SAME per-request response
            # already used for correctness (dispatch_batch tracks
            # req_id -> result 1:1, race-free by construction), NOT on a
            # separate broadcast/poll mechanism: with a shared
            # multiprocessing.Queue, a broadcast sentinel per worker is not
            # consumed one per worker (a fast worker can drain two while a
            # busy one gets none), which would misattribute calls across
            # problems.
            result_queue.put((req_id, True, result, calls_after - calls_before))
        except Exception as e:  # noqa: BLE001 -- report to the dispatcher, don't crash the worker
            result_queue.put((req_id, False, repr(e), 0))


class DeadlineExceeded(Exception):
    """dispatch_batch(deadline=...) passed its deadline; the batch was abandoned."""


class ParallelFeasibilityPool:
    def __init__(self, n_workers: int = 4, timeout_s: float = config.CHECKER_TIMEOUT_S):
        self._ctx = mp.get_context("spawn")
        self.task_queue = self._ctx.Queue()
        self.result_queue = self._ctx.Queue()
        self.timeout_s = timeout_s
        self._batch_no = 0
        self.workers = [
            self._ctx.Process(target=_worker_loop, args=(self.task_queue, self.result_queue, timeout_s))
            for _ in range(n_workers)
        ]
        for w in self.workers:
            w.start()
        self._calls_since_reset = 0

    def dispatch_batch(self, requests: list[tuple[str, tuple]], deadline: float | None = None) -> list:
        """requests: list of (kind, args). Returns results in the SAME
        order as requests, regardless of completion order across workers.
        Also accumulates each request's real-MP-call delta (reported by the
        worker that handled it) into self._calls_since_reset -- see
        total_calls_since_reset()/reset_call_counter().

        deadline: optional time.perf_counter() value (the search's time
        limit). Once it passes, the batch is abandoned: its not-yet-started
        requests are taken back off the task queue, results still in flight
        are discarded when they arrive (each request id carries its batch
        number), and DeadlineExceeded is raised -- so the time limit bounds
        the feature-building calls too, not only the search's pop loop."""
        n = len(requests)
        if n == 0:
            return []
        self._batch_no += 1
        batch = self._batch_no
        for i, (kind, args) in enumerate(requests):
            self.task_queue.put(((batch, i), kind, args))

        results: list = [None] * n
        # Generous overall safety net beyond each task's own internal
        # SIGALRM timeout -- catches a whole dead/hung worker, not just one
        # slow task (every external call has its own timeout; this is the
        # same rule one layer up).
        overall_timeout = max(self.timeout_s * n, 60.0)
        remaining = n
        while remaining > 0:
            wait = overall_timeout
            if deadline is not None:
                left = deadline - time.perf_counter()
                if left <= 0:
                    self._abandon_queued()
                    raise DeadlineExceeded(f"{remaining}/{n} results outstanding at the deadline")
                wait = min(wait, left)
            try:
                req_id, ok, payload, calls_delta = self.result_queue.get(timeout=wait)
            except queue_module.Empty:
                if wait < overall_timeout:
                    continue  # the deadline, not the safety net, ran out: re-checked above
                raise RuntimeError(
                    f"ParallelFeasibilityPool: timed out waiting for {remaining}/{n} results "
                    f"after {overall_timeout:.0f}s -- a worker may have died"
                ) from None
            req_batch, i = req_id
            if req_batch != batch:
                continue  # a late result of an abandoned batch
            if not ok:
                raise RuntimeError(f"worker task {i} ({requests[i][0]}) failed: {payload}")
            results[i] = payload
            self._calls_since_reset += calls_delta
            remaining -= 1
        return results

    def _abandon_queued(self) -> None:
        """Take this process's not-yet-started requests back off the task queue."""
        while True:
            try:
                item = self.task_queue.get_nowait()
            except queue_module.Empty:
                return
            if item is None:  # a poison pill is not ours to take
                self.task_queue.put(None)
                return

    def total_calls_since_reset(self) -> int:
        """Real MP calls made via dispatch_batch since the last
        reset_call_counter() (or pool construction). Pure in-process
        bookkeeping in the dispatcher -- accumulated synchronously as
        dispatch_batch receives each result, so it is race-free by
        construction (see the note in _worker_loop)."""
        return self._calls_since_reset

    def reset_call_counter(self) -> None:
        self._calls_since_reset = 0

    def shutdown(self) -> None:
        for _ in self.workers:
            self.task_queue.put(None)
        for w in self.workers:
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()
