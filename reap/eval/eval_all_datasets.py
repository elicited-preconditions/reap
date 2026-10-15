"""Shared code of the open-tabletop suite evaluation (Table 1, sample efficiency, Fig. 3 inputs).

The 15 evaluation sets (20 problems each, data/tabletop/eval/<set>), the search-mode menu passed to
``rollouts()``, checkpoint selection and problem loading. ``reap.eval.eval_single_model`` runs one model over the
suite with these helpers.

Datasets evaluated:
  * object-count scaling, no plan-length filter: obj{10..24}
  * random placement, paired N x min-plan-length: obj10/12_plan3, obj14/16_plan5
  * biased (target-blockers chain): obj10_plan5_biased2, obj12_plan7_biased3, obj14_plan9_biased3
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Relative paths (data/, checkpoints/, outputs/) resolve against the repository root.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
os.chdir(REPO_ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from reap.rl.av_star.agent import Model  # noqa: E402
from reap.rl.av_star.search import bfs, gbfs_hff, gbfs_value, rollouts  # noqa: E402
from reap.rl.data_layout import InputData  # noqa: E402
# Importing sim triggers domain-module imports (registry registration).
# filter_action is not called here directly, but eval modes that pass
# --filter_actions route through the same registry; importing here ensures
# all domains the local install supports are visible to get_domain().
from reap.rl.av_star import sim  # noqa: E402,F401
from reap.domains.registry import get_domain  # noqa: E402


# Search-mode menu, passed to rollouts() as the `func` argument.
#   greedy    : no search; pure policy/value rollout (no backtracking)
#   gbfs      : greedy best-first search using value as heuristic, w=0
#   bfs       : plain breadth-first search baseline (no heuristic, no model)
#   hff       : GBFS guided by pymimir's hFF relaxed-plan heuristic (no model)
# The keys are CLI-safe (no '*' or '.').
SEARCH_FUNCTIONS = {
    "greedy":    None,
    "gbfs":      lambda x, y, z, d, e, filter_actions=False: gbfs_value(
        x, y, z, d, e, weight=0.0, weight2=1, filter_actions=filter_actions),
    "bfs":       lambda x, y, z, d, e, filter_actions=False: bfs(
        x, y, z, d, e, filter_actions=filter_actions),
    "hff":       lambda x, y, z, d, e, filter_actions=False: gbfs_hff(
        x, y, z, d, e, filter_actions=filter_actions),
}

# ---------------------------------------------------------------------------
# Eval datasets.  Each entry is (label, validation_dir_relative_to_repo).
# The sets live under data/tabletop/eval/.
# ---------------------------------------------------------------------------

_DATA = os.environ.get("REAP_DATA") or "data"   # the dataset asset's data/ (see reap/paths.py)
EVAL_DATASETS: list[tuple[str, str]] = [
    # object-count scaling
    ("obj10",  f"{_DATA}/tabletop/eval/obj10"),
    ("obj12",  f"{_DATA}/tabletop/eval/obj12"),
    ("obj14",  f"{_DATA}/tabletop/eval/obj14"),
    ("obj16",  f"{_DATA}/tabletop/eval/obj16"),
    ("obj18",  f"{_DATA}/tabletop/eval/obj18"),
    ("obj20",  f"{_DATA}/tabletop/eval/obj20"),
    ("obj22",  f"{_DATA}/tabletop/eval/obj22"),
    ("obj24",  f"{_DATA}/tabletop/eval/obj24"),

    # paired (N, min-plan-len), random placement
    ("obj10_plan3",  f"{_DATA}/tabletop/eval/obj10_plan3"),
    ("obj12_plan3",  f"{_DATA}/tabletop/eval/obj12_plan3"),
    ("obj14_plan5",  f"{_DATA}/tabletop/eval/obj14_plan5"),
    ("obj16_plan5",  f"{_DATA}/tabletop/eval/obj16_plan5"),

    # paired (N, min-plan-len), biased with a --target-blockers chain
    ("obj10_plan5_biased2",  f"{_DATA}/tabletop/eval/obj10_plan5_biased2"),
    ("obj12_plan7_biased3",  f"{_DATA}/tabletop/eval/obj12_plan7_biased3"),
    ("obj14_plan9_biased3",  f"{_DATA}/tabletop/eval/obj14_plan9_biased3"),
]

# ---------------------------------------------------------------------------
# Rollout configuration.
# ---------------------------------------------------------------------------

STOCHASTIC = False
CLOSED     = True


# ---------------------------------------------------------------------------

def latest_checkpoint(save_path: Path) -> Path:
    """Pick save_path/checkpoint/checkpoint_<max N>/ if any exist; else save_path itself."""
    save_path = Path(save_path)
    ck_dir = save_path / "checkpoint"
    if ck_dir.exists():
        ck_dirs = [p for p in ck_dir.glob("checkpoint_*") if p.is_dir()]
        if ck_dirs:
            return max(ck_dirs, key=lambda p: int(p.name.split("_")[1]))
    return save_path


def reference_problem_dir(domain_file: str, domain: str = "tabletop") -> str:
    """Return the data_path that supplies *domain_file* for InputData.

    Eval suites ship problem instances but no domain.pddl; InputData
    therefore borrows the domain from a "reference problem dir", the data
    folder of the environment.  Looked up via the domain registry.
    """
    return get_domain(domain).reference_problem_dir


def eval_suites_for(domain: str) -> list[tuple[str, str]]:
    """Lookup the per-domain eval-suite list from the registry."""
    return [(s.label, s.path) for s in get_domain(domain).eval_suites]


def load_problems(suite_path: Path, domain_file: str, domain: str = "tabletop"):
    """Use InputData with train_dir overridden to the suite's absolute path.

    ``domain`` selects the registered :class:`DomainSpec` used to find the
    reference problem dir (which holds domain.pddl; eval suites only ship
    problem instances).
    """
    data = InputData(
        reference_problem_dir(domain_file, domain=domain),
        train_dir=str(suite_path),
        instances="all",
        domain_file=domain_file,
    )
    return data.train_problems


def run_rollouts(model, problems, *, time_limit_s: int, max_budget: int,
                 val_retries: int, search_fn, filter_actions: bool = False):
    # rmse is only computed in the default (greedy) rollout path; when a
    # search function is supplied the upstream rollouts() never defines
    # rmse and would raise UnboundLocalError on append.  Disable rmse
    # whenever search_fn is active.
    #
    # filter_actions only affects the search-based modes (gbfs/bfs/hff):
    # it routes the physical-feasibility check (filter_action() from
    # tampanda) through the GBFS loop body.  Greedy already calls
    # filter_action inside rollout() up to ``val_retries`` times; that
    # plumbing is independent.
    return rollouts(
        problems, model, max_budget,
        time_limit_s=time_limit_s,
        closed=CLOSED, stochastic=STOCHASTIC,
        compute_rmse=(search_fn is None),
        max_infeasible_retries=val_retries,
        func=search_fn,
        filter_actions=filter_actions,
        # MP/plan accounting of the Table 1 evaluation (distinct (state, action) queries): per-problem reset of
        # the filter counters and verdict cache, columns mp_calls / total_filter_calls / infeasible_steps
        count_distinct_queries=True,
    )
