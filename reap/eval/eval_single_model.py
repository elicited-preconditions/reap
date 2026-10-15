"""Evaluate ONE trained model on the standard 15-dataset eval suite.

Takes the model save_path and label as CLI args and reuses EVAL_DATASETS,
latest_checkpoint, load_problems, run_rollouts and SEARCH_FUNCTIONS from
eval_all_datasets, so the rollout semantics are the same for every model.

Usage (a released checkpoint is loaded as given with --snapshot live):
    python -m reap.eval.eval_single_model \\
        --save_path checkpoints/tabletop/seed_1 --snapshot live \\
        --domain_file domain.pddl \\
        --label seed_1 \\
        --output_dir $REAP_OUT/eval/tabletop \\
        --search_mode gbfs --val_retries 3 --max_budget 16384 --time_limit_s 300 --filter_actions
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
os.chdir(REPO_ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import torch  # noqa: E402

# Reuse everything from the canonical eval driver so semantics stay locked.
from reap.eval.eval_all_datasets import (  # noqa: E402
    SEARCH_FUNCTIONS,
    eval_suites_for,
    latest_checkpoint,
    load_problems,
    run_rollouts,
)
from reap.rl.av_star.agent import Model  # noqa: E402


def main(
    save_path: str,
    domain_file: str,
    label: str,
    *,
    output_dir: str,
    search_mode: str = "gbfs",
    time_limit_s: int = 300,
    max_budget: int = 16384,
    val_retries: int = 3,
    snapshot: str = "best",
    filter_actions: bool = False,
    domain: str = "tabletop",
    only_datasets: str | None = None,
):
    out = Path(output_dir) / label
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if search_mode not in SEARCH_FUNCTIONS:
        raise SystemExit(f"unknown --search_mode {search_mode!r}; valid: {list(SEARCH_FUNCTIONS)}")
    search_fn = SEARCH_FUNCTIONS[search_mode]

    save = REPO_ROOT / save_path
    if not save.exists():
        raise SystemExit(f"model save_path does not exist: {save}")

    ck = save if snapshot == "live" else latest_checkpoint(save)

    print(f"label:        {label}")
    print(f"save_path:    {save}")
    print(f"checkpoint:   {ck.name}  (snapshot={snapshot})")
    print(f"domain_file:  {domain_file}")
    print(f"search_mode:  {search_mode}  time_limit={time_limit_s}s  max_budget={max_budget}  val_retries={val_retries}")
    print(f"filter_actions: {filter_actions}  "
          f"({'physical feasibility check ENABLED during search' if filter_actions else 'symbolic-only, filter_action NOT called'})")
    print(f"output dir:   {out.resolve()}")
    print()

    # Pre-flight on eval suites, looked up from the domain registry.
    valid_datasets: list[tuple[str, Path]] = []
    keys = [k.strip() for k in (only_datasets or "").split(",") if k.strip()]
    for ds_label, rel_path in eval_suites_for(domain):
        if keys and ds_label not in keys:
            continue
        p = REPO_ROOT / rel_path
        n = len(list(p.glob("*.pddl"))) if p.exists() else 0
        if not p.exists() or n == 0:
            print(f"  SKIP   dataset {ds_label:<24s}  (missing or empty: {p})")
            continue
        print(f"  OK    dataset {ds_label:<24s}  problems={n}")
        valid_datasets.append((ds_label, p))

    if not valid_datasets:
        raise SystemExit("no usable eval datasets found")

    print(f"\n=== loading model: {label} ({domain_file}) ===")
    model = Model.load_from_folder(str(ck))
    model.to(device)

    per_problem_rows: list[pd.DataFrame] = []
    per_dataset_pct: dict[str, float | None] = {}

    for ds_label, ds_path in valid_datasets:
        try:
            problems = load_problems(ds_path, domain_file, domain=domain)
        except Exception as e:
            print(f"  {ds_label:<24s} FAILED to load problems: {type(e).__name__}: {e}")
            per_dataset_pct[ds_label] = None
            continue
        if not problems:
            per_dataset_pct[ds_label] = None
            continue

        df = run_rollouts(
            model, problems,
            time_limit_s=time_limit_s, max_budget=max_budget,
            val_retries=val_retries, search_fn=search_fn,
            filter_actions=filter_actions,
        )
        n_total = len(df)
        n_solved = int((df["solved"] == 1).sum())
        pct = 100.0 * n_solved / max(n_total, 1)
        per_dataset_pct[ds_label] = round(pct, 1)
        print(f"  {ds_label:<24s} {n_solved:>3}/{n_total:<3}  ({pct:5.1f}%)")

        df = df.copy()
        df.insert(0, "model", label)
        df.insert(1, "domain_file", domain_file)
        df.insert(2, "dataset", ds_label)
        per_problem_rows.append(df)

        df.to_csv(out / f"{label}__{ds_label}.csv", index=False)

    if per_problem_rows:
        per_problem = pd.concat(per_problem_rows, ignore_index=True)
        per_problem.to_csv(out / "per_problem.csv", index=False)
        print(f"\n  wrote {len(per_problem)} rows -> {out / 'per_problem.csv'}")

    summary = pd.DataFrame([{"model": label, "domain": domain_file, **per_dataset_pct}])
    summary.set_index("model").to_csv(out / "summary.csv")
    print(f"  wrote summary -> {out / 'summary.csv'}\n")

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.set_index("model"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save_path", required=True,
                    help="model save dir: relative to the repo root, or absolute")
    ap.add_argument("--domain_file", required=True,
                    help="domain file name (domain.pddl)")
    ap.add_argument("--label", required=True,
                    help="short label used to name output CSVs")
    ap.add_argument("--output_dir", default=(os.environ.get("REAP_OUT") or "outputs") + "/eval/single_model",
                    help="parent dir; per-model results go in <output_dir>/<label>/")
    ap.add_argument("--search_mode", default="gbfs", choices=list(SEARCH_FUNCTIONS))
    ap.add_argument("--time_limit_s", type=int, default=300)
    ap.add_argument("--max_budget",   type=int, default=16384)
    ap.add_argument("--val_retries",  type=int, default=3)
    ap.add_argument("--snapshot",     default="best", choices=("best", "live"),
                    help="best = latest_validator_checkpoint; live = save_path itself")
    ap.add_argument("--filter_actions", action="store_true",
                    help="Enable filter_action() during search-based modes (gbfs/bfs/hff); the search "
                         "protocols of configs/tabletop.yaml set it. When off (default), search runs on the "
                         "symbolic domain alone and can return symbolically valid but physically infeasible "
                         "plans. Greedy calls filter_action through --val_retries; this flag is ignored for "
                         "--search_mode=greedy.")
    ap.add_argument("--domain", default="tabletop",
                    help="Registered domain name (key in reap.domains.registry). "
                         "Used to look up the eval-suite list + reference_problem_dir. "
                         "Default 'tabletop' (the open tabletop); pass e.g. 'multilevel-blocks' "
                         "for 3D stacking.")
    ap.add_argument("--only_datasets", default=None,
                    help="comma-separated eval-suite labels (e.g. obj10,obj24); default: every suite of the domain")
    args = ap.parse_args()
    main(**vars(args))
