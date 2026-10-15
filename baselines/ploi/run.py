#!/usr/bin/env python
"""PLOI (Silver et al., AAAI 2021) on the open tabletop: training labels, GNN training, and the two Table 1 rows.

    python baselines/ploi/run.py labels [--workers 16] [--nproc 1]      (regenerate the shipped labels)
    python baselines/ploi/run.py train  [--labels DIR]
    python baselines/ploi/run.py eval   [--mode ploi_hff|ploi_bfs] [--model FILE] [--nproc 1] [--shards ...]

labels  12 shards of 50 training and 3 shards of 16 validation problems (bfs_then_hff, single_plan) ->
        <out>/ploi_labels/{train,validation}
train   the GNN (patience 150) from --labels (default: the shipped labels, data/baselines/ploi/labels)
        -> <out>/ploi.pt
eval    the 300 Table 1 problems in 30 shards of 10 -> <out>/ploi_hff/shard_<k>.jsonl or <out>/ploi_brfs/shard_<k>.jsonl,
        which python scripts/report.py table1_baselines reads. Each shard is located by its index, so keep the
        sharding.
Default <out>: $REAP_OUT/baselines (outputs/baselines); default model: checkpoints/baselines/ploi/ploi.pt.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from reap.eval.runners import parallel, run_script  # noqa: E402
from reap import config as C  # noqa: E402
from reap.paths import ckpt_root, data_root, out_root  # noqa: E402

TRAIN_SET = "tabletop"   # training problems and domain ($REAP_DATA)
OUT_DIR = {"ploi_hff": "ploi_hff", "ploi_bfs": "ploi_brfs"}


def shards(spec, n):
    return [int(k) for k in spec.split(",")] if spec else list(range(n))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("labels", "train", "eval"))
    ap.add_argument("--out", type=Path, default=None, help="output root (default: $REAP_OUT/baselines)")
    ap.add_argument("--mode", choices=sorted(OUT_DIR), action="append", default=None,
                    help="eval: ploi_hff and/or ploi_bfs (default: both)")
    ap.add_argument("--labels", type=Path, default=None,
                    help="train: labels directory with train/ and validation/ (default: $REAP_DATA/baselines/ploi/labels)")
    ap.add_argument("--model", type=Path, default=None,
                    help="eval: GNN weights (default: checkpoints/baselines/ploi/ploi.pt)")
    ap.add_argument("--workers", type=int, default=16, help="labels: checker processes per shard")
    ap.add_argument("--nproc", type=int, default=1, help="shards run at a time")
    ap.add_argument("--shards", default=None, help="comma-separated shard indices (default: all)")
    ap.add_argument("--config", default=None,
                    help="config with a baselines.ploi section of option defaults (default: configs/tabletop.yaml)")
    C.baseline_defaults(ap, "ploi", sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    out = a.out or out_root() / "baselines"
    ds = data_root() / TRAIN_SET
    logs = out / "logs"

    if a.step == "labels":
        lab = out / "ploi_labels"
        jobs = []
        for sub, name, n, size in (("train", "train", 12, 50), ("validation", "validation", 3, 16)):
            (lab / name).mkdir(parents=True, exist_ok=True)
            jobs += [lambda k=k, sub=sub, name=name, size=size: run_script(
                HERE / "scripts" / "generate_training_labels.py",
                ["--problems-dir", ds / sub, "--domain", ds / "domain.pddl", "--out-dir", lab / name,
                 "--start", k * size, "--end", k * size + size, "--workers", a.workers, "--plan-fn", "bfs_then_hff",
                 "--max-expansion", 5000, "--time-limit-s", 120, "--strategy", "single_plan"],
                env={"COLLISION_POOL_WORKERS": "1"}, log=logs / f"ploi_labels_{sub}_shard{k}.log")
                for k in shards(a.shards, n)]
        if parallel(jobs, a.nproc):
            sys.exit("some label shards failed (logs in %s)" % logs)
    elif a.step == "train":
        labels = a.labels or data_root() / "baselines" / "ploi" / "labels"
        out.mkdir(parents=True, exist_ok=True)
        run_script(HERE / "scripts" / "train_model.py",
                   ["--labels-dir", labels / "train", "--val-labels-dir", labels / "validation",
                    "--out", out / "ploi.pt", "--patience", 150], log=logs / "ploi_train.log", show=".")
    else:
        model = a.model or ckpt_root() / "baselines" / "ploi" / "ploi.pt"
        if not Path(model).is_file():
            sys.exit(f"no {model} (extract reap_ckpt_baselines.tar.gz at the repository root, or pass --model)")
        fail = 0
        for mode in a.mode or ["ploi_hff", "ploi_bfs"]:
            (out / OUT_DIR[mode]).mkdir(parents=True, exist_ok=True)
            jobs = [lambda k=k, mode=mode: run_script(
                HERE / "scripts" / "run_evaluation.py",
                ["--eval-root", data_root() / "tabletop" / "eval", "--domain", ds / "domain.pddl", "--mode", mode,
                 "--model", model, "--start", k * 10, "--end", k * 10 + 10,
                 "--out", out / OUT_DIR[mode] / f"shard_{k}.jsonl"],
                env={"COLLISION_POOL_WORKERS": "3", "FILTER_TIMEOUT_S": "30"},
                log=logs / f"{mode}_shard{k}.log") for k in shards(a.shards, 30)]
            fail += parallel(jobs, a.nproc)
        if fail:
            sys.exit("some evaluation shards failed (logs in %s)" % logs)


if __name__ == "__main__":
    main()
