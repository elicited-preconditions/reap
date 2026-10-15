#!/usr/bin/env python
"""SAHS (Kim and Shimanuki, CoRL 2019; Kim et al., IJRR 2022) on the open tabletop: training data, network training,
and the Table 1 evaluation.

    python baselines/sahs/run.py data  [--workers 16] [--nproc 1] [--shards 0,1,...]
    python baselines/sahs/run.py train
    python baselines/sahs/run.py eval  [--q-pick FILE --q-put FILE] [--workers 16] [--nproc 1] [--shards ...]

data   the 600 training problems in 30 shards of 20 -> <out>/sahs_training/training/shards/shard_<k>.jsonl,
       concatenated in file-name order (shard_0, shard_1, shard_10, ...) into <out>/sahs_training/training/examples.jsonl
train  the pick and the put network (duplicate filter) -> <out>/sahs_training/results/models/sahs_{pick,put}.pt
       (eval: --q-pick, --q-put)
eval   the 300 Table 1 problems in 12 shards of 25 (300 s, at most 3000 pops per problem) -> <out>/sahs/shard_<k>.json,
       which python scripts/report.py table1_baselines reads
Default <out>: $REAP_OUT/baselines (outputs/baselines); default weights: checkpoints/baselines/sahs/sahs_{pick,put}.pt.
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


def shards(spec, n):
    return [int(k) for k in spec.split(",")] if spec else list(range(n))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("data", "train", "eval"))
    ap.add_argument("--out", type=Path, default=None, help="output root (default: $REAP_OUT/baselines)")
    ap.add_argument("--workers", type=int, default=16, help="checker processes per shard")
    ap.add_argument("--nproc", type=int, default=1, help="shards run at a time")
    ap.add_argument("--shards", default=None, help="comma-separated shard indices (default: all)")
    ap.add_argument("--q-pick", type=Path, default=None,
                    help="eval: pick network (default: checkpoints/baselines/sahs/sahs_pick.pt)")
    ap.add_argument("--q-put", type=Path, default=None,
                    help="eval: put network (default: checkpoints/baselines/sahs/sahs_put.pt)")
    ap.add_argument("--config", default=None,
                    help="config with a baselines.sahs section of option defaults (default: configs/tabletop.yaml)")
    C.baseline_defaults(ap, "sahs", sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    out = a.out or out_root() / "baselines"
    ds = data_root() / TRAIN_SET
    work = out / "sahs_training"
    logs = out / "logs"

    if a.step == "data":
        (work / "training" / "shards").mkdir(parents=True, exist_ok=True)
        jobs = [lambda k=k: run_script(
            HERE / "scripts" / "generate_training_data.py",
            ["--problems-dir", ds / "train", "--domain", ds / "domain.pddl", "--start", k * 20, "--end", k * 20 + 20,
             "--workers", a.workers, "--out", work / "training" / "shards" / f"shard_{k}.jsonl"],
            log=logs / f"sahs_data_shard{k}.log") for k in shards(a.shards, 30)]
        if parallel(jobs, a.nproc):
            sys.exit("some data shards failed (logs in %s)" % logs)
        # concatenated in file-name order (shard_0, shard_1, shard_10, ...), the order of the released weights' data
        parts = sorted((work / "training" / "shards").glob("shard_*.jsonl"), key=lambda p: p.name)
        with open(work / "training" / "examples.jsonl", "wb") as fh:
            for p in parts:
                fh.write(p.read_bytes())
        print(f"{len(parts)} shards -> {work / 'training' / 'examples.jsonl'}")
    elif a.step == "train":
        run_script(HERE / "scripts" / "train_networks.py",
                   ["--examples", work / "training" / "examples.jsonl", "--out-dir", work, "--dedup-filter"],
                   log=logs / "sahs_train.log", show=".")
    else:
        qp = a.q_pick or ckpt_root() / "baselines" / "sahs" / "sahs_pick.pt"
        qu = a.q_put or ckpt_root() / "baselines" / "sahs" / "sahs_put.pt"
        for f in (qp, qu):
            if not Path(f).is_file():
                sys.exit(f"no {f} (extract reap_ckpt_baselines.tar.gz at the repository root, or pass --q-pick/--q-put)")
        (out / "sahs").mkdir(parents=True, exist_ok=True)
        jobs = [lambda k=k: run_script(
            HERE / "scripts" / "evaluate.py",
            ["--eval-root", data_root() / "tabletop" / "eval", "--domain", ds / "domain.pddl", "--q-pick", qp, "--q-put", qu,
             "--start", k * 25, "--end", k * 25 + 25, "--workers", a.workers, "--time-limit-s", 300,
             "--max-pops", 3000, "--out", out / "sahs" / f"shard_{k}.json"],
            log=logs / f"sahs_eval_shard{k}.log") for k in shards(a.shards, 12)]
        if parallel(jobs, a.nproc):
            sys.exit("some evaluation shards failed (logs in %s)" % logs)


if __name__ == "__main__":
    main()
