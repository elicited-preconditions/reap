#!/usr/bin/env python
"""BT-h (Cieslar et al., CoRL 2024) on the open tabletop: instances and seed demonstrations, classifier training, and
the Table 1 evaluation (BT-h, given the plan ordering).

    python baselines/bt_h/run.py data        (instances + seed demonstrations; about 45 min of checker calls)
    python baselines/bt_h/run.py train       (data collection loop + classifier; about 30 min)
    python baselines/bt_h/run.py eval [--classifier FILE] [--nproc 1] [--shards ...]

data and train write under $REAP_OUT/baselines/bt_h/ (train: checkpoints/bt_h.pt); run them in this order.
eval runs the 300 Table 1 problems in 30 shards of 10 and merges them, in shard order, into
<out>/bt_h/per_problem.jsonl, which python scripts/report.py table1_baselines reads.
Default <out>: $REAP_OUT/baselines (outputs/baselines); default classifier: checkpoints/baselines/bt_h/bt_h.pt.
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


def shards(spec, n):
    return [int(k) for k in spec.split(",")] if spec else list(range(n))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("data", "train", "eval"))
    ap.add_argument("--out", type=Path, default=None, help="output root (default: $REAP_OUT/baselines)")
    ap.add_argument("--classifier", type=Path, default=None,
                    help="eval: classifier weights (default: checkpoints/baselines/bt_h/bt_h.pt)")
    ap.add_argument("--nproc", type=int, default=1, help="shards run at a time")
    ap.add_argument("--shards", default=None, help="comma-separated shard indices (default: all)")
    ap.add_argument("--config", default=None,
                    help="config with a baselines.bt_h section of option defaults (default: configs/tabletop.yaml)")
    C.baseline_defaults(ap, "bt_h", sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    out = a.out or out_root() / "baselines"
    logs = out / "logs"

    if a.step == "data":
        run_script(HERE / "scripts" / "generate_instances.py", [], log=logs / "bt_h_instances.log", show=".")
        run_script(HERE / "scripts" / "collect_seed_demos.py", [], log=logs / "bt_h_demos.log", show=".")
    elif a.step == "train":
        run_script(HERE / "scripts" / "run_data_collection.py", [], log=logs / "bt_h_train.log", show=".")
    else:
        clf = a.classifier or ckpt_root() / "baselines" / "bt_h" / "bt_h.pt"
        if not Path(clf).is_file():
            sys.exit(f"no {clf} (extract reap_ckpt_baselines.tar.gz at the repository root, or pass --classifier)")
        sh = out / "bt_h" / "shards"
        sh.mkdir(parents=True, exist_ok=True)
        jobs = [lambda k=k: run_script(
            HERE / "scripts" / "run_bt_h.py",
            ["--eval-root", data_root() / "tabletop" / "eval", "--classifier", clf, "--start", k * 10, "--end", k * 10 + 10,
             "--out", sh / f"shard_{k}.jsonl"], log=logs / f"bt_h_shard{k}.log") for k in shards(a.shards, 30)]
        fail = parallel(jobs, a.nproc)
        with open(out / "bt_h" / "per_problem.jsonl", "w") as fh:
            for k in range(30):
                f = sh / f"shard_{k}.jsonl"
                if f.exists():
                    fh.write(f.read_text())
        n = sum(1 for _ in open(out / "bt_h" / "per_problem.jsonl"))
        print(f"bt_h: {n} records -> {out / 'bt_h' / 'per_problem.jsonl'}")
        if fail:
            sys.exit("some evaluation shards failed (logs in %s)" % logs)


if __name__ == "__main__":
    main()
