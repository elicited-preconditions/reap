#!/usr/bin/env python
"""Trains the SAHS pick and put networks on the examples of generate_training_data.py (sahs/train.py: the large-margin
ranking loss of the IJRR 2022 version, Adam, learning rate 1e-3, full batch) and writes
<out-dir>/results/models/sahs_{pick,put}.pt.

    python baselines/sahs/scripts/train_networks.py \\
        [--examples $REAP_OUT/baselines/sahs_training/training/examples.jsonl] \\
        [--out-dir $REAP_OUT/baselines/sahs_training]

The defaults (--dedup-filter, --epochs 300) are the configuration of the released weights (python
baselines/sahs/run.py train). --no-dedup-filter keeps the candidates with the same input features as the chosen one in
the ranking loss, as the paper's loss is written; the put network then does not learn.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from sahs.dataset import read_jsonl  # noqa: E402
from sahs.features import DEFAULT_EDGE_FEATURES  # noqa: E402
from sahs.train import train_operator  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    work = Path(os.environ.get("REAP_OUT") or "outputs") / "baselines" / "sahs_training"
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--examples", type=Path, default=work / "training" / "examples.jsonl")
    parser.add_argument("--out-dir", type=Path, default=work)
    parser.add_argument("--dedup-filter", action=argparse.BooleanOptionalAction, default=True,
                        help="exclude the candidates with the same input features as the chosen one from the ranking "
                             "loss (default, the released weights); --no-dedup-filter keeps them, as the paper's loss "
                             "is written")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None, help="first N examples per operator (short test runs)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_root = args.out_dir

    examples = read_jsonl(args.examples)
    pick_examples = [e for e in examples if e.operator == "pick"]
    put_examples = [e for e in examples if e.operator == "put"]
    if args.limit is not None:
        pick_examples, put_examples = pick_examples[:args.limit], put_examples[:args.limit]
    print(f"loaded {len(examples)} examples ({len(pick_examples)} pick, {len(put_examples)} put)")

    models_dir = out_root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    log: dict = {}
    for op_name, op_examples in (("pick", pick_examples), ("put", put_examples)):
        print(f"training q_{op_name} on {len(op_examples)} examples...")
        t0 = time.perf_counter()
        net, history = train_operator(op_examples, epochs=args.epochs, dedup_filter=args.dedup_filter)
        elapsed = time.perf_counter() - t0
        torch.save(net.state_dict(), models_dir / f"sahs_{op_name}.pt")
        log[op_name] = history
        print(f"  final: margin_loss={history[-1]['margin_loss']:.4f}, wall_s={elapsed:.1f}", flush=True)

    log["config"] = {"edge_features": DEFAULT_EDGE_FEATURES, "dedup_filter": args.dedup_filter,
                     "epochs": args.epochs, "lr": 1e-3, "examples": str(args.examples), "limit": args.limit}
    log_path = out_root / "results" / "training_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"wrote {log_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, op_name in zip(axes, ("pick", "put")):
            history = log[op_name]
            epochs = [h["epoch"] for h in history]
            ax.plot(epochs, [h["margin_loss"] for h in history], label="margin loss")
            ax.set_title(f"q_{op_name}")
            ax.set_xlabel("epoch")
            ax.legend()
        fig.tight_layout()
        plot_path = out_root / "results" / "training_loss.png"
        fig.savefig(plot_path)
        print(f"wrote {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping training_loss.png")


if __name__ == "__main__":
    main()
