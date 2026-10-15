"""Train PLOI's object-importance GNN on the label set produced by
generate_training_labels.py.

Usage (the reported weights; python baselines/ploi/run.py train):
    python baselines/ploi/scripts/train_model.py \\
        --labels-dir data/baselines/ploi/labels/train \\
        --val-labels-dir data/baselines/ploi/labels/validation \\
        --out $REAP_OUT/baselines/ploi.pt --patience 150
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ploi._bootstrap import ensure_upstream_importable  # noqa: E402

ensure_upstream_importable()

import torch  # noqa: E402

from ploi import config, features  # noqa: E402
from ploi.dataset import TrainingExample  # noqa: E402
from ploi.model import ObjectImportanceGNN  # noqa: E402
from ploi.train import train  # noqa: E402


def load_examples(labels_dir: Path) -> list[TrainingExample]:
    examples = []
    for path in sorted(labels_dir.glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        examples.append(TrainingExample.from_dict(json.loads(path.read_text())))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--val-labels-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.TRAIN_LR)
    parser.add_argument("--val-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=None,
                         help="stop early after this many epochs with no held-out improvement "
                              "(requires --val-labels-dir); omit to keep the paper's fixed-epoch recipe")
    parser.add_argument("--features", choices=["default", "positional"], default=None,
                        help="node features: default (6 inputs, the reported weights) or positional (8, + cylinder "
                             "pose); default: default, or positional when PLOI_POSITIONAL_FEATURES=1")
    args = parser.parse_args()
    if args.features is not None:
        features.set_feature_set(args.features)
    print(f"node features: {features.feature_set()} ({features.NODE_DIM} inputs)")

    examples = load_examples(args.labels_dir)
    print(f"loaded {len(examples)} training examples from {args.labels_dir}")
    if not examples:
        raise SystemExit(f"no *.json label files found in {args.labels_dir}")

    val_examples = load_examples(args.val_labels_dir) if args.val_labels_dir else None
    if val_examples:
        print(f"loaded {len(val_examples)} held-out validation examples from {args.val_labels_dir}")
    elif args.patience is not None:
        raise SystemExit("--patience requires --val-labels-dir")

    model = ObjectImportanceGNN()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    result = train(
        model, examples, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        checkpoint_path=args.out, checkpoint_every=100,
        val_examples=val_examples, val_every=args.val_every, patience=args.patience,
    )
    print(f"final train loss: {result.epoch_losses[-1]:.4f}")
    if result.best_val_loss is not None:
        print(f"best held-out val loss: {result.best_val_loss:.4f} @ epoch {result.best_epoch} "
              f"(of {len(result.epoch_losses)} epochs run"
              + (f", stopped early at {result.stopped_early_at}" if result.stopped_early_at is not None else "")
              + ")")

    torch.save(model.state_dict(), args.out)
    print(f"saved model to {args.out} (best-val-loss weights, not necessarily final epoch)"
          if result.best_val_loss is not None else f"saved model to {args.out}")


if __name__ == "__main__":
    main()
