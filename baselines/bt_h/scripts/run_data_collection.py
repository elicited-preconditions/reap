#!/usr/bin/env python3
"""The full Algorithm 3 bootstrap data-collection loop (descending prefix
length l, capture-tree BT-h, iterative retraining). Requires
scripts/collect_seed_demos.py to have already populated <out>/data/demos/
(<out> = $REAP_OUT/baselines/bt_h).

Saves the final classifier checkpoint to <out>/checkpoints/bt_h.pt,
the growing positive/negative example counts and per-iteration retrain
metrics to <out>/data/classifier_dataset/history.json, and the raw examples
themselves (sigma/pi_full/label, JSON) to <out>/data/classifier_dataset/ for
inspection or reuse without re-running the (expensive) search.

    python baselines/bt_h/scripts/run_data_collection.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# outputs: $REAP_OUT/baselines/bt_h/
WORK = Path(__import__("os").environ.get("REAP_OUT") or "outputs") / "baselines" / "bt_h"

from bt_h._bootstrap import ensure_upstream_importable  # noqa: E402

ensure_upstream_importable()

import torch  # noqa: E402

from bt_h import config  # noqa: E402
from bt_h.checker import InstrumentedFeasibilityChecker  # noqa: E402
from bt_h.classifier.inference import TrainedClassifier  # noqa: E402
from bt_h.data_collection import load_demonstrations, run_data_collection_loop  # noqa: E402
from bt_h.types import GroundAction  # noqa: E402


def _example_to_json(ex) -> dict:
    return {
        "sigma": ex.sigma,
        "pi_full": [list(a) for a in ex.pi_full],
        "label": ex.label,
    }


def main() -> None:
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    demos = load_demonstrations(WORK / "data" / "demos")
    print(f"loaded {len(demos)} seed demonstrations")
    if not demos:
        raise SystemExit("no demonstrations found -- run scripts/collect_seed_demos.py first")

    checker = InstrumentedFeasibilityChecker()
    t0 = time.time()
    classifier, positives, negatives, history = run_data_collection_loop(
        demos, checker, seed=0,
        r_dp=config.DATA_COLLECTION_R_DP,
        train_max_steps=config.DATA_COLLECTION_TRAIN_MAX_STEPS,
        checkpoint_dir=WORK / "checkpoints" / "data_collection_progress",
    )
    elapsed = time.time() - t0

    out_dir = WORK / "data" / "classifier_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "positives.json").write_text(json.dumps([_example_to_json(e) for e in positives], indent=2))
    (out_dir / "negatives.json").write_text(json.dumps([_example_to_json(e) for e in negatives], indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    ckpt_dir = WORK / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(classifier, TrainedClassifier):
        torch.save(classifier.model.state_dict(), ckpt_dir / "bt_h.pt")
        print(f"saved trained classifier -> {ckpt_dir / 'bt_h.pt'}")
    else:
        print("WARNING: final classifier is still the constant-1 fallback "
              "(never had enough of both classes to train) -- check history.json")

    print(f"\nAlgorithm 3 loop complete in {elapsed:.1f}s wall-clock, "
          f"{checker.stats.total_calls} total checker calls")
    print(f"{len(positives)} positive / {len(negatives)} negative examples collected")
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
