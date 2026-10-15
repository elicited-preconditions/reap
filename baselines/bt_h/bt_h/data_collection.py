"""Algorithm 3 (Sec 4.3 of Cieslar et al.), applied to this domain. Two
phases:

  1. SEED-DEMONSTRATION generation (collect_seed_demonstrations): run
     classic_backtrack (Algorithm 1) over a batch of freshly generated
     instances, before any classifier exists. Every prefix of a successful
     demonstration's state sequence is a free positive training example.

  2. The DESCENDING-PREFIX-LENGTH BOOTSTRAP LOOP (run_data_collection_loop):
     starting from h := Constant(1), iterate l from (the longest demo's
     plan length - 2) down to 0; at each l, sample r_dp demos with plan
     length >= l+2, restart bt_h_capture_tree from each one's real prefix
     sigma[0..l], harvest every node on a discovered successful path as
     positive and every true dead-end as negative (see bt_h.py's
     _capture_tree docstring), retrain the classifier on the growing
     dataset, and move to the next (shorter) l: Algorithm 3, with
     r_dp = 30 search trees per level (config.DATA_COLLECTION_R_DP; the
     paper: 4,000), each keeping all its nodes as examples.
"""
from __future__ import annotations

import json
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from bt_h import config
from bt_h.backtracking import classic_backtrack
from bt_h.bt_h import bt_h_capture_tree
from bt_h.checker import InstrumentedFeasibilityChecker
from bt_h.classifier.dataset import FeasibilityDataset, FeasibilityExample
from bt_h.classifier.inference import ConstantClassifier, TrainedClassifier
from bt_h.classifier.train import train_classifier
from bt_h.domain.problem_gen import Instance, emit_pddl
from bt_h.domain.task_planner import plan_task
from bt_h.state import initial_state
from bt_h.types import GroundAction


@dataclass
class Demonstration:
    """One successfully solved instance: pi (task plan), thetas (the
    controller-parameter sequence actually used), sigma (the resulting
    real state sequence, sigma_0..sigma_n). Exactly the paper's <pi, p, sigma>
    demonstration triple (p == thetas here, since our controllers are
    otherwise parameterless)."""

    name: str
    k: int
    pi: list
    thetas: list
    sigma: list

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "k": self.k,
            "pi": [list(a) for a in self.pi],
            "thetas": self.thetas,
            "sigma": self.sigma,
        }

    @staticmethod
    def from_json(d: dict) -> "Demonstration":
        return Demonstration(
            name=d["name"],
            k=d["k"],
            pi=[GroundAction(*a) for a in d["pi"]],
            thetas=d["thetas"],
            sigma=d["sigma"],
        )


def collect_seed_demonstrations(
    instances: list[Instance],
    checker: InstrumentedFeasibilityChecker,
    out_dir: Path | None = None,
    max_calls_per_instance: int = 300,
    time_limit_s_per_instance: float = 90.0,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[list[Demonstration], list[dict]]:
    """Run classic_backtrack over `instances`. Returns (successful
    Demonstrations FROM THIS CALL, per-instance attempt logs FROM THIS
    CALL); the logs measure the actual yield.

    If out_dir is given, each demo is saved immediately upon success
    (rather than only at the very end), and any instance whose demo file
    already exists there is skipped entirely -- makes this resumable after
    an interruption without
    losing already-completed work or re-spending compute re-solving
    instances already solved. Callers doing a resumed run should use
    load_demonstrations(out_dir) afterward to get the TRUE full result set
    across all invocations, not just this call's return value."""
    demos: list[Demonstration] = []
    logs: list[dict] = []

    for i, instance in enumerate(instances):
        if out_dir is not None and (out_dir / f"{instance.name}.json").exists():
            if verbose:
                print(f"[{i + 1}/{len(instances)}] {instance.name} (K={instance.k}): SKIP (already collected)")
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".pddl", delete=False) as f:
            f.write(emit_pddl(instance))
            problem_path = f.name

        pi = plan_task(config.relocate_domain_path(), problem_path)
        sigma0 = initial_state(instance.cylinder_cells)
        rng = random.Random(seed + i)

        t0 = time.time()
        result = classic_backtrack(
            sigma0, pi, checker, rng=rng,
            max_calls=max_calls_per_instance, time_limit_s=time_limit_s_per_instance,
        )
        elapsed = time.time() - t0

        logs.append({
            "name": instance.name,
            "k": instance.k,
            "solved": result.solved,
            "checker_calls": result.checker_calls,
            "wall_time_s": round(elapsed, 2),
            "termination_reason": result.termination_reason,
        })
        if verbose:
            status = "OK" if result.solved else f"FAIL({result.termination_reason})"
            print(f"[{i + 1}/{len(instances)}] {instance.name} (K={instance.k}): {status} "
                  f"calls={result.checker_calls} wall={elapsed:.1f}s")

        if result.solved:
            demo = Demonstration(name=instance.name, k=instance.k, pi=pi,
                                  thetas=result.thetas, sigma=result.sigma)
            demos.append(demo)
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{demo.name}.json").write_text(json.dumps(demo.to_json(), indent=2))

    return demos, logs


def save_demonstrations(demos: list[Demonstration], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for demo in demos:
        (out_dir / f"{demo.name}.json").write_text(json.dumps(demo.to_json(), indent=2))


def load_demonstrations(in_dir: Path) -> list[Demonstration]:
    return [
        Demonstration.from_json(json.loads(p.read_text()))
        for p in sorted(in_dir.glob("*.json"))
        if p.name != "_collection_log.json"
    ]


def seed_positive_examples(demos: list[Demonstration]) -> list[FeasibilityExample]:
    """Every prefix sigma[0..j] of every successful demonstration is a free
    positive example for the FULL original plan pi (paper Sec 4.3: "any
    prefix of the controller plan... and its associated sequence of states...
    should be accepted by our classifier"), added before Algorithm 3's
    loop. The loop itself only finds positives from capture-tree runs that
    start at a prefix sigma_l, so without these the early states of long
    plans would have no positive example."""
    return [
        FeasibilityExample(sigma=demo.sigma[:j], pi_full=demo.pi, label=1)
        for demo in demos
        for j in range(1, len(demo.sigma) + 1)
    ]


def _checkpoint_progress(checkpoint_dir: Path, classifier, positives, negatives, history) -> None:
    """Overwrite a running checkpoint after each l-iteration -- a safety
    net against interruption (this loop has no per-iteration resume logic
    of its own). Not a full resume
    mechanism (the caller would need to manually wire a partial
    positives/negatives/classifier back in), just enough to avoid losing
    ALL progress if the process dies."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))
    (checkpoint_dir / "positives_checkpoint.json").write_text(
        json.dumps([{"sigma": e.sigma, "pi_full": [list(a) for a in e.pi_full], "label": e.label} for e in positives])
    )
    (checkpoint_dir / "negatives_checkpoint.json").write_text(
        json.dumps([{"sigma": e.sigma, "pi_full": [list(a) for a in e.pi_full], "label": e.label} for e in negatives])
    )
    if isinstance(classifier, TrainedClassifier):
        import torch
        torch.save(classifier.model.state_dict(), checkpoint_dir / "classifier_checkpoint.pt")


def run_data_collection_loop(
    demos: list[Demonstration],
    checker: InstrumentedFeasibilityChecker,
    r_dp: int = 20,
    max_calls_per_capture: int = 100,
    time_limit_s_per_capture: float = 30.0,
    train_max_steps: int = 200,
    min_examples_per_class_to_train: int = 5,
    seed: int = 0,
    verbose: bool = True,
    checkpoint_dir: Path | None = None,
):
    """Algorithm 3's full descending-prefix-length bootstrap loop (module
    docstring, phase 2). Returns (final_classifier, positives, negatives,
    history) -- final_classifier is a ConstantClassifier if no retraining
    ever had enough data of both classes (shouldn't happen in practice at
    this domain's scale, but a possible degenerate case worth reporting
    rather than crashing on).

    If checkpoint_dir is given, progress (classifier weights, positives/
    negatives, history) is overwritten there after every l-iteration, as a
    safety net against interruption -- NOT a full auto-resume (unlike
    collect_seed_demonstrations' out_dir), just enough to recover the
    latest state manually rather than losing everything."""
    rng = random.Random(seed)
    classifier = ConstantClassifier()
    positives: list[FeasibilityExample] = seed_positive_examples(demos)
    negatives: list[FeasibilityExample] = []
    history: list[dict] = []
    if verbose:
        print(f"seeded {len(positives)} trivial positive examples from {len(demos)} demonstrations "
              f"(every prefix of every successful demo)")

    max_n = max(len(demo.pi) for demo in demos)
    for l in range(max_n - 2, -1, -1):
        eligible = [demo for demo in demos if len(demo.pi) >= l + 2]
        if not eligible:
            if verbose:
                print(f"l={l}: no eligible demos (need plan length >= {l + 2}), skipping")
            continue

        sampled = [rng.choice(eligible) for _ in range(r_dp)]
        iter_pos, iter_neg = 0, 0
        for demo in sampled:
            sigma_pref = demo.sigma[: l + 1]
            tree = bt_h_capture_tree(
                sigma_pref, demo.pi, checker, classifier,
                rng=random.Random(rng.randrange(2**31)),
                max_calls=max_calls_per_capture, time_limit_s=time_limit_s_per_capture,
            )
            for s in tree.successes:
                positives.append(FeasibilityExample(sigma=s, pi_full=demo.pi, label=1))
                iter_pos += 1
            for s in tree.stuck:
                negatives.append(FeasibilityExample(sigma=s, pi_full=demo.pi, label=0))
                iter_neg += 1

        val_loss, val_acc = None, None
        if len(positives) >= min_examples_per_class_to_train and len(negatives) >= min_examples_per_class_to_train:
            all_examples = positives + negatives
            shuffled = list(all_examples)
            rng.shuffle(shuffled)
            n_val = max(1, len(shuffled) // 5)
            val_ds = FeasibilityDataset(shuffled[:n_val], seed=rng.randrange(2**31))
            train_ds = FeasibilityDataset(shuffled[n_val:], seed=rng.randrange(2**31))
            model, hist = train_classifier(train_ds, val_ds, max_steps=train_max_steps, verbose=False)
            classifier = TrainedClassifier(model, rng=random.Random(rng.randrange(2**31)))
            val_loss = hist["val_loss"][-1] if hist["val_loss"] else None
            val_acc = hist["val_acc"][-1] if hist["val_acc"] else None

        history.append({
            "l": l,
            "n_eligible": len(eligible),
            "n_sampled": len(sampled),
            "new_positives": iter_pos,
            "new_negatives": iter_neg,
            "total_positives": len(positives),
            "total_negatives": len(negatives),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "checker_calls_cumulative": checker.stats.total_calls,
        })
        if verbose:
            print(f"l={l}: +{iter_pos} pos, +{iter_neg} neg "
                  f"(totals: {len(positives)}/{len(negatives)}) "
                  f"val_loss={val_loss} val_acc={val_acc} "
                  f"cum_calls={checker.stats.total_calls}")

        if checkpoint_dir is not None:
            _checkpoint_progress(checkpoint_dir, classifier, positives, negatives, history)

    return classifier, positives, negatives, history
