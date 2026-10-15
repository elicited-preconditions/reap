"""Training loop for the feasibility classifier (Sec 4.2/Appendix C of the
paper): Adam and binary cross-entropy, early-stopped on a held-out
validation loss (the data collection loop trains for at most
config.DATA_COLLECTION_TRAIN_MAX_STEPS steps per round, batches of 32; the
paper uses 5,000 iterations with batches of 4,000).
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bt_h.classifier.dataset import FeasibilityDataset, collate_examples
from bt_h.classifier.model import FeasibilityClassifier

TRANSFORMER_LR = 3e-4
OTHER_LR = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_STEPS = 1000
DEFAULT_PATIENCE = 200  # early-stop if val loss hasn't improved in this many steps
EVAL_EVERY = 20


def make_optimizer(model: FeasibilityClassifier) -> torch.optim.Optimizer:
    transformer_params = list(model.transformer.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("transformer.")]
    return torch.optim.Adam([
        {"params": transformer_params, "lr": TRANSFORMER_LR},
        {"params": other_params, "lr": OTHER_LR},
    ])


def _make_loader(dataset: FeasibilityDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    collate_rng = random.Random(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda batch: collate_examples(batch, collate_rng),
    )


@torch.no_grad()
def evaluate(model: FeasibilityClassifier, loader: DataLoader, loss_fn) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        out = model(
            batch["object_features"], batch["type_flags"], batch["latest_flags"],
            batch["refined_flags"], batch["positions"], attn_mask=batch["attn_mask"],
        )
        loss = loss_fn(out, batch["label"])
        total_loss += loss.item() * len(batch["label"])
        total_correct += ((out > 0.5).float() == batch["label"]).sum().item()
        total_n += len(batch["label"])
    model.train()
    return total_loss / total_n, total_correct / total_n


def train_classifier(
    train_dataset: FeasibilityDataset,
    val_dataset: FeasibilityDataset,
    model: FeasibilityClassifier | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_steps: int = DEFAULT_MAX_STEPS,
    patience: int = DEFAULT_PATIENCE,
    eval_every: int = EVAL_EVERY,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[FeasibilityClassifier, dict]:
    """Adam + BCE, early-stopped on held-out val loss. Returns the
    best-val-loss model (state reloaded at the end) plus a history dict
    ('step'/'train_loss'/'val_loss'/'val_acc' lists) for diagnosis/plots."""
    if model is None:
        model = FeasibilityClassifier()
    model.train()

    optimizer = make_optimizer(model)
    loss_fn = nn.BCELoss()

    train_loader = _make_loader(train_dataset, batch_size, shuffle=True, seed=seed)
    val_loader = _make_loader(val_dataset, batch_size=max(1, len(val_dataset)), shuffle=False, seed=seed + 1)

    history: dict = {"step": [], "train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    steps_since_improvement = 0
    step = 0

    def _cycle(loader):
        while True:
            yield from loader

    train_iter = _cycle(train_loader)

    while step < max_steps:
        batch = next(train_iter)
        out = model(
            batch["object_features"], batch["type_flags"], batch["latest_flags"],
            batch["refined_flags"], batch["positions"], attn_mask=batch["attn_mask"],
        )
        loss = loss_fn(out, batch["label"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        if step % eval_every == 0 or step == max_steps:
            val_loss, val_acc = evaluate(model, val_loader, loss_fn)
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if verbose:
                print(f"step {step}/{max_steps}: train_loss={loss.item():.4f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                steps_since_improvement = 0
            else:
                steps_since_improvement += eval_every
                if steps_since_improvement >= patience:
                    if verbose:
                        print(f"early stopping at step {step} (no val_loss improvement in {patience} steps)")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def constant_classifier_predict(*_args, **_kwargs) -> float:
    """h === 1: the initial classifier before any real data exists (Sec 4.3
    of the paper: 'the feasibility classifier always outputs a confidence
    of 1' at Algorithm 3's first data-collection iteration). Used by
    bt_h.py / data_collection.py until a trained checkpoint is available."""
    return 1.0
