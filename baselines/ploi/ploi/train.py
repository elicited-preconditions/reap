"""Training loop for PLOI's object-importance GNN (Silver et al., AAAI 2021,
Sections 4 and 5, Appendix A): Adam lr=1e-3, batch size 16, weighted BCE with
false-negative weight 10.

Epochs: the paper trains for a fixed 1000 epochs. With `val_examples`, the
validation loss is computed periodically and the weights with the lowest
validation loss are returned; with `patience`, training stops early. The
reported weights use early stopping on a validation set (the paper's
appendix; python baselines/ploi/run.py train: patience 150). Without
`val_examples`, the defaults are the paper's fixed schedule.

The loss decomposes as a sum over (problem, object) pairs (paper's own
stated decomposition) -- implemented here as one BCEWithLogitsLoss call per
minibatch over the concatenated per-node logits/labels of every graph in
that batch (torch_geometric.data.Batch keeps each graph's nodes contiguous
but not split for a plain per-node loss, which is exactly what's
needed since there's no batch-level pooling in this model).

MASKED TO CYLINDER NODES ONLY: TrainingExample.labels only ever contains
cylinder entries (labels.py never scores/labels cells -- see its module
docstring). Cell nodes still participate in the GNN's forward pass (message
passing needs their spatial/adjacency context), but must not enter the
loss -- treating their absent label as an implicit y=0 would falsely teach
the model that cells are generally irrelevant-and-excludable, when in fact
they're structurally always-kept regardless of score (ploi.py's incremental
loop never even reads a cell's score). `example_to_data` stores a
`train_mask` boolean tensor (True for cylinder nodes) alongside `y`; the
training loop selects only masked nodes before computing the loss.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import reap.xmimir as xmi
from torch_geometric.data import Batch, Data

from ploi import config
from ploi.dataset import TrainingExample
from ploi.features import featurize
from ploi.reduce_problem import CYLINDER_TYPE


def example_to_data(example: TrainingExample) -> Data:
    _domain, problem = xmi.parse(example.domain_path, example.problem_path)
    objects, idx, x, edge_index, edge_attr = featurize(problem)
    y = torch.zeros(len(objects))
    train_mask = torch.zeros(len(objects), dtype=torch.bool)
    for name, i in idx.items():
        if name in example.labels:
            y[i] = float(example.labels[name])
            train_mask[i] = True
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask)


def _assert_masked_nodes_are_cylinders(data: Data, objects) -> None:
    for i, o in enumerate(objects):
        expected = o.type_.name == CYLINDER_TYPE
        assert bool(data.train_mask[i]) == expected, (
            f"train_mask mismatch for {o.name}: expected {expected} (type={o.type_.name})"
        )


@torch.no_grad()
def compute_loss(model: nn.Module, data_list: list[Data], loss_fn: nn.Module) -> float:
    if not data_list:
        return float("nan")
    was_training = model.training
    model.eval()
    batch = Batch.from_data_list(data_list)
    logits = model(batch.x, batch.edge_index, batch.edge_attr)
    mask = batch.train_mask
    loss = loss_fn(logits[mask], batch.y[mask]).item()
    model.train(was_training)
    return loss


@dataclass
class TrainResult:
    epoch_losses: list[float]
    val_losses: dict[int, float] = field(default_factory=dict)  # epoch -> val loss, only when val tracked
    best_val_loss: float | None = None
    best_epoch: int | None = None
    stopped_early_at: int | None = None


def train(
    model: nn.Module,
    examples: list[TrainingExample],
    epochs: int = config.TRAIN_EPOCHS,
    batch_size: int = config.TRAIN_BATCH_SIZE,
    lr: float = config.TRAIN_LR,
    false_negative_weight: float = config.FALSE_NEGATIVE_WEIGHT,
    seed: int = 0,
    log_every: int = 50,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 100,
    val_examples: list[TrainingExample] | None = None,
    val_every: int = 25,
    patience: int | None = None,
) -> TrainResult:
    """Trains `model` in place. Data objects are built once up front
    (featurization needs no gradient and is cheap relative to a training
    epoch).

    val_examples: if given, validation loss is computed every `val_every`
        epochs; the model's weights at the end of training are the BEST-
        validation-loss checkpoint seen (not necessarily the final epoch's).
    patience: if given (requires val_examples), stop early once `patience`
        epochs have passed with no new best validation loss.
    """
    rng = random.Random(seed)
    data_list = [example_to_data(ex) for ex in examples]
    val_data_list = [example_to_data(ex) for ex in val_examples] if val_examples else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(false_negative_weight)))

    result = TrainResult(epoch_losses=[])
    best_state = None
    epochs_since_improvement = 0
    final_epoch = epochs - 1

    for epoch in range(epochs):
        rng.shuffle(data_list)
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(data_list), batch_size):
            batch_items = data_list[start:start + batch_size]
            batch = Batch.from_data_list(batch_items)

            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            mask = batch.train_mask
            loss = loss_fn(logits[mask], batch.y[mask])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / max(1, n_batches)
        result.epoch_losses.append(mean_loss)
        log_now = log_every and epoch % log_every == 0
        if log_now:
            print(f"epoch {epoch}: mean_loss={mean_loss:.4f}")

        if val_data_list and (epoch % val_every == 0 or epoch == final_epoch):
            val_loss = compute_loss(model, val_data_list, loss_fn)
            result.val_losses[epoch] = val_loss
            improved = val_loss < result.best_val_loss if result.best_val_loss is not None else True
            if improved:
                result.best_val_loss = val_loss
                result.best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += val_every
            if log_now:
                print(f"  val_loss={val_loss:.4f} (best={result.best_val_loss:.4f} @ epoch {result.best_epoch})")
            if patience is not None and epochs_since_improvement >= patience:
                print(f"early stopping at epoch {epoch}: no val improvement for {epochs_since_improvement} epochs")
                result.stopped_early_at = epoch
                final_epoch = epoch
                break

        if checkpoint_path is not None and checkpoint_every and epoch % checkpoint_every == 0:
            torch.save(model.state_dict(), checkpoint_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path is not None:
        torch.save(model.state_dict(), checkpoint_path)
    return result
