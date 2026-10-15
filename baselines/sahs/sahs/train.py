"""Training of the SAHS networks Q_pick/Q_put with the large-margin ranking loss of the IJRR 2022 version (Section
5.1; the CoRL 2019 version adds an MSE term):

    L_LM(theta) = sum_{(s,G,delta) in D_a} max(0, 1 - M(alpha(s,G),delta;theta))
    M(alpha(s,G),delta;theta) = F(alpha(s,G),delta;theta)
                                  - max_{delta' in Delta, delta' != delta} F(alpha(s,G),delta';theta)

The IJRR version has no regression term ("to obtain accurate values of all the actions of a state, we would need
the target values for many of them. However, in our training data, we only have values of actions that led to the
goal"): the loss only needs the chosen action to outrank the others.

Full batch: the gradients of all examples accumulate (one .backward() per example) before one optimizer.step() per
epoch.
"""
from __future__ import annotations

import time

import torch

from sahs.dataset import TrainingExample, predicate_set_from_dict
from sahs.features import (
    DEFAULT_EDGE_FEATURES, build_edge_tensors, delta_index, entities_for_pick, entities_for_put,
)
from sahs.gnn import TripleRelationalQNet


def _prepare_example(
    ex: TrainingExample, edge_features: str = DEFAULT_EDGE_FEATURES, dedup_filter: bool = True,
) -> dict:
    ps = predicate_set_from_dict(ex.predicates)
    if ex.operator == "pick":
        obj_e, reg_e = entities_for_pick(ex.cylinders)
    else:
        # The regions of a put example are the cells of its applicable deltas (the empty cells of the state, as
        # in sahs.py and generate_training_data.py); they include the chosen delta's cell.
        regions = sorted({d["region"] for d in ex.applicable_deltas if d["region"] is not None})
        obj_e, reg_e = entities_for_put(ex.cylinders, regions)
    x_obj, edge = build_edge_tensors(obj_e, reg_e, ps, ex.target, feature_set=edge_features)
    delta_idx = delta_index(obj_e, reg_e, ex.delta)
    applicable_idx = [delta_index(obj_e, reg_e, d) for d in ex.applicable_deltas]

    # Candidates whose whole feature column edge[:, row, c, :] equals the chosen delta's ("duplicates", e.g. two
    # distant empty cells for put). Round 0's aggregation is region-independent, so the network gives duplicates the
    # chosen delta's value and the margin term of such an example cannot drop below 1. dedup_filter=True (the default,
    # the released weights) excludes the duplicates from the ranking loss (`distinguishable_idx`). dedup_filter=False
    # ranks the chosen delta against every other delta' in Delta, as the paper's loss is written; the put network then
    # does not learn.
    row, col = delta_idx
    chosen_slice = edge[:, row, col, :]
    other_idx = [(r, c) for (r, c) in applicable_idx if (r, c) != delta_idx]
    duplicate_idx = [
        (r, c) for (r, c) in other_idx if (edge[:, r, c, :] - chosen_slice).norm().item() <= 1e-6
    ]
    distinguishable_idx = [rc for rc in other_idx if rc not in duplicate_idx]

    return {
        "x_obj": x_obj,
        "edge": edge,
        "num_regions": len(reg_e),
        "delta_idx": delta_idx,
        "applicable_idx": applicable_idx,
        "distinguishable_idx": distinguishable_idx,
        "duplicate_idx": duplicate_idx,
        "rank_idx": distinguishable_idx if dedup_filter else other_idx,
    }


def train_operator(
    examples: list[TrainingExample],
    epochs: int = 300,
    lr: float = 1e-3,
    edge_features: str = DEFAULT_EDGE_FEATURES,
    dedup_filter: bool = True,
) -> tuple[TripleRelationalQNet, list[dict]]:
    net = TripleRelationalQNet(edge_features=edge_features)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    t_prep = time.perf_counter()
    prepared = [_prepare_example(ex, edge_features, dedup_filter) for ex in examples]
    n_dup = sum(1 for p in prepared if p["duplicate_idx"])
    print(f"  prepared {len(prepared)} examples ({edge_features} edges, dedup_filter={dedup_filter}) in "
          f"{time.perf_counter() - t_prep:.1f}s; {n_dup} have a feature-duplicate of the chosen delta", flush=True)
    n = len(prepared)

    history: list[dict] = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_margin_loss = 0.0

        for p in prepared:
            f_table = net(p["x_obj"], p["edge"], num_regions=p["num_regions"])
            row, col = p["delta_idx"]
            f_delta = f_table[row, col]

            other_vals = [f_table[r, c] for (r, c) in p["rank_idx"]]
            if other_vals:
                f_other_max = torch.stack(other_vals).max()
                margin_term = torch.relu(1.0 - (f_delta - f_other_max))
            else:
                # No alternative delta to rank against at this state -- e.g.
                # exactly one pick candidate, or (with dedup_filter) every
                # other candidate is an exact feature-duplicate of the chosen
                # delta (see _prepare_example).
                # The margin is trivially satisfied, zero loss, but must
                # stay a differentiable function of f_delta (not a detached
                # constant) or .backward() has no grad_fn to run through.
                margin_term = 0.0 * f_delta

            margin_term.backward()
            total_margin_loss += float(margin_term.item())

        optimizer.step()
        history.append({"epoch": epoch, "margin_loss": total_margin_loss / n})

    return net, history
