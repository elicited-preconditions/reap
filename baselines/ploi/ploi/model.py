# Ported from github.com/tomsilver/ploi (gnn/gnn.py: EncodeProcessDecode, GraphModel, MLP). MIT License,
# Copyright (c) 2020 Tom Silver and Rohan Chitnis; the notice is in LICENSES/ploi-MIT.txt.
"""PLOI's object-importance scorer: Encode-Process-Decode Graph Network
(Battaglia et al. 2018), K=3 message-passing iterations (Silver et al., AAAI
2021, Section 5 and Appendix A).

The architecture follows the authors' code (github.com/tomsilver/ploi,
gnn/gnn.py: EncodeProcessDecode, GraphModel, MLP; MIT license, Copyright (c)
2020 Tom Silver and Rohan Chitnis, LICENSES/ploi-MIT.txt). The paper's text
does not say whether the same block is reused across rounds; the code uses
a single shared "core" (edge_model + node_model), applied K=3 times, with
the encoded input concatenated in at every step. Their code works on
PDDLGym graph dicts; this port works on torch_geometric tensors:

  - Encoder: separate node/edge MLPs, raw features -> hidden_dim=16.
  - Core (SHARED across all K steps): edge_model then node_model, each an
    MLP with one hidden layer of width 16. At every step, the CURRENT
    node/edge state is concatenated with the ORIGINAL ENCODED state
    (`latent0`) before the edge/node update -- not just once at the start.
  - Decoder: a single Linear(16, 1), no hidden layer, no LayerNorm (their
    MLP() helper with a 1-element `layers` list gives one Linear; the
    LayerNorm is added only if `len(layers) > 1`). Applied after the last
    step only: their train_model and get_single_model_prediction use only
    `outputs[-1]`.
  - MLP structure (their own `MLP()` helper, `layers=[16,16]`): the ONLY
    LayerNorm sits AFTER the whole two-linear stack (normalizing the
    MODULE'S OUTPUT), not between the hidden layer and the final Linear.

Output: a single score per node (sigmoid of the decoder's output) -- the
output graph's own edges are discarded, as in the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ploi import config
from ploi import features
from ploi.features import EDGE_DIM


class _MLP(nn.Module):
    """As the authors' `MLP(layers, input_dim)` helper:
    `Linear(input_dim, layers[0])`, then for each subsequent entry
    `ReLU()` + `Linear(layers[i], layers[i+1])`, then -- only if `len(layers)
    > 1` -- a final `LayerNorm(layers[-1])` normalizing the whole stack's
    OUTPUT. `layers=[hidden_dim]` (a 1-element list, used for the decoder)
    therefore yields a single bare Linear with no LayerNorm."""

    def __init__(self, layers: list[int], input_dim: int):
        super().__init__()
        mlp_layers = [nn.Linear(input_dim, layers[0])]
        for i in range(len(layers) - 1):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(layers[i], layers[i + 1]))
        if len(layers) > 1:
            mlp_layers.append(nn.LayerNorm(layers[-1]))
        self.net = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObjectImportanceGNN(nn.Module):
    def __init__(
        self,
        node_dim: int | None = None,
        edge_dim: int = EDGE_DIM,
        hidden_dim: int = config.GNN_HIDDEN_DIM,
        num_steps: int = config.GNN_NUM_MESSAGE_PASSING_ROUNDS,
    ):
        super().__init__()
        if node_dim is None:  # the active feature set (features.set_feature_set), read when the model is built
            node_dim = features.NODE_DIM
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        # Encoder: raw features -> hidden_dim, one hidden layer each.
        self.node_encoder = _MLP([hidden_dim, hidden_dim], node_dim)
        self.edge_encoder = _MLP([hidden_dim, hidden_dim], edge_dim)

        # Core (SHARED across all num_steps rounds -- not reinstantiated per
        # round). Input dims account for the concat([latent0, current])
        # skip-connection applied before every use: node/edge tensors are
        # 2*hidden_dim wide at the point they're consumed here.
        self.edge_model = _MLP([hidden_dim, hidden_dim], 2 * hidden_dim + 2 * hidden_dim + 2 * hidden_dim)
        self.node_model = _MLP([hidden_dim, hidden_dim], 2 * hidden_dim + hidden_dim)

        # Decoder: single Linear, no hidden layer, no LayerNorm (see class docstring).
        self.node_decoder = _MLP([1], hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Returns per-node LOGITS, shape (N,)."""
        n = x.size(0)
        latent0_x = self.node_encoder(x)          # (N, h)
        latent0_e = self.edge_encoder(edge_attr)   # (E, h)
        cur_x, cur_e = latent0_x, latent0_e

        for _ in range(self.num_steps):
            # Residual skip-connection: concat the running state with the
            # ORIGINAL encoded state at every step, not just at the start.
            in_x = torch.cat([latent0_x, cur_x], dim=-1)  # (N, 2h)

            if edge_index.numel() == 0:
                new_e = latent0_e.new_zeros(0, self.hidden_dim)
                agg = in_x.new_zeros(n, self.hidden_dim)
            else:
                in_e = torch.cat([latent0_e, cur_e], dim=-1)  # (E, 2h)
                src, dst = edge_index[0], edge_index[1]
                # Marshal receiver, sender, edge (their marshalling_func).
                edge_in = torch.cat([in_x[dst], in_x[src], in_e], dim=-1)  # (E, 6h)
                new_e = self.edge_model(edge_in)  # (E, h)

                agg = in_x.new_zeros(n, self.hidden_dim)
                agg.index_add_(0, dst, new_e)  # (N, h) -- their aggregation_func

            node_in = torch.cat([in_x, agg], dim=-1)  # (N, 3h)
            new_x = self.node_model(node_in)  # (N, h)

            cur_x, cur_e = new_x, new_e

        return self.node_decoder(cur_x).squeeze(-1)

    @torch.no_grad()
    def score(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x, edge_index, edge_attr))


def make_gnn_scorer(model: "ObjectImportanceGNN"):
    """Wraps a trained model into the Callable[[XProblem], dict[str, float]]
    signature ploi.run_ploi's `scorer` parameter expects -- a single forward
    pass per problem, per the paper's own stated efficiency property."""
    from ploi.features import featurize

    def scorer(problem) -> dict[str, float]:
        objects, idx, x, edge_index, edge_attr = featurize(problem)
        scores = model.score(x, edge_index, edge_attr)
        return {o.name: scores[i].item() for i, o in enumerate(objects)}

    return scorer
