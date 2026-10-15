"""The SAHS relational GNN (one per operator) that gives a rank value to every (object, region) pair, in plain
PyTorch. It follows the IJRR 2022 version (Kim, Shimanuki, Kaelbling and Lozano-Perez, Section 5.2 and Appendix
A.3), whose appendix gives the formulas:
  u_oi^(0) = f(Node(oi); theta1),  v_oj^(0) = f(Node(oj); theta1)   [sic in
    the printed formula -- both use theta1; "we found we achieve better
    performance if we use a separate embedding function for the sender and
    receiver" in the surrounding prose implies u should use theta1 and v a
    distinct theta2, and this is what round 1's formula literally shows
    (u^(1)=f(.;theta1), v^(1)=f(.;theta2)) -- read as a transcription
    inconsistency in the round-0 formula specifically, resolved here as
    theta1=u/sender, theta2=v/receiver, consistently at both rounds.]
  c_{oi,oj,rk} = f(Edge(oi,oj,rk); theta3)
  m_{oi,oj,rk}^(0) = f(u_oi^(0), v_oj^(0), c_{oi,oj,rk}; theta4)
  m_oj^(0) = mean over (oi, rk) of m_{oi,oj,rk}^(0)   -- ROUND 0 COLLAPSES
    THE REGION DIMENSION: "although we do not have an edge between
    components [regions], we aggregate messages across components, because
    the occlusion information in both regions are necessary to predict the
    rank value of moving an object to a region." Region-INDEPENDENT after
    this step.
  u_oj^(1) = f(m_oj^(0); theta1),  v_oj^(1) = f(m_oj^(0); theta2)
  m_{oi,oj,rk}^(1) = f(u_oi^(1), v_oj^(1), c_{oi,oj,rk}; theta4)   -- same
    theta4, same c (not recomputed), new u/v
  m_{oj,rk}^(1) = mean over oi of m_{oi,oj,rk}^(1)   -- ROUND 1 AGGREGATES
    OBJECTS ONLY, preserving the region dimension this time.
  rank(oj, rk) = f(m_{oj,rk}^(1); theta5)

Two readings of the printed formulas:
1. theta1/theta2 (u/v) are NOT literally the same weight matrices at round 0
   (input: raw UNARY_DIM Node features) and round 1 (input: m_dim-sized
   aggregated messages) here -- those inputs have different dimensionality,
   and nothing in the text describes a projection step reconciling them.
   Kept as separate round-0/round-1 networks (theta1_r0/theta1_r1,
   theta2_r0/theta2_r1), each still playing the "u" or "v" role consistently.
   theta3, theta4, theta5 ARE literally shared where the formulas call for
   it (theta4 reused verbatim at both rounds -- its inputs are m_dim-sized
   both times, so no such conflict there).
2. The paper's aggregation divisors as printed ("1/(|O^(M)|+|R|)" for round
   0, "1/|O^(M)|" for round 1) are read as the STANDARD mean over the actual
   number of terms summed (round 0: (N_obj-1)*R terms, excluding oi=oj;
   round 1: (N_obj-1) terms) rather than literally divided by the printed
   sums-of-cardinalities, since the surrounding text calls this "the
   standard averaging-aggregation function" and a sum-of-cardinalities
   divisor is not a standard mean of that many terms.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sahs.features import DEFAULT_EDGE_FEATURES, UNARY_DIM, edge_dim, feature_set_for_edge_dim


class MLP(nn.Module):
    """2 FC layers, 32 hidden units, ReLU except optionally the final layer
    -- matches the paper's stated architecture for every f(.;theta_n)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 32, relu_output: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.relu_output = relu_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        if self.relu_output:
            out = torch.relu(out)
        return out


class TripleRelationalQNet(nn.Module):
    def __init__(self, m_dim: int = 32, hidden: int = 32, edge_features: str = DEFAULT_EDGE_FEATURES):
        super().__init__()
        self.edge_features = edge_features  # features.py: "ijrr22" (edge width 22)
        self.m_dim = m_dim
        self.theta1_r0 = MLP(UNARY_DIM, m_dim, hidden)  # u, round 0: Node(oi) -> u
        self.theta2_r0 = MLP(UNARY_DIM, m_dim, hidden)  # v, round 0: Node(oj) -> v
        self.theta1_r1 = MLP(m_dim, m_dim, hidden)      # u, round 1: m_oi^(0) -> u
        self.theta2_r1 = MLP(m_dim, m_dim, hidden)      # v, round 1: m_oj^(0) -> v
        self.theta3 = MLP(edge_dim(edge_features), m_dim, hidden)      # edge embedding, shared both rounds
        self.theta4 = MLP(3 * m_dim, m_dim, hidden, relu_output=False)  # message, shared both rounds
        self.theta5 = MLP(m_dim, 1, hidden, relu_output=False)          # final readout

    def forward(self, x_obj: torch.Tensor, edge: torch.Tensor, num_regions: int) -> torch.Tensor:
        """x_obj: (N,UNARY_DIM), edge: (N,N,R,edge_dim(self.edge_features)). Returns rank/Q
        values of shape (N,R) -- one per (object, region) pair."""
        n = x_obj.shape[0]
        r = num_regions
        diag = torch.eye(n, dtype=torch.bool).unsqueeze(-1).unsqueeze(-1)  # (N,N,1,1)

        u0 = self.theta1_r0(x_obj)  # (N, m_dim)
        v0 = self.theta2_r0(x_obj)  # (N, m_dim)
        c = self.theta3(edge)       # (N,N,R,m_dim) -- reused at both rounds

        u0_i = u0.unsqueeze(1).unsqueeze(2).expand(n, n, r, self.m_dim)  # u0[i]
        v0_j = v0.unsqueeze(0).unsqueeze(2).expand(n, n, r, self.m_dim)  # v0[j]
        m0 = self.theta4(torch.cat([u0_i, v0_j, c], dim=-1))  # (N,N,R,m_dim)
        m0 = m0.masked_fill(diag, 0.0)

        # Round 0: aggregate over BOTH oi and rk -- collapses the region
        # dimension, per the paper's explicit cross-region pooling.
        denom0 = max((n - 1) * r, 1)
        m_obj0 = m0.sum(dim=(0, 2)) / denom0  # (N, m_dim), region-independent

        u1 = self.theta1_r1(m_obj0)  # (N, m_dim)
        v1 = self.theta2_r1(m_obj0)  # (N, m_dim)

        u1_i = u1.unsqueeze(1).unsqueeze(2).expand(n, n, r, self.m_dim)
        v1_j = v1.unsqueeze(0).unsqueeze(2).expand(n, n, r, self.m_dim)
        m1 = self.theta4(torch.cat([u1_i, v1_j, c], dim=-1))  # same theta4, same c
        m1 = m1.masked_fill(diag, 0.0)

        # Round 1: aggregate over oi ONLY -- preserves the region dimension.
        denom1 = max(n - 1, 1)
        m_obj_region1 = m1.sum(dim=0) / denom1  # (N, R, m_dim)

        return self.theta5(m_obj_region1).squeeze(-1)  # (N, R)


def edge_features_of_state_dict(state_dict: dict) -> str:
    """The edge feature set of a checkpoint, from its theta3 input width (22 = "ijrr22")."""
    return feature_set_for_edge_dim(state_dict["theta3.fc1.weight"].shape[1])


def load_qnet(path) -> TripleRelationalQNet:
    state_dict = torch.load(path)
    net = TripleRelationalQNet(edge_features=edge_features_of_state_dict(state_dict))
    net.load_state_dict(state_dict)
    net.eval()
    return net
