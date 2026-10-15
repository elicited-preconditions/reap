"""Feasibility classifier h(sigma, pi_suffix): encoder-only transformer over
per-(state, action) tokens (Sec 4.2/Appendix C of Cieslar et al.). Smaller
than in the paper, as its appendix states: width 32, 4 heads and 2 blocks
instead of 128, 8 and 4.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from bt_h.classifier.features import OBJECT_FEATURE_SIZE, R_MAXOBJ

TRANSFORMER_WIDTH = 32
N_HEADS = 4
N_BLOCKS = 2
FFN_HIDDEN = 64
FEATURIZER_HIDDEN = 64
POS_EMBED_BASE = 24.0
DROPOUT = 0.1

TOKEN_RAW_SIZE = R_MAXOBJ * OBJECT_FEATURE_SIZE  # flattened per-token object features
EXTRA_FLAG_SIZE = 2 + 1  # action-type one-hot (2) + latest-refined flag (1)


class Featurizer(nn.Module):
    """One small FFN mapping a token's raw object features to
    TRANSFORMER_WIDTH. Sec 4.2 of the paper uses one such network per
    lifted-action-type, further split into a 'refined' and 'unrefined'
    variant each (4 total here: pick x {refined,unrefined}, relocate x
    {refined,unrefined}) -- refined-state inputs (real controller output)
    and unrefined-state inputs (the 'latest known state' stand-in) come
    from different distributions and shouldn't share one network."""

    def __init__(self, in_size: int = TOKEN_RAW_SIZE, hidden: int = FEATURIZER_HIDDEN, out: int = TRANSFORMER_WIDTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sinusoidal_position_embedding(positions: torch.Tensor, dim: int, base: float = POS_EMBED_BASE) -> torch.Tensor:
    """positions: (B, T) float tensor of (possibly randomly offset)
    position indices -- see classifier/dataset.py's collate_examples, which
    draws one random offset per example (length-generalization trick, Zhou
    et al. 2024). Returns (B, T, dim)."""
    device = positions.device
    half = dim // 2
    freqs = torch.exp(-math.log(base) * torch.arange(half, device=device).float() / half)
    angles = positions.unsqueeze(-1) * freqs  # (B, T, half)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class FeasibilityClassifier(nn.Module):
    def __init__(
        self,
        width: int = TRANSFORMER_WIDTH,
        n_heads: int = N_HEADS,
        n_blocks: int = N_BLOCKS,
        ffn_hidden: int = FFN_HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.width = width
        self.featurizers = nn.ModuleDict({
            "pick_refined": Featurizer(out=width),
            "pick_unrefined": Featurizer(out=width),
            "relocate_refined": Featurizer(out=width),
            "relocate_unrefined": Featurizer(out=width),
        })
        self.extra_proj = nn.Linear(EXTRA_FLAG_SIZE, width)
        self.input_norm = nn.LayerNorm(width)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads, dim_feedforward=ffn_hidden,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(nn.Linear(width, 1), nn.Sigmoid())

    @staticmethod
    def _featurizer_key(kind_idx: int, refined: bool) -> str:
        kind = "pick" if kind_idx == 0 else "relocate"
        return f"{kind}_{'refined' if refined else 'unrefined'}"

    def forward(
        self,
        object_features: torch.Tensor,        # (B, T, TOKEN_RAW_SIZE)
        type_flags: torch.Tensor,              # (B, T, 2) one-hot
        latest_flags: torch.Tensor,            # (B, T)
        refined_flags: torch.Tensor,           # (B, T) in {0,1}
        positions: torch.Tensor,               # (B, T) float, pre-offset by caller
        attn_mask: torch.Tensor | None = None,  # (B, T) bool, True = PAD (ignore)
    ) -> torch.Tensor:
        B, T, _ = object_features.shape
        tokens = torch.zeros(B, T, self.width, device=object_features.device)

        # Route each token through the right one of the 4 featurizer nets
        # (split by action kind x refined/unrefined, which vary per
        # (batch, token) position) via boolean masks.
        kind_idx = type_flags.argmax(dim=-1)  # (B, T)
        for k in (0, 1):
            for refined in (True, False):
                mask = (kind_idx == k) & (refined_flags.bool() == refined)
                if not mask.any():
                    continue
                key = self._featurizer_key(k, refined)
                tokens[mask] = self.featurizers[key](object_features[mask])

        extra = torch.cat([type_flags, latest_flags.unsqueeze(-1)], dim=-1)  # (B, T, 3)
        tokens = tokens + self.extra_proj(extra)
        tokens = tokens + sinusoidal_position_embedding(positions, self.width)
        tokens = self.input_norm(tokens)

        encoded = self.transformer(tokens, src_key_padding_mask=attn_mask)
        if attn_mask is not None:
            valid = (~attn_mask).unsqueeze(-1).float()
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoded.mean(dim=1)

        return self.head(pooled).squeeze(-1)  # (B,)
