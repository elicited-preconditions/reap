"""torch Dataset over collected (sigma_prefix, pi_full, label) examples --
positive: prefixes of a successful demonstration's real state sequence;
negative: prefixes from failed/stuck backtracking branches (Algorithm 3's
full bootstrap loop harvests both from
bt_h_capture_tree's search trees). featurize_example() (features.py) is
applied lazily in __getitem__ so a fresh random identity-map is drawn each
epoch (Sec 4.2 of the paper: object identities are randomized PER TRAINING
EXAMPLE, not fixed once at collection time).
"""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from bt_h.classifier.features import featurize_example


class FeasibilityExample:
    __slots__ = ("sigma", "pi_full", "label")

    def __init__(self, sigma: list, pi_full: list, label: int):
        self.sigma = sigma
        self.pi_full = pi_full
        self.label = label


class FeasibilityDataset(Dataset):
    def __init__(self, examples: list[FeasibilityExample], seed: int = 0):
        self.examples = examples
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        feats = featurize_example(ex.sigma, ex.pi_full, self._rng)
        return {
            "object_features": torch.tensor(feats["object_features"], dtype=torch.float32),
            "type_flags": torch.tensor(feats["type_flags"], dtype=torch.float32),
            "latest_flags": torch.tensor(feats["latest_flags"], dtype=torch.float32),
            "refined_flags": torch.tensor(feats["refined_flags"], dtype=torch.float32),
            "label": torch.tensor(float(ex.label)),
        }


def collate_examples(batch: list[dict], rng: random.Random, pos_embed_max_offset: float = 8.0) -> dict:
    """Pads variable-length token sequences to the batch max (plan lengths
    range 3..13 across K=1..6), builds a padding mask, and draws ONE random
    positional offset per example (length-generalization trick, Zhou et al.
    2024)."""
    max_t = max(b["object_features"].shape[0] for b in batch)
    B = len(batch)
    feat_size = batch[0]["object_features"].shape[-1]

    object_features = torch.zeros(B, max_t, feat_size)
    type_flags = torch.zeros(B, max_t, 2)
    latest_flags = torch.zeros(B, max_t)
    refined_flags = torch.zeros(B, max_t)
    positions = torch.zeros(B, max_t)
    attn_mask = torch.ones(B, max_t, dtype=torch.bool)  # True = pad
    labels = torch.zeros(B)

    for b_idx, b in enumerate(batch):
        t = b["object_features"].shape[0]
        object_features[b_idx, :t] = b["object_features"]
        type_flags[b_idx, :t] = b["type_flags"]
        latest_flags[b_idx, :t] = b["latest_flags"]
        refined_flags[b_idx, :t] = b["refined_flags"]
        offset = rng.uniform(0, pos_embed_max_offset)
        positions[b_idx, :t] = torch.arange(t, dtype=torch.float32) + offset
        attn_mask[b_idx, :t] = False
        labels[b_idx] = b["label"]

    return {
        "object_features": object_features,
        "type_flags": type_flags,
        "latest_flags": latest_flags,
        "refined_flags": refined_flags,
        "positions": positions,
        "attn_mask": attn_mask,
        "label": labels,
    }
