"""Single-example, runtime classifier queries -- used by bt_h.py during
search (as opposed to classifier/dataset.py's batched training-time path).
Two callables sharing the same (sigma, pi_full) -> float-in-[0,1] interface:
ConstantClassifier (h === 1, Algorithm 3's Line 2 initial classifier before
any data exists) and TrainedClassifier (wraps a real FeasibilityClassifier
checkpoint).
"""
from __future__ import annotations

import random

import torch

from bt_h.classifier.features import featurize_example
from bt_h.classifier.model import FeasibilityClassifier


class ConstantClassifier:
    """h === 1: Algorithm 3, Line 2 ('h := Constant(1)') -- every state
    looks refinable until real data exists to learn otherwise."""

    def __call__(self, sigma: list, pi_full: list) -> float:
        return 1.0


class TrainedClassifier:
    def __init__(self, model: FeasibilityClassifier, rng: random.Random | None = None):
        self.model = model
        self.model.eval()
        self.rng = rng or random.Random()

    @torch.no_grad()
    def __call__(self, sigma: list, pi_full: list) -> float:
        feats = featurize_example(sigma, pi_full, self.rng)
        n = feats["n_tokens"]
        obj = torch.tensor([feats["object_features"]], dtype=torch.float32)
        type_flags = torch.tensor([feats["type_flags"]], dtype=torch.float32)
        latest_flags = torch.tensor([feats["latest_flags"]], dtype=torch.float32)
        refined_flags = torch.tensor([feats["refined_flags"]], dtype=torch.float32)
        positions = torch.tensor([list(range(n))], dtype=torch.float32)
        out = self.model(obj, type_flags, latest_flags, refined_flags, positions)
        return float(out.item())
