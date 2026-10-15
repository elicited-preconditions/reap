# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

import reap.xmimir as xmimir
from .logging_setup import setup_logger

setup_logger(__name__)

import torch
import torch_geometric as pyg

from . import encoding, models, rl

# Register custom classes for serialization with torch > 2.6.
# This is necessary for loading models that use custom classes (from this package or pyg). Before 2.6 this prompted
# a warning, from 2.6 onwards it raises an error.
# https://pytorch.org/docs/2.6/notes/serialization.html#torch-load-with-weights-only-true
# https://pytorch.org/docs/2.6/generated/torch.load.html#torch-load
# If more classes from pyg or this package are saved with torch.save, they need to be added here to not trigger an
# exception.
torch.serialization.add_safe_globals(
    [
        pyg.data.storage.BaseStorage,
        pyg.data.storage.NodeStorage,
        pyg.data.storage.EdgeStorage,
        pyg.data.Data,
        pyg.data.HeteroData,
        pyg.data.Batch,
        xmimir.ActionHistoryDataPack,
        xmimir.ActionDataPack,
    ]
)

__all__ = ["encoding", "models", "rl"]
