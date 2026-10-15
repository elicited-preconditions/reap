# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from __future__ import annotations

from typing import Callable, List, Literal, Union

import torch
import torch_geometric as pyg
from tensordict import TensorDict, TensorDictBase
from torch import Tensor

from reap.logging_setup import get_logger
from reap.utils.reshape import unsqueeze_right, unsqueeze_right_like


class SetEmbedding:
    dense_embedding: Tensor
    padding_mask: Tensor

    default_padding_value: float = 0.0

    def __init__(
        self,
        dense_embedding: Tensor,
        padding_mask: Tensor,
        padding_value: float = default_padding_value,
    ) -> None:
        """
        Wrap dense per-item symbol embeddings together with a validity mask.

        The name stems from packing a set of N tensor-sets S_i with |S_i| many
        symbol embeddings in them together. We compute the maximum tensor count
        of all sets O = max_i |S_i|, and pad any tensor-set up to this number of
        elements, if necessary. To ensure we know which embeddings were valid,
        we mark it down in padding mask, where the TRUE entries are valid embeddings.

        Notes:
        - "Symbols" typically refer to objects in our planning domains, but this
          abstraction is generic and may be used for other symbol sets.
        - The mask indicates which positions are real and which are padding.
          A truthy entry means a real embedding at that position.

        Args:
            dense_embedding:
                Tensor of shape ``[N, O, *F]`` or ``[N, *F]``. If 2D, it is promoted to
                ``[N, 1, *F]`` internally.
            padding_mask:
                Boolean tensor of shape ``[N, O]`` or broadcastable to that shape. If it has
                fewer than 2 dimensions, it is unsqueezed on the right to match
                ``dense_embedding`` in the first two dims.

        Raises:
            ValueError:
                If ``dense_embedding`` has fewer than 2 dimensions or if the first two
                dimensions of ``dense_embedding`` and ``padding_mask`` do not match after
                broadcasting.
        """
        padding_dim = padding_mask.ndim
        if (dim_diff := 2 - padding_dim) > 0:
            padding_mask = unsqueeze_right(padding_mask, dim_diff)
        embedding_dim = dense_embedding.ndim
        if embedding_dim < 2:
            raise ValueError(
                f"Embeddings tensor must have at least 2 dimensions. Got {dense_embedding.shape}."
            )
        elif embedding_dim == 2:
            dense_embedding = dense_embedding.unsqueeze(1)
        self.dense_embedding = dense_embedding
        self.padding_mask = padding_mask
        self.padding_value = padding_value
        self._assert_dim_match(0, 1)

    def _assert_dim_match(self, *dims: int):
        dense_embedding = self.dense_embedding
        padding_mask = self.padding_mask
        for dim in dims:
            if dense_embedding.size(dim) != padding_mask.size(dim):
                raise ValueError(
                    f"Sizes of 'dense_embedding' and 'padding_mask' at dimension {dim} do not match: "
                    f"{dense_embedding.size(dim)} != {padding_mask.size(dim)}"
                )

    def __eq__(self, other: object):
        if not isinstance(other, SetEmbedding):
            return NotImplemented
        return torch.equal(
            self.dense_embedding[self.padding_mask],
            other.dense_embedding[other.padding_mask],
        )

    def __repr__(self):
        return f"SetEmbedding(dense_embedding={self.dense_embedding}, padding_mask={self.padding_mask})"

    def to_tensordict(self) -> TensorDict:
        return TensorDict(
            {
                "dense_embedding": self.dense_embedding,
                "padding_mask": self.padding_mask,
            },
            batch_size=(self.dense_embedding.size(0),),
        )

    def as_flat(self) -> tuple[Tensor, Tensor]:
        """
        Return a flat view suitable for torch_geometric-style global pooling.

        Returns:
            x_flat: Tensor[num_valid, *F] with only valid set items
            batch:  Tensor[num_valid] mapping each row in x_flat to its graph index
        """
        x_flat = self.dense_embedding[self.padding_mask]
        batch = mask_to_batch_indices(self.padding_mask)
        return x_flat, batch

    def tensor_split(self, split_indices_tensor: torch.Tensor) -> List[SetEmbedding]:
        if not split_indices_tensor.is_cpu:
            get_logger(__name__).info(
                "Received split tensor that was on the GPU."
                " Torch requires it to be on the CPU."
                " Performing implicit device move."
            )
            split_indices_tensor = split_indices_tensor.cpu()
        split_dense = self.dense_embedding.tensor_split(split_indices_tensor)
        split_mask = self.padding_mask.tensor_split(split_indices_tensor)
        return [
            SetEmbedding(dense, mask) for dense, mask in zip(split_dense, split_mask)
        ]

    def to_masked_tensor(self):
        """
        Convert the set embedding to a masked tensor.
        """
        unsqueezed_mask = unsqueeze_right_like(self.padding_mask, self.dense_embedding)
        return torch.masked.masked_tensor(
            self.dense_embedding,
            unsqueezed_mask.expand_as(self.dense_embedding),
            requires_grad=self.dense_embedding.requires_grad,
        )

    def to(self, device: torch.device) -> SetEmbedding:
        """
        Move the set embedding to the specified device.
        """
        self.dense_embedding = self.dense_embedding.to(device)
        self.padding_mask = self.padding_mask.to(device)
        return self

    def cpu(self):
        """
        Move the set embedding to the CPU.
        """
        self.dense_embedding = self.dense_embedding.cpu()
        self.padding_mask = self.padding_mask.cpu()
        return self

    def clone(self) -> SetEmbedding:
        """
        Clone the set embedding.
        """
        # shortcut the __init__ logic which only verifies the tensor shapes needlessly in this case
        obj = SetEmbedding.__new__(SetEmbedding)
        obj.dense_embedding = self.dense_embedding.clone()
        obj.padding_mask = self.padding_mask.clone()
        return obj

    def detach(self) -> SetEmbedding:
        """
        Detach the set embedding from the computation graph.
        """
        return SetEmbedding(
            self.dense_embedding.detach(),
            self.padding_mask.detach(),
        )

    def allclose(
        self,
        other: TensorDict | SetEmbedding,
    ) -> bool:
        if isinstance(other, TensorDict):
            other = SetEmbedding.from_tensordict(other)
        return (
            torch.allclose(self.dense_embedding, other.dense_embedding)
            and (self.padding_mask == other.padding_mask).all()
        )

    @staticmethod
    def from_tensordict(tensordict: TensorDictBase) -> SetEmbedding:
        return SetEmbedding(tensordict["dense_embedding"], tensordict["padding_mask"])

    @staticmethod
    def from_masked_tensor(
        masked_tensor: torch.masked.MaskedTensor,
        padding_value: float = default_padding_value,
    ) -> SetEmbedding:
        padding_mask = masked_tensor.get_mask()
        return SetEmbedding(
            dense_embedding=masked_tensor.masked_fill(~padding_mask, padding_value),
            padding_mask=padding_mask,
            padding_value=padding_value,
        )

    @staticmethod
    def from_sparse(
        sparse_embedding: Tensor | dict[str, Tensor],
        batch: torch.Tensor | dict[str, Tensor],
    ) -> SetEmbedding:
        if not type(sparse_embedding) == type(batch):
            raise ValueError(
                f"'sparse_embedding' and 'batch' must be of the same type. Got {type(sparse_embedding)} and {type(batch)}."
            )
        if isinstance(sparse_embedding, dict):
            assert len(sparse_embedding) == 1 == len(batch), (
                "Heterogeneous embeddings are not supported."
            )
            key = next(iter(sparse_embedding.keys()))
            sparse_embedding = sparse_embedding[key]
            batch = batch[key]
        dense, mask = pyg.utils.to_dense_batch(
            sparse_embedding, batch, fill_value=0.0
        )
        return SetEmbedding(dense_embedding=dense, padding_mask=mask)

    @staticmethod
    def from_flat(
        flat: Tensor,
        *,
        device: torch.device | None = None,
    ) -> SetEmbedding:
        """
        Build a SetEmbedding from per-candidate flat embeddings.

        Args:
            flat: Tensor of shape [N, F] or [N], one row per candidate.
            device: Optional device override.

        Returns:
            SetEmbedding with dense shape [N, 1, F] (F defaults to 1 when input is [N]) and mask all True.
        """
        if flat.dim() == 1:
            flat = flat.unsqueeze(-1)
        elif flat.dim() != 2:
            raise ValueError(
                f"Expected flat [N,F] or [N] tensor, got shape {tuple(flat.shape)}"
            )
        if device is not None:
            flat = flat.to(device)
        N, F = flat.shape
        dense = flat.unsqueeze(1)
        mask = torch.ones((N, 1), dtype=torch.bool, device=dense.device)
        return SetEmbedding(dense, mask)


def mask_to_batch_indices(mask: torch.Tensor):
    """
        Given the mask indicating which symbols are real, return the batch indices.
        This can be seen as the inverse of `torch_geometric.utils.to_dense_batch`.
        batch[i] = j means that the i-th object belongs to the j-th graph.
    :param mask: A boolean mask indicating which symbol embeddings are real and which are paddings.
        A truthy value means a real embedding at this location.
        Can be with or without batch dimension.
    :return: A tensor of batch indices. shape [N] where N is the number of objects.
    """
    return torch.arange(mask.size(0), device=mask.device).repeat_interleave(
        mask.sum(dim=-1)
    )


class SetPooling(torch.nn.Module):
    """
    Global pooling for SetEmbedding tensors.

    Why this exists:
    - SetEmbedding stores dense, masked per-symbol embeddings with shape [B, O, F].
      Pooling along the set axis using NaN-masked reductions is often faster and
      simpler than flattening to (x, batch) and using scatter-based poolers for
      small sets.

    Use cases:
    - Pool per-symbol (typically per-object) embeddings into a fixed-size per-state
      embedding efficiently.
    - Provide a pendant to torch_geometric global poolers when you already have a
      SetEmbedding and prefer dense masked ops.

    Interop:
    - If you want to reuse torch_geometric global pooling, call
      SetEmbedding.as_flat() to get (x_flat, batch) and feed them to
      GlobalAddPool/GlobalMeanPool/etc.
    """

    options = ["sum", "mean", "max", "add"]

    @staticmethod
    def sum(dense_embedding: torch.Tensor):
        return SetPooling.add(dense_embedding)

    @staticmethod
    def add(dense_embedding: torch.Tensor):
        return torch.nansum(dense_embedding, dim=-2)

    @staticmethod
    def mean(dense_embedding: torch.Tensor):
        return torch.nansum(dense_embedding, dim=-2)

    @staticmethod
    def max(dense_embedding: torch.Tensor):
        return dense_embedding.nan_to_num(torch.finfo(dense_embedding.dtype).min).max(
            dim=-2
        )

    def __init__(
        self,
        pooling: Union[
            Literal["sum", "mean", "max", "add"],
            Callable[[Tensor], Tensor],
        ] = "sum",
    ) -> None:
        super().__init__()
        self.pooling: Callable[[Tensor], Tensor]

        if isinstance(pooling, str):
            pooling = self._verify_pool_str(pooling)
            self.pooling = getattr(SetPooling, pooling)
        else:
            self.raw_tensor_based = True
            self.pooling = pooling

    def _verify_pool_str(self, pooling):
        if pooling not in SetPooling.options:
            raise ValueError(
                f"Invalid pooling function: {pooling}. Choose from {self.options}."
            )
        return pooling.replace("sum", "add")

    def forward(self, object_embedding: SetEmbedding | TensorDictBase | torch.Tensor):
        if isinstance(object_embedding, torch.Tensor):
            return self.pooling(object_embedding)
        if isinstance(object_embedding, TensorDictBase):
            object_embedding = SetEmbedding.from_tensordict(object_embedding)
        else:
            assert isinstance(object_embedding, SetEmbedding)
        dense_embedding = object_embedding.dense_embedding
        assert dense_embedding.dim() >= 2
        return self.pooling(dense_embedding)


class ActionPoolingModule(torch.nn.Module):
    """
    Action pooling-functor: pools symbol embeddings into a per-graph context
    and concatenates it with each action position's own symbol embedding.

    Four configurable modes:
      - "mask" (default):
        pool only NON-action positions, concat per-action emb with that pool.
        The 'action' input MUST be a bool tensor for this mode.
        Output dim = 2 × embedding_size.
      - "no_mask": pool ALL symbol positions, concat per-action emb with
        full-graph pool, so the context includes the other action
        positions.  Accepts both bool and
        integer 'action' input.  Output dim = 2 × embedding_size.
      - "separate": pool action positions and non-action positions
        SEPARATELY, concat per-action emb with both pools.  Gives the
        readout three distinct signals per (action, graph) pair.  Output
        dim = 3 × embedding_size.
      - "args": concat per-action emb with its two ARGUMENT node embeddings
        (moved object, target cell) plus the non-action pool.  The pooled
        context is constant per state, so it cannot rank actions by anything
        the action embedding itself does not carry; the argument embeddings
        add that per-action information.  Requires the encoder's `action_args` graph attribute (arity-2
        actions only) passed via the `action_args` kwarg.  Output dim =
        4 × embedding_size.

    Downstream readouts must size their input dim via output_multiplier.
    """

    _OUTPUT_MULT = {"mask": 2, "no_mask": 2, "separate": 3, "args": 4}

    def __init__(
        self,
        pooling: torch.nn.Module,
        symbol_string: str = "_symbol_",
        mode: str = "mask",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.symbol_string = symbol_string
        if mode not in self._OUTPUT_MULT:
            raise ValueError(
                f"ActionPoolingModule: unknown mode {mode!r}; "
                f"valid: {list(self._OUTPUT_MULT)}"
            )
        self.mode = mode

    @property
    def output_multiplier(self) -> int:
        """Readout input dim = output_multiplier × embedding_size."""
        return self._OUTPUT_MULT[self.mode]

    def _resolve_action_bool(self, emb_size, device, action, target_positions):
        """Build a bool mask over all symbol positions identifying action rows.
        Accepts either a bool tensor or an int-index tensor; if both
        action and target_positions are None, raises."""
        if action is not None:
            if action.dtype == torch.bool:
                return action.flatten()
            mask = torch.zeros(emb_size, dtype=torch.bool, device=device)
            mask[action.flatten()] = True
            return mask
        if target_positions is None:
            raise ValueError("either action or target_positions must be provided")
        mask = torch.zeros(emb_size, dtype=torch.bool, device=device)
        mask[target_positions] = True
        return mask

    def pool(
        self,
        embeddings: torch.Tensor,  # [Batch, EmbeddingSize]
        batch_ids: torch.Tensor,   # [Batch]
        action: torch.Tensor,      # bool mask or int index tensor [Batch]
        target_positions: torch.Tensor = None,  # int [Batch] (used when action is None)
        action_args: torch.Tensor = None,  # long [n_actions, 2], graph-local ("args" mode)
    ):
        emb = embeddings[self.symbol_string]
        b = batch_ids[self.symbol_string]

        if self.mode == "args":
            if action is None or action.dtype != torch.bool:
                raise ValueError("ActionPoolingModule(args): a bool action mask is required")
            if action_args is None:
                raise ValueError(
                    "ActionPoolingModule(args): action_args missing, as the encoder "
                    "withholds it for domains whose actions are not all arity-2"
                )
            abool = action.flatten()
            action_embeddings = emb[abool]
            action_batch = b[abool]
            non_action_mask = ~abool
            object_embeddings = self.pooling(emb[non_action_mask], b[non_action_mask])
            # action_args rows align with the True entries of the mask (encoder
            # contract); indices are local per graph, offset by each graph's
            # first symbol row.  Symbol rows are grouped by graph in batch order.
            counts = torch.bincount(b)
            starts = torch.cumsum(counts, dim=0) - counts
            gidx = action_args.to(emb.device) + starts[action_batch].unsqueeze(1)
            paired_embeddings = torch.cat(
                [
                    action_embeddings,
                    emb[gidx[:, 0]],
                    emb[gidx[:, 1]],
                    object_embeddings[action_batch],
                ],
                dim=-1,
            )
            return paired_embeddings, action_batch

        if self.mode == "mask":
            # ---- "mask" mode ----
            if action is not None:
                non_action_mask = ~action.flatten()  # requires action to be bool
            else:
                non_action_mask = torch.ones(emb.size(0), dtype=torch.bool, device=emb.device)
                non_action_mask[target_positions] = False

            object_embeddings = self.pooling(emb[non_action_mask], b[non_action_mask])

            if action is not None:
                action_embeddings = emb[action.flatten()]
                action_batch = b[action.flatten()]
            else:
                action_embeddings = emb[target_positions]
                action_batch = b[target_positions]

            matched_objects = object_embeddings[action_batch]
            paired_embeddings = torch.cat([action_embeddings, matched_objects], dim=-1)
            return paired_embeddings, action_batch

        # ---- "no_mask" and "separate" share index/mask resolution ----
        action_bool = self._resolve_action_bool(emb.size(0), emb.device, action, target_positions)
        action_embeddings = emb[action_bool]
        action_batch = b[action_bool]

        if self.mode == "no_mask":
            # Pool ALL symbols (including action positions themselves)
            object_embeddings = self.pooling(emb, b)
            matched_objects = object_embeddings[action_batch]
            paired_embeddings = torch.cat([action_embeddings, matched_objects], dim=-1)
            return paired_embeddings, action_batch

        # mode == "separate"
        non_action_bool = ~action_bool
        action_pool = self.pooling(emb[action_bool], b[action_bool])
        non_action_pool = self.pooling(emb[non_action_bool], b[non_action_bool])
        matched_actions = action_pool[action_batch]
        matched_non_actions = non_action_pool[action_batch]
        paired_embeddings = torch.cat(
            [action_embeddings, matched_actions, matched_non_actions], dim=-1
        )
        return paired_embeddings, action_batch

