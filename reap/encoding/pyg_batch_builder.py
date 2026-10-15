# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch_geometric.data import Batch, HeteroData

from .pyg_builder import PygBuilder, PygBuilderBase


class BatchBuilderBase(ABC):
    @abstractmethod
    def append(self, builder: PygBuilderBase) -> None: ...

    @abstractmethod
    def build(self) -> Batch: ...


class PygDataBatchBuilder(BatchBuilderBase):
    """Stream a batch of homogeneous graphs without intermediate Data objects."""

    def __init__(self) -> None:
        self._node_names: List[List[str]] = []
        self._x_parts: List[torch.Tensor] = []
        self._edge_index_parts: List[torch.Tensor] = []
        self._edge_attr_parts: List[torch.Tensor] = []
        self._batch_parts: List[torch.Tensor] = []
        self._ptr: List[int] = [0]
        self._node_offset = 0
        self._graph_count = 0
        self._has_x: bool | None = None
        self._has_edge_attr = False

    def append(self, builder: PygBuilderBase) -> None:
        node_keys = builder.node_keys[PygBuilder._NODE_TYPE]
        node_count = len(node_keys)
        self._node_names.append(list(node_keys))

        node_types = builder.node_attrs[PygBuilder._NODE_TYPE].get("type", [])
        has_x = bool(node_types)
        if self._has_x is None:
            self._has_x = has_x
        elif self._has_x != has_x:
            raise ValueError("Inconsistent node feature usage across batch.")

        if has_x:
            values = [v if v is not None else 0 for v in node_types]
            x = torch.as_tensor(values).view(-1, 1)
            self._x_parts.append(x)
        elif node_count:
            self._x_parts.append(torch.empty((node_count, 0), dtype=torch.float32))

        if node_count:
            self._batch_parts.append(
                torch.full((node_count,), self._graph_count, dtype=torch.long)
            )
        else:
            self._batch_parts.append(torch.empty((0,), dtype=torch.long))
        self._ptr.append(self._ptr[-1] + node_count)

        edge_attr_values: List[Any] = []
        edges: List[Tuple[int, int]] = []
        for edge_key, indices in builder.edge_indices.items():
            attr_list = builder.edge_attrs.get(edge_key, {}).get("type", [])
            for idx, (src_idx, dst_idx) in enumerate(indices):
                value = attr_list[idx] if idx < len(attr_list) else 0
                edges.append((src_idx + self._node_offset, dst_idx + self._node_offset))
                if attr_list:
                    edge_attr_values.append(value)
                    self._has_edge_attr = True
                if src_idx != dst_idx:
                    edges.append(
                        (dst_idx + self._node_offset, src_idx + self._node_offset)
                    )
                    if attr_list:
                        edge_attr_values.append(value)

        if edges:
            self._edge_index_parts.append(
                torch.tensor(edges, dtype=torch.long).t().contiguous()
            )
        if self._has_edge_attr and edge_attr_values:
            self._edge_attr_parts.append(torch.as_tensor(edge_attr_values).view(-1, 1))

        self._node_offset += node_count
        self._graph_count += 1

    def build(self) -> Batch:
        batch = Batch()
        total_nodes = self._ptr[-1]

        if self._has_x:
            batch.x = (
                torch.cat(self._x_parts, dim=0)
                if self._x_parts
                else torch.empty((0, 1), dtype=torch.float32)
            )
        else:
            batch.x = None
            batch.num_nodes = total_nodes

        if self._edge_index_parts:
            batch.edge_index = torch.cat(self._edge_index_parts, dim=1)
        else:
            batch.edge_index = torch.empty((2, 0), dtype=torch.long)

        if self._has_edge_attr:
            batch.edge_attr = (
                torch.cat(self._edge_attr_parts, dim=0)
                if self._edge_attr_parts
                else torch.empty((0, 1), dtype=torch.float32)
            )
        else:
            batch.edge_attr = None

        batch.batch = (
            torch.cat(self._batch_parts, dim=0)
            if self._batch_parts
            else torch.empty((0,), dtype=torch.long)
        )
        batch.ptr = torch.tensor(self._ptr, dtype=torch.long)
        batch.node_names = list(self._node_names)
        batch._num_graphs = self._graph_count
        return batch


class HeteroBatchBuilder(BatchBuilderBase):
    """Stream a batch of hetero graphs without intermediate HeteroData objects."""

    def __init__(
        self,
        node_types: Iterable[str],
        edge_types: Iterable[Tuple[str, str, str]],
        *,
        strict_types: bool = True,
    ) -> None:
        self._node_types = list(node_types)
        self._edge_types = list(edge_types)
        self._strict_types = strict_types
        self._node_offsets: Dict[str, int] = {ntype: 0 for ntype in self._node_types}
        self._node_counts: Dict[str, List[int]] = {
            ntype: [] for ntype in self._node_types
        }
        self._node_names: Dict[str, List[List[str]]] = {
            ntype: [] for ntype in self._node_types
        }
        self._x_parts: Dict[str, List[torch.Tensor]] = {
            ntype: [] for ntype in self._node_types
        }
        self._batch_parts: Dict[str, List[torch.Tensor]] = {
            ntype: [] for ntype in self._node_types
        }
        self._node_attr_parts: Dict[str, Dict[str, List[Any]]] = {
            ntype: defaultdict(list) for ntype in self._node_types
        }
        self._edge_parts: Dict[Tuple[str, str, str], List[torch.Tensor]] = {
            etype: [] for etype in self._edge_types
        }
        self._graph_attrs: Dict[str, List[Any]] = defaultdict(list)
        self._graph_count = 0

    def append(self, builder: PygBuilderBase) -> None:
        start_offsets = dict(self._node_offsets)
        nodes_dict = builder.node_keys
        if self._strict_types:
            unexpected_nodes = set(nodes_dict.keys()) - set(self._node_types)
            if unexpected_nodes:
                raise KeyError(f"Unexpected node types in builder: {unexpected_nodes}.")
            unexpected_edges = set(builder.edge_indices.keys()) - set(self._edge_types)
            if unexpected_edges:
                raise KeyError(f"Unexpected edge types in builder: {unexpected_edges}.")
        for node_type in self._node_types:
            nodes = list(nodes_dict.get(node_type, []))
            self._node_names[node_type].append(nodes)
            self._node_counts[node_type].append(len(nodes))
            x = self._build_x(builder, node_type, nodes)
            self._x_parts[node_type].append(x)
            if nodes:
                self._batch_parts[node_type].append(
                    torch.full((len(nodes),), self._graph_count, dtype=torch.long)
                )
            else:
                self._batch_parts[node_type].append(torch.empty((0,), dtype=torch.long))
            extra_attrs = self._build_node_attrs(builder, node_type, nodes)
            for attr, value in extra_attrs.items():
                self._node_attr_parts[node_type][attr].append(value)
            self._node_offsets[node_type] = start_offsets[node_type] + len(nodes)

        for edge_type in self._edge_types:
            indices = builder.edge_indices.get(edge_type, [])
            if not indices:
                continue
            src_type, _, dst_type = edge_type
            idx = torch.tensor(indices, dtype=torch.long).t().contiguous()
            idx[0] += start_offsets[src_type]
            idx[1] += start_offsets[dst_type]
            self._edge_parts[edge_type].append(idx)

        graph_attrs = self._build_graph_attrs(builder)
        for key, value in graph_attrs.items():
            self._graph_attrs[key].append(value)

        self._graph_count += 1

    def build(self) -> Batch:
        batch = Batch(_base_cls=HeteroData)
        for node_type in self._node_types:
            x = torch.cat(self._x_parts[node_type], dim=0)
            batch[node_type].x = x
            batch[node_type].batch = torch.cat(self._batch_parts[node_type], dim=0)
            counts = self._node_counts[node_type]
            ptr = [0]
            for count in counts:
                ptr.append(ptr[-1] + count)
            batch[node_type].ptr = torch.tensor(ptr, dtype=torch.long)
            batch[node_type].node_names = list(self._node_names[node_type])

            for attr, parts in self._node_attr_parts[node_type].items():
                if parts and all(torch.is_tensor(p) for p in parts):
                    batch[node_type][attr] = torch.cat(parts, dim=0)
                else:
                    batch[node_type][attr] = list(parts)

        for edge_type in self._edge_types:
            parts = self._edge_parts[edge_type]
            if parts:
                batch[edge_type].edge_index = torch.cat(parts, dim=1)
            else:
                batch[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        for key, values in self._graph_attrs.items():
            if values and all(torch.is_tensor(v) for v in values):
                setattr(batch, key, torch.cat(values, dim=0))
            else:
                setattr(batch, key, list(values))

        batch._num_graphs = self._graph_count
        return batch

    def _build_x(
        self,
        builder: PygBuilderBase,
        node_type: str,
        nodes: List[str],
    ) -> torch.Tensor:
        raise NotImplementedError

    def _build_node_attrs(
        self,
        builder: PygBuilderBase,
        node_type: str,
        nodes: List[str],
    ) -> Dict[str, Any]:
        return {}

    def _build_graph_attrs(self, builder: PygBuilderBase) -> Dict[str, Any]:
        return {}


class HGraphBatchBuilder(HeteroBatchBuilder):
    def __init__(
        self,
        *,
        relation_dict: Dict[str, int],
        symbol_type_id: str,
        edge_types: Iterable[Tuple[str, str, str]],
    ) -> None:
        node_types = list(relation_dict.keys())
        if symbol_type_id not in node_types:
            node_types.append(symbol_type_id)
        super().__init__(
            node_types=node_types, edge_types=edge_types, strict_types=True
        )
        self._relation_dict = relation_dict
        self._symbol_type_id = symbol_type_id

    def _build_x(
        self,
        builder: PygBuilderBase,
        node_type: str,
        nodes: List[str],
    ) -> torch.Tensor:
        size = (
            1 if node_type == self._symbol_type_id else self._relation_dict[node_type]
        )
        return torch.zeros((len(nodes), size), dtype=torch.float32)

    def _build_graph_attrs(self, builder: PygBuilderBase) -> Dict[str, Any]:
        object_names: List[str] = []
        if self._symbol_type_id in builder.node_attrs:
            object_names = list(
                builder.node_attrs[self._symbol_type_id].get("name", [])
            )
        if not object_names and self._symbol_type_id in builder.node_keys:
            object_names = list(builder.node_keys[self._symbol_type_id])
        return {"object_names": object_names}


class ILGBatchBuilder(HeteroBatchBuilder):
    def __init__(
        self,
        *,
        relation_dict: Dict[str, int],
        symbol_type_id: str,
        edge_types: Iterable[Tuple[str, str, str]],
    ) -> None:
        node_types = list(relation_dict.keys())
        if symbol_type_id not in node_types:
            node_types.append(symbol_type_id)
        super().__init__(
            node_types=node_types, edge_types=edge_types, strict_types=True
        )
        self._relation_dict = relation_dict
        self._symbol_type_id = symbol_type_id

    def _build_x(
        self,
        builder: PygBuilderBase,
        node_type: str,
        nodes: List[str],
    ) -> torch.Tensor:
        if node_type == self._symbol_type_id:
            return torch.zeros((len(nodes), 2), dtype=torch.float32)
        arity = self._relation_dict[node_type]
        x = torch.empty((len(nodes), arity + 1), dtype=torch.float32)
        statuses = builder.node_attrs[node_type].get("status", [])
        for i in range(len(nodes)):
            status = statuses[i] if i < len(statuses) else None
            x[i].fill_(status.encode() if status is not None else 0)
        return x

    def _build_graph_attrs(self, builder: PygBuilderBase) -> Dict[str, Any]:
        object_names: List[str] = []
        if self._symbol_type_id in builder.node_attrs:
            object_names = list(
                builder.node_attrs[self._symbol_type_id].get("name", [])
            )
        if not object_names and self._symbol_type_id in builder.node_keys:
            object_names = list(builder.node_keys[self._symbol_type_id])
        return {"object_names": object_names}


class HorizonBatchBuilder(HeteroBatchBuilder):
    def __init__(
        self,
        *,
        relation_dict: Dict[str, int],
        symbol_type_id: str,
        edge_types: Iterable[Tuple[str, str, str]],
        target_symbol_prefix: str,
        parent_relation: str,
        exclude_root_candidate: bool,
    ) -> None:
        node_types = list(relation_dict.keys())
        if symbol_type_id not in node_types:
            node_types.append(symbol_type_id)
        super().__init__(
            node_types=node_types, edge_types=edge_types, strict_types=True
        )
        self._relation_dict = relation_dict
        self._symbol_type_id = symbol_type_id
        self._target_symbol_prefix = target_symbol_prefix
        self._parent_relation = parent_relation
        self._exclude_root_candidate = exclude_root_candidate

    def _build_x(
        self,
        builder: PygBuilderBase,
        node_type: str,
        nodes: List[str],
    ) -> torch.Tensor:
        size = (
            1 if node_type == self._symbol_type_id else self._relation_dict[node_type]
        )
        return torch.zeros((len(nodes), size), dtype=torch.float32)

    def _build_node_attrs(
        self,
        builder: PygBuilderBase,
        node_type: str,
        nodes: List[str],
    ) -> Dict[str, Any]:
        if node_type != self._symbol_type_id:
            return {}
        if not nodes:
            return {"is_choice": torch.empty((0,), dtype=torch.bool)}
        target_positions, _, _, _ = self._collect_target_metadata(builder, nodes)
        is_choice = torch.zeros(len(nodes), dtype=torch.bool)
        candidate_positions = (
            target_positions[1:] if self._exclude_root_candidate else target_positions
        )
        if candidate_positions:
            is_choice[candidate_positions] = True
        return {"is_choice": is_choice}

    def _build_graph_attrs(self, builder: PygBuilderBase) -> Dict[str, Any]:
        object_names: List[str] = []
        for node_type, nodes in builder.node_keys.items():
            if node_type == self._symbol_type_id:
                continue
            names = builder.node_attrs.get(node_type, {}).get("name", [])
            if names:
                object_names.extend(names)

        symbol_nodes = builder.node_keys.get(self._symbol_type_id, [])
        target_positions, target_indices, target_depths, target_names = (
            self._collect_target_metadata(builder, list(symbol_nodes))
        )

        return {
            "object_names": object_names,
            "target_symbol_prefix": self._target_symbol_prefix,
            "target_names": target_names,
            "target_positions": target_positions,
            "parent_relation": self._parent_relation,
            "target_depths": target_depths,
            "target_indices": (
                torch.tensor(target_indices, dtype=torch.long)
                if target_indices
                else torch.empty(0, dtype=torch.long)
            ),
        }

    def _collect_target_metadata(
        self,
        builder: PygBuilderBase,
        symbol_nodes: List[str],
    ) -> tuple[list[int], list[int], list[int | None], list[str | None]]:
        target_positions: list[int] = []
        target_indices: list[int] = []
        target_depths: list[int | None] = []
        target_names: list[str | None] = []

        symbol_attrs = builder.node_attrs.get(self._symbol_type_id, {})
        target_index_list = symbol_attrs.get("target_index", [])
        depth_list = symbol_attrs.get("depth", [])
        target_name_list = symbol_attrs.get("target", [])

        for idx, _node in enumerate(symbol_nodes):
            if idx >= len(target_index_list):
                continue
            target_index = target_index_list[idx]
            target_indices.append(target_index)
            target_positions.append(idx)
            depth = depth_list[idx] if idx < len(depth_list) else None
            target_depths.append(depth)
            target_name = target_name_list[idx] if idx < len(target_name_list) else None
            target_names.append(target_name)

        return target_positions, target_indices, target_depths, target_names
