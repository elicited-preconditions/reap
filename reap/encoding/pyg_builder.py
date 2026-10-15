# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple, TypeVar, Union

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData

PygDataT = TypeVar("PygDataT", bound=Union[Data, HeteroData])


def _to_tensor_or_list(values: List[Any]) -> Union[Tensor, List[Any]]:
    if not values:
        return []

    # Filter out None values to decide on the type
    non_none_values = [v for v in values if v is not None]
    if not non_none_values:
        return values

    first = non_none_values[0]
    if isinstance(first, (int, float, bool, list, tuple, Tensor)):
        try:
            # Replace None with zeros of appropriate shape
            if any(v is None for v in values):
                if isinstance(first, Tensor):
                    default = torch.zeros_like(first)
                elif isinstance(first, (list, tuple)):
                    default = [0] * len(first)
                else:
                    default = 0
                values = [v if v is not None else default for v in values]
            return torch.as_tensor(values)
        except (TypeError, ValueError, RuntimeError):
            return values
    return values


def _nested_defaultdict_list() -> "defaultdict[str, List[Any]]":
    # Module-level so that PygBuilderBase instances can be pickled: local
    # lambda factories block torch.multiprocessing IPC of any object that
    # transitively holds a builder (e.g. Model.encoder).
    return defaultdict(list)


class PygBuilderBase:
    def __init__(self):
        self.node_keys: Dict[str, List[str]] = defaultdict(list)
        self.node_indices: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.node_attrs: Dict[str, Dict[str, List[Any]]] = defaultdict(
            _nested_defaultdict_list
        )
        self.edge_indices: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = (
            defaultdict(list)
        )
        self.edge_attrs: Dict[Tuple[str, str, str], Dict[str, List[Any]]] = defaultdict(
            _nested_defaultdict_list
        )
        self.graph_attrs: Dict[str, Any] = {}
        self.encoder_hash: str | None = None
        self._last_edge_key: Tuple[str, str, str] | None = None
        self._last_edge_idx: int | None = None

    def reset(self):
        self.node_keys.clear()
        self.node_indices.clear()
        self.node_attrs.clear()
        self.edge_indices.clear()
        self.edge_attrs.clear()
        self.graph_attrs.clear()
        self._last_edge_key: Tuple[str, str, str] | None = None
        self._last_edge_idx: int | None = None

    def set_graph_attr(self, key: str, value: Any):
        self.graph_attrs[key] = value

    def get_graph_attr(self, key: str, default: Any = None) -> Any:
        return self.graph_attrs.get(key, default)


class PygHeteroBuilder(PygBuilderBase):
    """Builder for `torch_geometric.data.HeteroData`."""

    def add_node(self, node_key: str, node_type: str, **attrs) -> int:
        """Add a node to a hetero graph.

        Assumptions:
          - `node_type` must be provided.
        """
        if node_type is None:
            raise ValueError(
                "PygHeteroBuilder.add_node requires an explicit node_type."
            )

        if node_key not in self.node_indices[node_type]:
            idx = len(self.node_indices[node_type])
            self.node_indices[node_type][node_key] = idx
            self.node_keys[node_type].append(node_key)
            for k, v in attrs.items():
                self.node_attrs[node_type][k].append(v)
            return idx

        idx = self.node_indices[node_type][node_key]
        for k, v in attrs.items():
            attr_list = self.node_attrs[node_type][k]
            while len(attr_list) <= idx:
                attr_list.append(None)
            attr_list[idx] = v
        return idx

    def set_node_attr(
        self, node_key: str, attr: str, value: Any, node_type: str
    ) -> None:
        """Set a node attribute in a hetero graph.

        Assumptions:
          - `node_type` must be provided.
          - The node must already exist under that type.
        """
        if node_key not in self.node_indices[node_type]:
            raise KeyError(f"Node '{node_key}' not found in node_type '{node_type}'.")

        idx = self.node_indices[node_type][node_key]
        attr_list = self.node_attrs[node_type][attr]
        while len(attr_list) <= idx:
            attr_list.append(None)
        attr_list[idx] = value

    def add_edge(
        self,
        src_key: str,
        dst_key: str,
        src_type: str,
        dst_type: str,
        edge_type: str,
    ) -> None:
        """Add an edge to a hetero graph.

        Assumptions:
          - `src_type`, `dst_type`, and `edge_type` must be provided.
        """
        if src_type is None or dst_type is None or edge_type is None:
            raise ValueError(
                "PygHeteroBuilder.add_edge requires src_type, dst_type, and edge_type."
            )

        if src_key not in self.node_indices[src_type]:
            raise KeyError(
                f"Source node '{src_key}' not found in node_type '{src_type}'."
            )
        if dst_key not in self.node_indices[dst_type]:
            raise KeyError(
                f"Destination node '{dst_key}' not found in node_type '{dst_type}'."
            )

        src_idx = self.node_indices[src_type][src_key]
        dst_idx = self.node_indices[dst_type][dst_key]
        key = (src_type, edge_type, dst_type)
        self.edge_indices[key].append((src_idx, dst_idx))
        self._last_edge_key = key
        self._last_edge_idx = len(self.edge_indices[key]) - 1

    def set_edge_attr(
        self,
        src_key: str,
        dst_key: str,
        attr: str,
        value: Any,
        src_type: str,
        dst_type: str,
        edge_type: str,
    ) -> None:
        """Set an edge attribute for the most recently added edge of a given edge type.

        Assumptions:
          - `src_type`, `dst_type`, and `edge_type` must be provided.
          - The edge must already exist (typically you call this right after `add_edge`).

        This method does not attempt to infer types.
        """
        key = (src_type, edge_type, dst_type)
        if key not in self.edge_indices or not self.edge_indices[key]:
            raise KeyError(f"No edges exist for edge type {key}.")

        # Default to the last edge that was added for this key.
        idx = len(self.edge_indices[key]) - 1
        attr_list = self.edge_attrs[key][attr]
        while len(attr_list) <= idx:
            attr_list.append(None)
        attr_list[idx] = value

    def build(self) -> HeteroData:
        data = HeteroData()
        if self.encoder_hash is not None:
            data.encoder_hash = self.encoder_hash

        for ntype, keys in self.node_keys.items():
            data[ntype].node_names = keys
            for attr, values in self.node_attrs[ntype].items():
                data[ntype][attr] = _to_tensor_or_list(values)
            # Ensure x is present if not set by attrs
            if "x" not in data[ntype]:
                data[ntype].x = torch.zeros((len(keys), 0))

        for (src_type, edge_type, dst_type), indices in self.edge_indices.items():
            if not indices:
                data[(src_type, edge_type, dst_type)].edge_index = torch.empty(
                    (2, 0), dtype=torch.long
                )
            else:
                data[(src_type, edge_type, dst_type)].edge_index = (
                    torch.tensor(indices, dtype=torch.long).t().contiguous()
                )
                for attr, values in self.edge_attrs[
                    (src_type, edge_type, dst_type)
                ].items():
                    data[(src_type, edge_type, dst_type)][attr] = _to_tensor_or_list(
                        values
                    )
        for k, v in self.graph_attrs.items():
            setattr(data, k, v)

        return data


class PygBuilder(PygBuilderBase):
    """Builder for `torch_geometric.data.Data` (homogeneous graphs).

    Assumptions:
      - There is exactly one node type.
      - `add_node`/`set_node_attr` do not accept a node type. (why?)
      - `add_edge` does not accept `src_type`/`dst_type`.
      - Edge types may be provided as a string and will be stored in `edge_attr` if any non-default type is used.
    """

    _NODE_TYPE = "node"

    def add_node(self, node_key: str, node_type: str | None = None, **attrs) -> int:
        if node_type is not None:
            raise ValueError(
                "PygBuilder.add_node does not accept node_type (homogeneous graph)."
            )
        if node_key not in self.node_indices[self._NODE_TYPE]:
            idx = len(self.node_indices[self._NODE_TYPE])
            self.node_indices[self._NODE_TYPE][node_key] = idx
            self.node_keys[self._NODE_TYPE].append(node_key)
            for k, v in attrs.items():
                self.node_attrs[self._NODE_TYPE][k].append(v)
            return idx

        idx = self.node_indices[self._NODE_TYPE][node_key]
        for k, v in attrs.items():
            attr_list = self.node_attrs[self._NODE_TYPE][k]
            while len(attr_list) <= idx:
                attr_list.append(None)
            attr_list[idx] = v
        return idx

    def set_node_attr(
        self, node_key: str, attr: str, value: Any, node_type: str | None = None
    ) -> None:
        if node_type is not None:
            raise ValueError(
                "PygBuilder.set_node_attr does not accept node_type (homogeneous graph)."
            )
        if node_key not in self.node_indices[self._NODE_TYPE]:
            raise KeyError(f"Node '{node_key}' not found.")

        idx = self.node_indices[self._NODE_TYPE][node_key]
        attr_list = self.node_attrs[self._NODE_TYPE][attr]
        while len(attr_list) <= idx:
            attr_list.append(None)
        attr_list[idx] = value

    def add_edge(self, src_key: str, dst_key: str, *args, **kwargs) -> None:
        # Homogeneous: allow optional edge_type as positional arg OR keyword, but disallow src_type/dst_type.
        if "src_type" in kwargs or "dst_type" in kwargs:
            raise ValueError(
                "PygBuilder.add_edge does not accept src_type/dst_type (homogeneous graph)."
            )

        edge_type = kwargs.pop("edge_type", None)
        if len(args) == 1:
            edge_type = args[0]
        elif len(args) != 0:
            raise TypeError(
                "PygBuilder.add_edge accepts at most one positional arg: edge_type."
            )

        edge_type = edge_type or "edge"

        if src_key not in self.node_indices[self._NODE_TYPE]:
            raise KeyError(f"Source node '{src_key}' not found.")
        if dst_key not in self.node_indices[self._NODE_TYPE]:
            raise KeyError(f"Destination node '{dst_key}' not found.")

        src_idx = self.node_indices[self._NODE_TYPE][src_key]
        dst_idx = self.node_indices[self._NODE_TYPE][dst_key]
        key = (self._NODE_TYPE, edge_type, self._NODE_TYPE)
        self.edge_indices[key].append((src_idx, dst_idx))
        self._last_edge_key = key
        self._last_edge_idx = len(self.edge_indices[key]) - 1

    def set_edge_attr(
        self,
        src_key: str,
        dst_key: str,
        attr: str,
        value: Any,
        src_type: str | None = None,
        dst_type: str | None = None,
        edge_type: str | None = None,
    ) -> None:
        # Homogeneous: src_type/dst_type must not be used. edge_type is optional.
        if src_type is not None or dst_type is not None:
            raise ValueError(
                "PygBuilder.set_edge_attr does not accept src_type/dst_type (homogeneous graph)."
            )

        edge_type = edge_type or "edge"
        key = (self._NODE_TYPE, edge_type, self._NODE_TYPE)
        if key not in self.edge_indices or not self.edge_indices[key]:
            raise KeyError(f"No edges exist for edge type {key}.")

        idx = len(self.edge_indices[key]) - 1
        attr_list = self.edge_attrs[key][attr]
        while len(attr_list) <= idx:
            attr_list.append(None)
        attr_list[idx] = value

    def build(self) -> Data:
        data = Data()
        if self.encoder_hash is not None:
            data.encoder_hash = self.encoder_hash

        node_names = self.node_keys[self._NODE_TYPE]
        data.node_names = node_names

        # Node attributes
        for attr, values in self.node_attrs[self._NODE_TYPE].items():
            data[attr] = _to_tensor_or_list(values)

        # Edges
        all_indices: List[Tuple[int, int]] = []
        all_edge_types: List[Union[int, str]] = []
        for (src_type, e_type, dst_type), indices in self.edge_indices.items():
            # src_type/dst_type will both be "node" here.
            for s, d in indices:
                all_indices.append((s, d))
                all_edge_types.append(
                    int(e_type)
                    if isinstance(e_type, str) and e_type.isdigit()
                    else e_type
                )

        if not all_indices:
            data.edge_index = torch.empty((2, 0), dtype=torch.long)
            data.edge_attr = None
        else:
            data.edge_index = (
                torch.tensor(all_indices, dtype=torch.long).t().contiguous()
            )
            if any(e != "edge" for e in all_edge_types):
                data.edge_attr = _to_tensor_or_list(all_edge_types)
            else:
                data.edge_attr = None

        for k, v in self.graph_attrs.items():
            setattr(data, k, v)

        return data
