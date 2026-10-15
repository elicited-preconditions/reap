# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from __future__ import annotations

import numbers
import operator
from collections import defaultdict
from itertools import chain
from typing import Dict, Iterable, List, NamedTuple, Sequence

import networkx as nx
import torch
from matplotlib.patches import Patch
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from reap.logging_setup import get_logger
from reap.xmimir import XAction, XAtom, XDomain, XLiteral, XPredicate, atom_str_template

from .base_encoder import (
    EncoderFactory,
    GraphEncoderBase,
    check_encoded_by_this,
)
from .hetero_encoder import HGraphEncoder
from .pyg_batch_builder import ILGBatchBuilder
from .pyg_builder import PygBuilderBase, PygHeteroBuilder
from .relation_dict import RelationDict
from .relation_formatter import Node, relation_formatter


class PredicateEdgeType(NamedTuple):
    src_type: str
    pos: str
    dst_type: str


class AtomStatus(NamedTuple):
    is_regular: bool = True
    is_negated: bool = False
    is_satisfied: bool = False
    goal_levels: tuple[int, ...] = tuple()  # (0 - top level, 1 - subgoal, ...)

    def encode(self) -> int:
        """
        Flatten AtomStatus into a single integer.
        Encoding order:
          bit0 = is_regular
          bit1 = is_negated
          bit2 = is_satisfied
          higher bits (>=3) = bitmask of goal_levels
        """
        bits = self.is_regular | (self.is_negated << 1) | (self.is_satisfied << 2)
        return (self._mask() << 3) | bits

    @staticmethod
    def decode(value: int | float) -> "AtomStatus":
        """Decode AtomStatus from an integer produced by `encode`."""
        value = int(value)
        bits = value & 0b111  # lower 3 bits
        mask = value >> 3
        # Convert mask back to an ordered tuple of levels
        levels: list[int] = []
        lvl = 0
        while mask:
            if mask & 1:
                levels.append(lvl)
            mask >>= 1
            lvl += 1
        return AtomStatus(
            is_regular=bool(bits & 0b001),
            is_negated=bool(bits & 0b010),
            is_satisfied=bool(bits & 0b100),
            goal_levels=tuple(levels),
        )

    @staticmethod
    def encoding_limits(max_goal_level: int | None) -> tuple[int, int]:
        """
        Return the exclusive upper limit of the integer encoding for a given maximum goal level.
        This can be used to define embedding layers that cover all possible AtomStatus values
        for a given maximum goal level.
        """
        # +1 because we encode from 0 onwards, so max_encoding + 1 is the number of distinct encodings
        if max_goal_level is None:
            return 0, AtomStatus(True, True, True, ()).encode() + 1
        return (
            0,
            AtomStatus(True, True, True, tuple(range(max_goal_level + 1))).encode() + 1,
        )

    def has_level(self, level: int) -> bool:
        return level in self.goal_levels

    def _mask(self) -> int:
        mask = 0
        for lvl in self.goal_levels:
            if lvl < 0:
                continue
            mask |= 1 << lvl
        return mask


class ILGHGraphEncoder(GraphEncoderBase[HeteroData]):
    """
    Encode a planning state as a heterogeneous graph (Instance‑Learning Graph, ILG).

    Nodes
    -----
    - **Objects**: one node per object, node type `symbol_type_id` (default: `"obj"`). Feature `x` has shape `[*, 2]`.
    - **Atoms**: one node per ground atom (including missing goal atoms), node type equals the predicate name. Feature `x`
      has shape `[*, arity + 1]` and is filled with the integer value of `AtomStatus`.

    Edges
    -----
    - For an atom `p(o0, ..., ok-1)`, connect each argument object to the atom with label `position = i`.
    - Arity‑0 predicates connect to a dedicated placeholder object when `add_nullary_predicates=True`;
      otherwise they remain isolated (no incident edges).
    - For PyG, we materialize **both directions**: `(obj -> atom)` and `(atom -> obj)` for each position.

    Notes
    -----
    - Follows the ILG approach (see [1]). Goals are encoded via the `AtomStatus` colour of the atom; we do not create
      separate goal nodes.
    - Edge types are keyed by `(src_type, position, dst_type)` so each positional incidence becomes its own relation.

    References
    ----------
    [1] https://arxiv.org/abs/2403.16508
    """

    def __init__(
        self,
        domain: XDomain,
        *,
        relation_dict: RelationDict = None,
        symbol_type_id: str = RelationDict.default_symbol_ntype,
        action_type_id: str = RelationDict.default_action_ntype,
        nullary_object_name: str = relation_formatter.default_nullary_symbol_name,
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        lgan_nn_edge_pos: str = "lgan_nn",
        **relation_dict_kwargs,
    ) -> None:
        super().__init__(
            domain=domain, builder=PygHeteroBuilder(), graph_t=nx.MultiGraph
        )
        self.symbol_type_id: str = symbol_type_id
        self.action_type_id: str = action_type_id
        self.add_nullary_predicates: bool = add_nullary_predicates
        self.include_lgan_edges: bool = include_lgan_edges
        self.lgan_nn_edge_pos: str = lgan_nn_edge_pos
        self.nullary_object_name: str = nullary_object_name
        self.predicates: tuple[XPredicate, ...] = self.domain.predicates()
        self.relation_dict: RelationDict = relation_dict or RelationDict(
            self.predicates, **relation_dict_kwargs
        )
        # now filter out all goal predicates from relation_dict, they are not duplicated in this encoder, but encoded
        # in the AtomStatus of the regular predicate node
        self.relation_dict.update(
            {
                pred: arity
                for pred, arity in self.relation_dict.items()
                if not any(
                    pred.endswith(suffix)
                    for suffix in filter(
                        bool, relation_formatter.goal_level_suffixes.values()
                    )
                )
            }
        )
        if self.domain.actions:
            max_action_arity = max(action.arity for action in self.domain.actions)
            self.relation_dict.update({self.action_type_id: max_action_arity})
        self.all_edge_types = []
        for predicate, arity in self.relation_dict.items():
            effective_arity = (
                1 if (self.add_nullary_predicates and arity == 0) else arity
            )
            for pos in range(effective_arity):
                self.all_edge_types.append((self.symbol_type_id, str(pos), predicate))
                self.all_edge_types.append((predicate, str(pos), self.symbol_type_id))
        if self.include_lgan_edges:
            for predicate in self.relation_dict.keys():
                self.all_edge_types.append(
                    (predicate, self.lgan_nn_edge_pos, self.symbol_type_id)
                )
        self._ensure_config_hash()

    def _config_dict(self) -> dict:
        return {
            "symbol_type_id": self.symbol_type_id,
            "action_type_id": self.action_type_id,
            "add_nullary_predicates": self.add_nullary_predicates,
            "include_lgan_edges": self.include_lgan_edges,
            "lgan_nn_edge_pos": self.lgan_nn_edge_pos,
            "relation_dict": self.relation_dict.__repr__(),
        }

    def __eq__(self, other) -> bool:
        if not isinstance(other, ILGHGraphEncoder):
            return NotImplemented
        return (
            self.symbol_type_id == other.symbol_type_id
            and self.predicates == other.predicates
            and self.relation_dict == other.relation_dict
            and self.add_nullary_predicates == other.add_nullary_predicates
            and self.include_lgan_edges == other.include_lgan_edges
            and self.nullary_object_name == other.nullary_object_name
        )

    def as_factory(self) -> EncoderFactory:
        """
        Return a lightweight factory that recreates this encoder without copying the domain.

        Useful for multiprocessing or (de)serialization of encoder configuration.
        """
        return EncoderFactory(
            encoder_class=self.__class__,
            kwargs={
                "symbol_type_id": self.symbol_type_id,
                "action_type_id": self.action_type_id,
                "relation_dict": self.relation_dict,
                "add_nullary_predicates": self.add_nullary_predicates,
                "include_lgan_edges": self.include_lgan_edges,
                "lgan_nn_edge_pos": self.lgan_nn_edge_pos,
                "nullary_object_name": self.nullary_object_name,
            },
        )

    def _encode(
        self,
        builder: PygBuilderBase,
        facts: Sequence[XAtom],
        goals: Sequence[XLiteral],
        actions: Sequence[XAction],
        **kwargs,
    ):
        # Build hetero graph from state
        # One node for each object
        # One node for each atom
        # Edge label = position in atom

        # Get goal level map from builder (set by base class)
        goal_level_map = builder.get_graph_attr("goal_level_map")

        missing_goal_facts, statuses = self._compute_statuses(
            facts, goals, goal_level_map
        )

        for obj in self._contained_objects(chain(facts, goals, actions)):
            builder.add_node(
                relation_formatter(obj), self.symbol_type_id, name=obj.name
            )

        if self.add_nullary_predicates:
            builder.add_node(
                self.nullary_object_name,
                self.symbol_type_id,
                name=self.nullary_object_name,
            )

        for atom in chain(facts, missing_goal_facts):
            predicate = atom.predicate
            atom_node = relation_formatter(atom)
            builder.add_node(
                atom_node,
                relation_formatter(predicate),
                status=statuses[atom],
            )
            objects = list(atom.objects)
            if not objects and predicate.arity == 0:
                if self.add_nullary_predicates:
                    objects = [self.nullary_object_name]
                else:
                    continue
            for pos, obj in enumerate(objects):
                # connect object nodes to atom node
                builder.add_edge(
                    relation_formatter(obj),
                    atom_node,
                    self.symbol_type_id,
                    relation_formatter(predicate),
                    str(pos),
                )
                # reverse: atom node to object node
                builder.add_edge(
                    atom_node,
                    relation_formatter(obj),
                    relation_formatter(predicate),
                    self.symbol_type_id,
                    str(pos),
                )

        for action in actions:
            action_node = relation_formatter(action)
            builder.add_node(
                action_node,
                self.action_type_id,
                name=action_node,
            )
            for pos, obj in enumerate(action.objects):
                builder.add_edge(
                    relation_formatter(obj),
                    action_node,
                    self.symbol_type_id,
                    self.action_type_id,
                    str(pos),
                )
                # reverse
                builder.add_edge(
                    action_node,
                    relation_formatter(obj),
                    self.action_type_id,
                    self.symbol_type_id,
                    str(pos),
                )

        if self.include_lgan_edges:
            self._add_lgan_edges(builder, facts, missing_goal_facts)

        builder.set_graph_attr("object_names", builder.node_keys[self.symbol_type_id])

    def _add_lgan_edges(self, builder: PygBuilderBase, facts, missing_goal_facts):
        # Identify neighbors of each object
        object_neighbors = defaultdict(set)
        obj_to_atoms = defaultdict(list)
        for atom in chain(facts, missing_goal_facts):
            objs = list(atom.objects)
            for o1 in objs:
                obj_to_atoms[o1].append(atom)
                for o2 in objs:
                    if o1 != o2:
                        object_neighbors[o1].add(o2)

        for target_obj, atoms in obj_to_atoms.items():
            target_obj_node = relation_formatter(target_obj)
            neighbors = object_neighbors[target_obj]
            for atom in chain(facts, missing_goal_facts):
                atom_objs = list(atom.objects)
                if not atom_objs:
                    continue
                if target_obj in atom_objs:
                    continue
                if all(o in neighbors for o in atom_objs):
                    atom_node = relation_formatter(atom)
                    builder.add_edge(
                        target_obj_node,
                        atom_node,
                        self.symbol_type_id,
                        relation_formatter(atom.predicate),
                        self.lgan_nn_edge_pos,
                    )
                    builder.add_edge(
                        atom_node,
                        target_obj_node,
                        relation_formatter(atom.predicate),
                        self.symbol_type_id,
                        self.lgan_nn_edge_pos,
                    )

    def _compute_statuses(
        self, facts: Collection[XAtom], goals: Sequence[XLiteral], goal_level_map: dict
    ) -> tuple[list[XAtom], dict[XAtom, AtomStatus]]:
        goal_matches = {
            goal.atom
            for goal in goals
            if any(goal.atom.semantic_eq(f) for f in facts) == goal.polarity
        }

        statuses: dict[XAtom, AtomStatus] = defaultdict(lambda: AtomStatus())
        missing_goal_facts: list[XAtom] = []
        for goal in goals:
            atom = goal.atom
            is_satisfied = atom in goal_matches
            prev = statuses[atom]
            new_levels = tuple(sorted(set(prev.goal_levels) | {goal_level_map[goal]}))
            statuses[atom] = AtomStatus(
                is_regular=False,
                is_negated=goal.is_negated,
                is_satisfied=is_satisfied,
                goal_levels=new_levels,
            )
            if not is_satisfied:
                missing_goal_facts.append(atom)
        return missing_goal_facts, statuses

    def _build_data(self, builder: PygBuilderBase) -> HeteroData:
        nodes_dict = builder.node_keys
        data = HeteroData()

        object_names = []
        if self.symbol_type_id in builder.node_attrs:
            object_names = list(builder.node_attrs[self.symbol_type_id].get("name", []))
        if not object_names and self.symbol_type_id in nodes_dict:
            object_names = list(nodes_dict[self.symbol_type_id])
        data.object_names = object_names

        for node_type, nodes_of_type in nodes_dict.items():
            if node_type == self.symbol_type_id:
                x = torch.zeros((len(nodes_of_type), 2), dtype=torch.float32)
            else:
                arity = self.relation_dict[node_type]
                x = torch.empty((len(nodes_of_type), arity + 1), dtype=torch.float32)
                statuses = builder.node_attrs[node_type].get("status", [])
                for i in range(len(nodes_of_type)):
                    status = statuses[i] if i < len(statuses) else None
                    x[i].fill_(status.encode() if status is not None else 0)
            data[node_type].x = x
            data[node_type].node_names = list(nodes_of_type)

        for unused_node_type in self.relation_dict.keys() - nodes_dict.keys():
            arity = self.relation_dict[unused_node_type]
            data[unused_node_type].x = torch.empty((0, arity + 1), dtype=torch.float32)
            data[unused_node_type].node_names = []

        if self.symbol_type_id not in nodes_dict:
            get_logger(__name__).warning(f"No object in graph ({nodes_dict})")
            data[self.symbol_type_id].x = torch.empty((0, 2), dtype=torch.float32)
            data[self.symbol_type_id].node_names = []

        for edge_type, indices in builder.edge_indices.items():
            if indices:
                data[edge_type].edge_index = (
                    torch.tensor(indices, dtype=torch.long).t().contiguous()
                )
            else:
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        for unused_edge_type in set(self.all_edge_types) - set(
            builder.edge_indices.keys()
        ):
            data[unused_edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        return data

    def _new_batch_builder(self) -> ILGBatchBuilder:
        return ILGBatchBuilder(
            relation_dict=self.relation_dict,
            symbol_type_id=self.symbol_type_id,
            edge_types=self.all_edge_types,
        )

    @check_encoded_by_this
    def draw(
        self,
        data: HeteroData,
        *,
        ax=None,
        with_labels: bool = True,
        edge_labels: bool = True,
        node_kwargs: dict | None = None,
        edge_kwargs: dict | None = None,
        layout: dict | None = None,
        node_size: float | None = None,
        node_alpha: float | None = None,
        edge_width: float | None = None,
        edge_alpha: float | None = None,
        label_font_size: float | None = None,
        label_nodes: Iterable[str] | None = None,
        label_node_types: Iterable[str] | None = None,
        label_edges: Iterable[tuple[str, ...]] | None = None,
        symbol_node_scale: float = 1.5,
        non_symbol_linestyle: str | None = "--",
    ):
        import matplotlib.pyplot as plt

        node_kwargs = node_kwargs or {}
        edge_kwargs = edge_kwargs or {}

        if ax is None:
            _, ax = plt.subplots()

        if isinstance(data, HeteroData):
            graph = self.to_networkx(data)
        else:
            graph = data

        pos = layout or nx.spring_layout(graph)

        node_types = [graph.nodes[n]["type"] for n in graph.nodes]
        unique_types = list(dict.fromkeys(node_types))
        if unique_types:
            cmap = plt.get_cmap("tab20_r")
            type_to_color = {
                node_type: cmap(i / max(1, len(unique_types) - 1))
                for i, node_type in enumerate(unique_types)
            }
            node_colors = [type_to_color[node_type] for node_type in node_types]
        else:
            node_colors = "#CCCCCC"

        base_node_kwargs = dict(node_kwargs)
        if node_size is not None:
            base_node_kwargs.setdefault("node_size", node_size)
        if node_alpha is not None:
            base_node_kwargs.setdefault("alpha", node_alpha)

        base_node_size_value = base_node_kwargs.get("node_size")
        if isinstance(base_node_size_value, Sequence) and not isinstance(
            base_node_size_value, (str, bytes)
        ):
            base_node_size_value = (
                base_node_size_value[0] if base_node_size_value else None
            )
        if base_node_size_value is None:
            inferred_size = node_size if node_size is not None else 300
            base_node_kwargs.setdefault("node_size", inferred_size)
            base_node_size_value = inferred_size
        elif not isinstance(base_node_size_value, numbers.Real):
            base_node_size_value = 300

        label_edge_set = None
        if label_edges is not None:
            label_edge_set = {tuple(edge) for edge in label_edges}

        symbol_nodes = [
            node
            for node in graph.nodes
            if graph.nodes[node]["type"] == self.symbol_type_id
        ]
        symbol_set = set(symbol_nodes)
        other_nodes = [node for node in graph.nodes if node not in symbol_set]

        other_kwargs = dict(base_node_kwargs)
        other_kwargs.setdefault("edgecolors", "#444444")
        other_kwargs.setdefault("linewidths", 1.2)

        if other_nodes:
            other_colors = [
                type_to_color[graph.nodes[node]["type"]] for node in other_nodes
            ]
            other_collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=other_nodes,
                node_color=other_colors,
                ax=ax,
                **other_kwargs,
            )
            if (
                non_symbol_linestyle
                and other_collection is not None
                and hasattr(other_collection, "set_linestyle")
            ):
                other_collection.set_linestyle(non_symbol_linestyle)

        if symbol_nodes:
            symbol_colors = [type_to_color[self.symbol_type_id] for _ in symbol_nodes]
            symbol_kwargs = dict(base_node_kwargs)
            if isinstance(base_node_size_value, numbers.Real):
                symbol_kwargs["node_size"] = base_node_size_value * symbol_node_scale
            else:
                symbol_kwargs["node_size"] = 300 * symbol_node_scale
            symbol_kwargs.setdefault("edgecolors", "black")
            symbol_kwargs.setdefault("linewidths", 2.4)
            symbol_collection = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=symbol_nodes,
                node_color=symbol_colors,
                ax=ax,
                **symbol_kwargs,
            )
            if symbol_collection is not None:
                symbol_collection.set_facecolor(symbol_colors)
                symbol_collection.set_edgecolor("black")

        labels_to_draw = {}
        explicit_labels = set(label_nodes or [])
        if label_node_types:
            type_set = {t for t in label_node_types}
            explicit_labels.update(
                node
                for node, data in graph.nodes(data=True)
                if data.get("type") in type_set
            )
        if explicit_labels:
            labels_to_draw = {
                node: node for node in graph.nodes if node in explicit_labels
            }
        elif with_labels:
            labels_to_draw = {node: node for node in graph.nodes}

        if labels_to_draw:
            label_kwargs = {}
            if label_font_size is not None:
                label_kwargs["font_size"] = label_font_size
            nx.draw_networkx_labels(
                graph, pos, labels=labels_to_draw, ax=ax, **label_kwargs
            )

        # Edge coloring by argument position
        edge_attr_name = "position"

        # Split edges into standard (numerical) and LGAN (structural - linegraph)
        standard_edges = []
        standard_colors = []
        lgan_edges = []
        lgan_colors = []

        if graph.is_multigraph():
            all_edge_iter = graph.edges(keys=True, data=True)
        else:
            all_edge_iter = graph.edges(data=True)

        all_edge_attrs = []
        for e_info in all_edge_iter:
            data_dict = e_info[-1]
            p_val = data_dict.get(edge_attr_name)
            if p_val is not None:
                all_edge_attrs.append(p_val)

        unique_positions = list(dict.fromkeys(p for p in all_edge_attrs))
        edge_pos_to_color = {}
        if unique_positions:
            cmap = plt.get_cmap("Dark2")
            edge_pos_to_color = {
                val: cmap(i / max(1, len(unique_positions) - 1))
                for i, val in enumerate(unique_positions)
            }

        # Re-iterate to separate by style
        if graph.is_multigraph():
            all_edge_iter = graph.edges(keys=True, data=True)
        else:
            all_edge_iter = graph.edges(data=True)

        for e_info in all_edge_iter:
            u, v = e_info[0], e_info[1]
            data_dict = e_info[-1]
            p_val = data_dict.get(edge_attr_name)
            color = edge_pos_to_color.get(p_val, "#666666")

            if p_val == self.lgan_nn_edge_pos:
                lgan_edges.append((u, v))
                lgan_colors.append(color)
            else:
                standard_edges.append((u, v))
                standard_colors.append(color)

        if edge_width is not None:
            edge_kwargs.setdefault("width", edge_width)
        if edge_alpha is not None:
            edge_kwargs.setdefault("alpha", edge_alpha)

        # Draw standard edges
        if standard_edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=standard_edges,
                edge_color=standard_colors,
                arrows=graph.is_directed(),
                ax=ax,
                **edge_kwargs,
            )

        # Draw LGAN (linegraph) edges as dashed
        if lgan_edges:
            lg_kwargs = dict(edge_kwargs)
            lg_kwargs["style"] = "dashed"
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=lgan_edges,
                edge_color=lgan_colors,
                arrows=graph.is_directed(),
                ax=ax,
                **lg_kwargs,
            )

        if unique_types:
            from matplotlib.patches import Patch

            legend_handles = [
                Patch(
                    facecolor=type_to_color[node_type],
                    edgecolor="none",
                    label=str(node_type),
                )
                for node_type in unique_types
            ]
            node_title = "Node Types"
            if self.include_lgan_edges:
                node_title += "\n(Relations = Linegraph Nodes)"
            node_legend = ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
                title=node_title,
            )
            ax.add_artist(node_legend)

        if unique_positions:
            from matplotlib.lines import Line2D

            edge_handles = [
                Line2D(
                    [0],
                    [0],
                    color=edge_pos_to_color[p_val],
                    linestyle="dashed" if p_val == self.lgan_nn_edge_pos else "solid",
                    label=f"pos: {p_val}"
                    if p_val != self.lgan_nn_edge_pos
                    else "LGAN (NN)",
                )
                for p_val in unique_positions
            ]
            ax.legend(
                handles=edge_handles,
                loc="lower left",
                bbox_to_anchor=(1.02, 0.0),
                frameon=False,
                title="Edge Roles",
            )

        ax.figure.subplots_adjust(right=0.8)

        draw_edge_labels = edge_labels or label_edges is not None
        if draw_edge_labels and unique_positions:
            if graph.is_multigraph():
                labels = {}
                for u, v, key, data in graph.edges(keys=True, data=True):
                    if (
                        label_edge_set is not None
                        and (u, v, key) not in label_edge_set
                        and (u, v) not in label_edge_set
                    ):
                        continue
                    labels[(u, v, key)] = data.get("position")
            else:
                labels = {}
                for u, v, data in graph.edges(data=True):
                    if label_edge_set is not None and (u, v) not in label_edge_set:
                        continue
                    labels[(u, v)] = data.get("position")
            labels = {
                edge: value for edge, value in labels.items() if value is not None
            }
            if labels:
                label_kwargs = {}
                if label_font_size is not None:
                    label_kwargs["font_size"] = label_font_size
                nx.draw_networkx_edge_labels(
                    graph,
                    pos,
                    edge_labels=labels,
                    font_color="black",
                    ax=ax,
                    **label_kwargs,
                )

        ax.set_axis_off()
        return ax

    def to_networkx(self, data: HeteroData) -> nx.MultiGraph:
        """Alias for from_pyg_data for consistency with base class."""
        return self.from_pyg_data(data)

    def from_pyg_data(self, data: HeteroData) -> nx.MultiGraph:
        """
        Reconstruct a NetworkX graph from `HeteroData` produced by this encoder.
        """
        graph = self._graph_t()

        node_names_by_type: Dict[NodeType, List[str]] = {}
        # 1. Add all nodes
        for ntype in data.node_types:
            node_names = getattr(data[ntype], "node_names", None)
            if node_names is None:
                num_nodes = data[ntype].num_nodes
                node_names = [f"{ntype}_{i}" for i in range(num_nodes)]
            node_names_by_type[ntype] = list(node_names)

            # Collect other attributes
            attr_keys = [
                k
                for k in data[ntype].keys()
                if k not in ("x", "edge_index", "node_names")
            ]

            for i, name in enumerate(node_names):
                node_attrs = {"type": ntype}
                for k in attr_keys:
                    val = data[ntype][k]
                    if torch.is_tensor(val):
                        node_attrs[k] = val[i].item()
                    else:
                        node_attrs[k] = val[i]

                # Special handling for AtomStatus in ILG
                if ntype != self.symbol_type_id and ntype != self.action_type_id:
                    status_val = (
                        data[ntype].x[i, 0].item()
                        if data[ntype].x.dim() > 1
                        else data[ntype].x[i].item()
                    )
                    node_attrs["status"] = AtomStatus.decode(status_val)

                graph.add_node(name, **node_attrs)

        # 2. Add all edges
        for edge_type in data.edge_types:
            src_type, rel, dst_type = edge_type
            if not rel.isdigit() and rel != self.lgan_nn_edge_pos:
                continue

            edge_index = data[edge_type].edge_index
            src_names = node_names_by_type.get(src_type, [])
            dst_names = node_names_by_type.get(dst_type, [])

            for i in range(edge_index.shape[1]):
                u_idx = edge_index[0, i].item()
                v_idx = edge_index[1, i].item()
                u_name = src_names[u_idx]
                v_name = dst_names[v_idx]

                # For undirected MultiGraph, only add one edge for bidirectional pairs
                if u_name < v_name:
                    graph.add_edge(
                        u_name, v_name, position=int(rel) if rel.isdigit() else rel
                    )
                elif u_name == v_name:  # self-loop
                    graph.add_edge(
                        u_name, v_name, position=int(rel) if rel.isdigit() else rel
                    )

        return graph
