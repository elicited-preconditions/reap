# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from __future__ import annotations

import numbers
from collections import defaultdict
from itertools import chain
from typing import Collection, Iterable, List, NamedTuple, Sequence

import networkx as nx
import torch
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType

from reap.logging_setup import get_logger
from reap.xmimir import (
    XAction,
    XActionSchema,
    XAtom,
    XDomain,
    XLiteral,
    XObject,
    XPredicate,
    atom_str_template,
)

from .base_encoder import (
    EncoderFactory,
    GraphEncoderBase,
    check_encoded_by_this,
)
from .pyg_batch_builder import HGraphBatchBuilder
from .pyg_builder import PygBuilderBase, PygHeteroBuilder
from .relation_dict import RelationDict
from .relation_formatter import Node, relation_formatter


class PredicateEdgeType(NamedTuple):
    src_type: str
    pos: str
    dst_type: str


class AdHocRelation(NamedTuple):
    name: str
    arity: int


class HGraphEncoder(GraphEncoderBase[HeteroData]):
    """
    A heterogeneous, bipartite-style encoder for symbolic planning states. It represents
    stable, identity-bearing **symbol** nodes (type given by ``symbol_type_id``) and
    contextual **relation** nodes (atoms, literals, predicates, goal-literals, (optionally) actions, ...).
    For each grounded atom/literal/action, it creates a relation node whose node *type*
    encodes the base predicate or schema (via ``relation_formatter``). Undirected edges
    connect symbol nodes to relation nodes, labeled by the argument position (``position ∈ {0, …, arity-1}``).

    The encoder can (optionally) handle nullary relations by introducing a synthetic
    symbol named ``nullary_object_name`` so that message passing remains uniform. It also
    supports goal-satisfaction derivations (e.g., satisfied/unsatisfied, added/removed)
    through the provided ``RelationDict`` configuration.

    Parameters
    -----------
    domain : XDomain
        The symbolic domain that provides predicates, action schemas, and canonical
        formatting utilities.
    relation_dict : RelationDict, optional
        Mapping from relation types (as produced by ``relation_formatter(predicate, ...)``) to their arity.
        If not provided, a default is constructed. It is advised to let the encoder create it, so that
        all implicit assumptions are aligned.
    symbol_type_id : str, defaulted to "_symbol_"
        Node-type string used for all symbol nodes in the heterogeneous graph and in the
        returned ``HeteroData``.
    ignore_actions : bool, default True
        If ``False``, action instances are encoded as relation nodes, connected to their
        object arguments plus an auxiliary symbol node at position ``0``.
    nullary_object_name : str, default ``relation_formatter.default_nullary_object_name``
        Auxiliary symbol name used when ``add_nullary_predicates`` is ``True`` and a relation has arity ``0``.
    add_nullary_predicates : bool, default False
        If ``True``, include a dedicated symbol node named ``nullary_object_name`` and
        connect nullary relations to it at position ``0`` so the edge pattern is
        consistent across arities.
    **relation_dict_kwargs
        Extra options forwarded to ``RelationDict``
    """

    def __init__(
        self,
        domain: XDomain,
        *,
        relation_dict: RelationDict | None = None,
        symbol_type_id: str = "_symbol_",
        ignore_actions: bool = True,
        nullary_object_name: str = relation_formatter.default_nullary_symbol_name,
        add_nullary_predicates: bool = False,
        include_lgan_edges: bool = False,
        lgan_nn_edge_pos: str = "lgan_nn",
        **relation_dict_kwargs,
    ) -> None:
        super().__init__(domain, builder=PygHeteroBuilder(), graph_t=nx.MultiGraph)
        self.symbol_type_id: str = symbol_type_id
        self.ignore_actions = ignore_actions
        # Initialize the default here to please jsonargparse.
        self.add_nullary_predicates = add_nullary_predicates
        self.include_lgan_edges = include_lgan_edges
        self.lgan_nn_edge_pos = lgan_nn_edge_pos
        self.nullary_object_name = nullary_object_name
        self.predicates: tuple[XPredicate, ...] = self.domain.predicates()
        self.actions: tuple[XActionSchema, ...] = (
            self.domain.actions if not ignore_actions else ()
        )
        self.relation_dict: RelationDict = relation_dict or RelationDict(
            self.predicates,
            # +1 to account for the action node connecting to its aux symbol node
            tuple(
                AdHocRelation(action.name, action.arity + 1) for action in self.actions
            ),
            goal_satisfaction_derivations=relation_dict_kwargs.pop(
                "goal_satisfaction_derivations", (True,)
            ),
            **relation_dict_kwargs,
        )
        # Generate all possible edge types
        all_edge_types: List[EdgeType] = []
        for predicate_node, arity in self.relation_dict.items():
            sender = self.symbol_type_id
            effective_arity = (
                1 if (self.add_nullary_predicates and arity == 0) else arity
            )
            for pos in range(effective_arity):
                fwd_edge = PredicateEdgeType(sender, str(pos), predicate_node)
                rev_edge = PredicateEdgeType(predicate_node, str(pos), sender)
                all_edge_types.append(fwd_edge)
                all_edge_types.append(rev_edge)
        if self.include_lgan_edges:
            # LGAN NN edges connect relation to symbol nodes
            all_edge_types.append(
                (self.lgan_nn_edge_pos, self.lgan_nn_edge_pos, self.symbol_type_id)
            )
        self.all_edge_types = all_edge_types
        self._ensure_config_hash()

    def _config_dict(self) -> dict:
        return {
            "symbol_type_id": self.symbol_type_id,
            "ignore_actions": self.ignore_actions,
            "nullary_object_name": self.nullary_object_name,
            "add_nullary_predicates": self.add_nullary_predicates,
            "include_lgan_edges": self.include_lgan_edges,
            "lgan_nn_edge_pos": self.lgan_nn_edge_pos,
            "relation_dict": self.relation_dict.__repr__(),
        }

    def __eq__(self, other: HGraphEncoder) -> bool:
        return (
            self.symbol_type_id == other.symbol_type_id
            and self.predicates == other.predicates
            and self.relation_dict == other.relation_dict
            and self.include_lgan_edges == other.include_lgan_edges
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["predicates"]  # predicates cannot be pickled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.predicates = self.domain.predicates()

    def as_factory(self) -> EncoderFactory:
        return EncoderFactory(
            encoder_class=self.__class__,
            kwargs={
                "symbol_type_id": self.symbol_type_id,
                "add_nullary_predicates": self.add_nullary_predicates,
                "nullary_object_name": self.nullary_object_name,
                "relation_dict": self.relation_dict,
                "ignore_actions": self.ignore_actions,
                "include_lgan_edges": self.include_lgan_edges,
                "lgan_nn_edge_pos": self.lgan_nn_edge_pos,
            },
        )

    def _encode(
        self,
        builder: PygBuilderBase,
        facts: Sequence[XAtom],
        goals: Sequence[XLiteral],
        actions: Sequence[XAction],
        objects: Sequence[XObject] | None = None,
        **kwargs,
    ) -> None:
        # Build hetero graph from state
        # One node for each object
        # One node for each atom
        # Edge label = position in atom
        glm = builder.graph_attrs["goal_level_map"]
        objects = objects or self._contained_objects(
            chain(facts, goals, actions if not self.ignore_actions else ())
        )
        self._encode_objects(objects, builder)
        self._encode_facts(facts, builder)
        self._encode_literals(goals, builder, goal_level_map=glm)
        if not self.ignore_actions:
            for action in actions:
                self._encode_action(action, builder)
        if self.relation_dict.goal_satisfaction_derivations and goals:
            self._encode_goal_satisfaction(
                self._goal_satisfaction_map(facts, goals),
                builder,
                goal_level_map=glm,
            )
        if self.include_lgan_edges:
            self._add_lgan_nn_edges(builder)
        builder.set_graph_attr("action", [False]*len(objects) + [True]*len(actions))
        # Per-action ARGUMENT indices into the symbol rows, for the "args"
        # readout mode (ActionPoolingModule).  Row order matches the True
        # entries of the "action" mask above; indices are LOCAL to this graph
        # (the pooling module adds per-graph offsets after batching).  Width is
        # fixed at 2 (pick/put arity); if any action deviates the attribute is
        # withheld entirely so the "args" mode fails loudly instead of
        # gathering garbage.
        if not self.ignore_actions:
            obj_pos = {relation_formatter(o): i for i, o in enumerate(objects)}
            arg_rows = []
            for action in actions:
                idxs = [obj_pos.get(relation_formatter(o)) for o in action.objects]
                if len(idxs) != 2 or any(i is None for i in idxs):
                    arg_rows = None
                    break
                arg_rows.append(idxs)
            if arg_rows is not None:
                builder.set_graph_attr("action_args", arg_rows)

    def _encode_objects(self, objects: list[XObject], builder: PygBuilderBase):
        for obj in objects:
            builder.add_node(
                relation_formatter(obj), self.symbol_type_id, name=obj.name
            )
        if self.add_nullary_predicates:
            builder.add_node(
                self.nullary_object_name,
                self.symbol_type_id,
                name=self.nullary_object_name,
            )

    def _encode_goal_satisfaction(
        self,
        satisfied_goals: dict[XLiteral, bool | str | None],
        builder: PygBuilderBase,
        node_prefix: str = "",
        extra_objects: Iterable[XObject | str] = (),
        goal_level_map: dict[XLiteral, int] | None = None,
    ):
        goal_level_map = (
            goal_level_map
            if goal_level_map is not None
            else builder.graph_attrs["goal_level_map"]
        )
        supported_satisfaction_derivs = self.relation_dict.goal_satisfaction_derivations
        for goal, satisfaction_level in satisfied_goals.items():
            if satisfaction_level not in supported_satisfaction_derivs:
                continue
            polarity = goal.polarity
            atom = goal.atom
            if atom.predicate.arity == 0:
                if not self.add_nullary_predicates:
                    continue
                objects = [self.nullary_object_name]
            else:
                objects = list(atom.objects)
            try:
                goal_level = goal_level_map[goal]
            except KeyError:
                raise KeyError(
                    f"{goal} is not present in goal level map: {goal_level_map}."
                )

            literal_node = relation_formatter(
                goal,
                goal_level=goal_level,
                goal_satisfaction=satisfaction_level,
                polarity=polarity,
            )
            literal_node = node_prefix + literal_node
            ntype = relation_formatter(
                goal.atom.predicate,
                goal_level=goal_level,
                goal_satisfaction=satisfaction_level,
                polarity=polarity,
            )
            builder.add_node(literal_node, ntype)
            for pos, obj in enumerate(chain(extra_objects, objects)):
                # Forward: object -> satisfied goal node
                builder.add_edge(
                    relation_formatter(obj),
                    literal_node,
                    self.symbol_type_id,
                    ntype,
                    str(pos),
                )
                # Reverse: satisfied goal node -> object
                builder.add_edge(
                    literal_node,
                    relation_formatter(obj),
                    ntype,
                    self.symbol_type_id,
                    str(pos),
                )

    def _encode_action(
        self,
        action: XAction,
        builder: PygBuilderBase,
        node_prefix: str = "",
        extra_objects: Iterable[XObject | str] | None = None,
    ):
        node = node_prefix + relation_formatter(action)
        ntype = relation_formatter(action.action_schema)
        builder.add_node(node, ntype, name=action.name)
        if extra_objects is None:
            # action symbol node is an auxiliary for embedding the action in latent space
            # (typically reserved for objects).
            action_symbol_node = f"target:{action.index}|" + node
            builder.add_node(action_symbol_node, self.symbol_type_id, name=action.name)
            # ensure the action node connects both to its dedicated symbol and the provided context nodes
            extra_objects = [action_symbol_node]
        for pos, obj in enumerate(chain(extra_objects, action.objects)):
            # Forward: object -> action node
            builder.add_edge(
                relation_formatter(obj),
                node,
                self.symbol_type_id,
                ntype,
                str(pos),
            )
            # Reverse: action node -> object
            builder.add_edge(
                node,
                relation_formatter(obj),
                ntype,
                self.symbol_type_id,
                str(pos),
            )

    def _encode_literals(
        self,
        literals: Iterable[XLiteral],
        builder: PygBuilderBase,
        node_prefix: str = "",
        extra_objects: Iterable[XObject | str] = (),
        goal_level_map: dict[XLiteral, int] | None = None,
    ):
        goal_level_map = (
            goal_level_map
            if goal_level_map is not None
            else builder.graph_attrs["goal_level_map"]
        )
        for literal in literals:
            atom = literal.atom
            predicate = atom.predicate
            arity = predicate.arity
            if arity == 0:
                if not self.add_nullary_predicates:
                    continue
                objects = [self.nullary_object_name]
            else:
                objects = list(atom.objects)

            goal_level = goal_level_map.get(literal, None)
            node = node_prefix + relation_formatter(
                literal, goal_level=goal_level, polarity=literal.polarity
            )
            ntype = relation_formatter(
                predicate, goal_level=goal_level, polarity=literal.polarity
            )
            builder.add_node(node, ntype)
            for pos, obj in enumerate(chain(extra_objects, objects)):
                # Connect literal node to object node
                builder.add_edge(
                    relation_formatter(obj),
                    node,
                    self.symbol_type_id,
                    ntype,
                    str(pos),
                )
                builder.add_edge(
                    node,
                    relation_formatter(obj),
                    ntype,
                    self.symbol_type_id,
                    str(pos),
                )

    def _encode_facts(
        self,
        facts: Iterable[XAtom],
        builder: PygBuilderBase,
        node_prefix: str = "",
        extra_objects: Iterable[XObject | str] = (),
    ):
        for atom in facts:
            predicate = atom.predicate
            arity = predicate.arity
            if arity == 0:
                if not self.add_nullary_predicates:
                    continue
                objects = [self.nullary_object_name]
            else:
                objects = list(atom.objects)

            node = node_prefix + relation_formatter(atom)
            ntype = relation_formatter(predicate)
            builder.add_node(node, ntype)
            for pos, obj in enumerate(chain(extra_objects, objects)):
                # Connect atom node to object node
                builder.add_edge(
                    relation_formatter(obj),
                    node,
                    self.symbol_type_id,
                    ntype,
                    str(pos),
                )
                builder.add_edge(
                    node,
                    relation_formatter(obj),
                    ntype,
                    self.symbol_type_id,
                    str(pos),
                )

    def _add_lgan_nn_edges(self, builder: PygBuilderBase):
        """
        Add virtual edges representing neighbor-neighbor (NN) relationships.
        """
        # 1. Identify relation and symbol nodes
        relation_node_types = [
            nt for nt in builder.node_indices if nt != self.symbol_type_id
        ]
        symbol_node_type = self.symbol_type_id

        # Maps relation_node (type, key) -> set of neighbor symbol keys
        rel_to_symbols = defaultdict(set)
        # Maps symbol_node key -> set of neighbor relation nodes (type, key)
        symbol_to_rels = defaultdict(set)

        for (src_type, edge_type, dst_type), edges in builder.edge_indices.items():
            # Filter for standard edges (positional)
            if not edge_type.isdigit():
                continue

            if src_type == symbol_node_type and dst_type in relation_node_types:
                for s_idx, r_idx in edges:
                    s_key = builder.node_keys[src_type][s_idx]
                    r_key = builder.node_keys[dst_type][r_idx]
                    rel_to_symbols[(dst_type, r_key)].add(s_key)
                    symbol_to_rels[s_key].add((dst_type, r_key))
            elif src_type in relation_node_types and dst_type == symbol_node_type:
                for r_idx, s_idx in edges:
                    s_key = builder.node_keys[dst_type][s_idx]
                    r_key = builder.node_keys[src_type][r_idx]
                    rel_to_symbols[(src_type, r_key)].add(s_key)
                    symbol_to_rels[s_key].add((src_type, r_key))

        # 2. For each symbol Target, compute TN(Target)
        target_to_tn_symbols = {}
        symbol_keys = builder.node_keys[symbol_node_type]
        for target_key in symbol_keys:
            tn = {target_key}
            for rel_info in symbol_to_rels[target_key]:
                tn.update(rel_to_symbols[rel_info])
            target_to_tn_symbols[target_key] = tn

        # 3. For each relation node, check which Target it connects to via LGAN
        for rel_info, arg_set in rel_to_symbols.items():
            if not arg_set:
                continue
            ntype, r_key = rel_info
            for target_key, tn_symbols in target_to_tn_symbols.items():
                if target_key in arg_set:
                    continue
                if arg_set.issubset(tn_symbols):
                    # Add virtual edge
                    builder.add_edge(
                        r_key,
                        target_key,
                        ntype,
                        symbol_node_type,
                        self.lgan_nn_edge_pos,
                    )
                    builder.add_edge(
                        target_key,
                        r_key,
                        symbol_node_type,
                        ntype,
                        self.lgan_nn_edge_pos,
                    )

    @staticmethod
    def _goal_satisfaction_map(facts: Collection[XAtom], goals: Iterable[XLiteral]):
        return {
            goal: any(goal.atom.semantic_eq(fact) for fact in facts) == goal.polarity
            for goal in goals
        }

    def _build_data(self, builder: PygBuilderBase) -> HeteroData:
        nodes_dict = builder.node_keys
        data = HeteroData()

        object_names = []
        if self.symbol_type_id in builder.node_attrs:
            object_names = list(builder.node_attrs[self.symbol_type_id].get("name", []))
        if not object_names and self.symbol_type_id in nodes_dict:
            object_names = list(nodes_dict[self.symbol_type_id])
        data.object_names = object_names

        # encode actions
        for k, v in builder.graph_attrs.items():
            if "goal_level_map" != k:
                setattr(data, k, v)


        for node_type, nodes_of_type in nodes_dict.items():
            size = (
                1 if node_type == self.symbol_type_id else self.relation_dict[node_type]
            )
            data[node_type].x = torch.zeros(
                (len(nodes_of_type), size), dtype=torch.float32
            )
            data[node_type].node_names = list(nodes_of_type)

        for unused_node_type in self.relation_dict.keys() - nodes_dict.keys():
            size = (
                1
                if unused_node_type == self.symbol_type_id
                else self.relation_dict[unused_node_type]
            )
            data[unused_node_type].x = torch.empty((0, size), dtype=torch.float32)
            data[unused_node_type].node_names = []

        if self.symbol_type_id not in nodes_dict:
            get_logger(__name__).warning(f"No symbol in graph ({nodes_dict})")
            data[self.symbol_type_id].x = torch.empty((0, 1), dtype=torch.float32)
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

    def _new_batch_builder(self) -> HGraphBatchBuilder:
        return HGraphBatchBuilder(
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

        graph = data if isinstance(data, nx.Graph) else self.to_networkx(data)
        pos = layout or nx.spring_layout(graph)

        node_types = [graph.nodes[n]["type"] for n in graph.nodes]
        unique_types = list(dict.fromkeys(node_types))
        if unique_types:
            cmap = plt.get_cmap("tab20_r")
            type_to_color = {
                ntype: cmap(i / max(1, len(unique_types) - 1))
                for i, ntype in enumerate(unique_types)
            }

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

        all_positions = []
        for e_info in all_edge_iter:
            data_dict = e_info[-1]
            p_val = data_dict.get(edge_attr_name)
            if p_val is not None:
                all_positions.append(p_val)

        unique_positions = list(dict.fromkeys(p for p in all_positions))
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
                    facecolor=type_to_color[ntype], edgecolor="none", label=str(ntype)
                )
                for ntype in unique_types
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
                for u, v, k, data in graph.edges(keys=True, data=True):
                    if (
                        label_edge_set is not None
                        and (u, v, k) not in label_edge_set
                        and (u, v) not in label_edge_set
                    ):
                        continue
                    labels[(u, v, k)] = data.get(edge_attr_name)
            else:
                labels = {}
                for u, v, data in graph.edges(data=True):
                    if label_edge_set is not None and (u, v) not in label_edge_set:
                        continue
                    labels[(u, v)] = data.get(edge_attr_name)
            labels = {
                edge: value for edge, value in labels.items() if value is not None
            }
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
        """
        Reconstruct a graph from a HeteroData object.
        """
        graph = self._graph_t()

        # 1. Add all nodes
        for ntype in data.node_types:
            node_names = getattr(data[ntype], "node_names", None)
            if node_names is None:
                # Fallback to indexed names if node_names missing
                num_nodes = data[ntype].x.shape[0]
                node_names = [f"{ntype}_{i}" for i in range(num_nodes)]

            # Add nodes with their type
            for name in node_names:
                graph.add_node(name, type=ntype)

            # Collect other attributes and add them to the already created nodes
            for attr_name, attr_value in data[ntype].items():
                if attr_name in ("x", "node_names", "edge_index", "edge_attr"):
                    continue
                # attr_value is a tensor or list [num_nodes, ...]
                for i in range(len(node_names)):
                    if i >= len(attr_value):
                        continue
                    val = attr_value[i]
                    if torch.is_tensor(val):
                        val = val.item()
                    graph.nodes[node_names[i]][attr_name] = val

        # 2. Add all edges
        for edge_type in data.edge_types:
            src_type, rel, dst_type = edge_type
            edge_index = data[edge_type].edge_index

            for i in range(edge_index.shape[1]):
                u_idx = edge_index[0, i].item()
                v_idx = edge_index[1, i].item()

                # Retrieve names using same logic as node creation
                def _get_names(t):
                    if hasattr(data[t], "node_names"):
                        return data[t].node_names
                    num = data[t].x.shape[0]
                    return [f"{t}_{j}" for j in range(num)]

                src_names = _get_names(src_type)
                dst_names = _get_names(dst_type)

                u_name = src_names[u_idx]
                v_name = dst_names[v_idx]

                # For undirected graphs, only add one edge for bidirectional pairs.
                if not graph.is_directed():
                    if src_type == dst_type:
                        if u_name < v_name:
                            continue
                    else:
                        # For bipartite-like relations, only skip if the reverse type also exists
                        # in the PyG data, to avoid duplicates while preserving
                        # edges that were only added in one direction.
                        inverse_type = (dst_type, rel, src_type)
                        if inverse_type in data.edge_types:
                            if src_type < dst_type:
                                pass
                            else:
                                continue

                edge_attrs = {}
                if "edge_attr" in data[edge_type]:
                    # handle edge attributes if present
                    val = data[edge_type].edge_attr[i]
                    if torch.is_tensor(val):
                        val = val.item()
                    edge_attrs["position"] = val
                else:
                    edge_attrs["position"] = int(rel) if rel.isdigit() else rel

                graph.add_edge(u_name, v_name, **edge_attrs)

        return graph

    def from_pyg_data(self, data: HeteroData) -> nx.MultiGraph:
        """Reconstruct a MultiGraph from HGraphEncoder data."""
        return self.to_networkx(data)
