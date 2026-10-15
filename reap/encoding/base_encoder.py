# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import warnings
from abc import ABC, abstractmethod
from copy import copy
from itertools import chain
from typing import (
    Generic,
    Iterable,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    get_args,
)

import networkx as nx
import torch_geometric as pyg
from torch_geometric.data import Batch

from reap.xmimir import XAction, XAtom, XDomain, XLiteral, XObject, XState, gather_objects

from .pyg_batch_builder import BatchBuilderBase
from .pyg_builder import PygBuilderBase

PygDataT = TypeVar("PygDataT", pyg.data.Data, pyg.data.HeteroData)
PygData = Union[pyg.data.Data, pyg.data.HeteroData]


class GraphEncoderBase(Generic[PygDataT], ABC):
    """
    Base class for encoders that map planning states to PyG data objects.

    Subclasses must specify the concrete PyG data type via `GraphEncoderBase[pyg.data.Data]` and implement `_encode`.
    """

    @property
    def pyg_data_t(self) -> Type[PygDataT]:
        """Return the concrete PyG data type this encoder produces."""
        # This assumes the first generic argument used in subclassing is the data type.
        try:
            return get_args(self.__class__.__orig_bases__[0])[0]
        except (AttributeError, IndexError) as e:
            raise NotImplementedError(
                f"Could not extract specific PyG data type from generic inheritance. "
                f"Remember to explicate the data type when subclassing {GraphEncoderBase.__name__} as e.g. such "
                f"`{GraphEncoderBase.__name__}[pyg.data.HeteroData].",
            ) from e

    @property
    def domain(self) -> XDomain:
        """Return the domain associated with this encoder."""
        return self._domain

    def __init__(
        self,
        domain: XDomain,
        builder: PygBuilderBase,
        graph_t: Type[nx.Graph] = nx.MultiGraph,
        *args,
        **kwargs,
    ):
        self._domain = domain
        self._builder = builder
        self._graph_t = graph_t
        self._watermark: str | None = None

    def encode(
        self,
        state: XState | Iterable[XAtom],
        goals: Iterable[XLiteral] | None = None,
        actions: Iterable[XAction] | None = None,
        *,
        subgoal_layers: Sequence[Iterable[XLiteral]] | None = None,
        **kwargs,
    ) -> PygDataT:
        """
        Encode a state (or explicit atoms/literals) into a PyG data object.

        Parameters
        ----------
        state : XState | Iterable[XAtom]
            Either an `XState` or an iterable of atoms.
        goals : Iterable[XLiteral] | None, optional
            If provided, these goal literals will be used as the goal to encode for a given complete state, defaulting
            to the problem-defined goal, otherwise.
        actions : Iterable[XAction] | None, optional
            Grounded actions to consider for the encoding. If None, no actions will be encoded.
        subgoal_layers : Sequence[Iterable[XLiteral]] | None, optional
            Hierarchical subgoals as a sequence of layers [depth1, depth2, ...].
            Each inner iterable is a set of literals for that depth. These are **always added**
            on top of the main goals, regardless of `goal_mode`.

        Returns
        -------
        PygDataT
            The encoded PyG data object.
        """
        if isinstance(state, (pyg.data.Data, pyg.data.HeteroData)):
            if not self._encoded_by_this(state):
                raise ValueError("Data must have been encoded by this encoder")
            return state

        self._encode_to_builder(
            self._builder,
            state,
            goals,
            actions,
            subgoal_layers=subgoal_layers,
            **kwargs,
        )
        data = self._build_data(self._builder)
        if getattr(data, "encoder_hash", None) is None:
            data.encoder_hash = self._watermark
        return data

    def _encode_to_builder(
        self,
        builder: PygBuilderBase,
        state: XState | Iterable[XAtom],
        goals: Iterable[XLiteral] | None = None,
        actions: Iterable[XAction] | None = None,
        *,
        subgoal_layers: Sequence[Iterable[XLiteral]] | None = None,
        **kwargs,
    ) -> None:
        self._ensure_config_hash()
        builder.reset()
        builder.encoder_hash = self._watermark

        objects = None
        if isinstance(state, XState):
            facts = list(state.atoms(with_statics=True))
            objects = state.problem.objects + state.problem.domain.constants
            if goals is None:
                top_goals = state.problem.goal()
            else:
                top_goals = list(goals)
        else:
            facts = list(state)
            top_goals = list(goals) if goals is not None else []

        subgoal_layers_dict: dict[XLiteral, int] = {}  # maps literal -> depth
        goals_list = []
        for depth, subgoals in enumerate(chain([top_goals], subgoal_layers or ())):
            goals_list.extend(subgoals)
            for subgoal in subgoals:
                subgoal_layers_dict[subgoal] = depth

        builder.set_graph_attr("goal_level_map", subgoal_layers_dict)

        self._encode(
            builder=builder,
            facts=facts,
            goals=goals_list,
            actions=actions or (),
            objects=objects,
            _state=state,
            **kwargs,
        )

    def _new_builder(self) -> PygBuilderBase:
        return copy(self._builder)

    @abstractmethod
    def _new_batch_builder(self) -> BatchBuilderBase: ...

    @abstractmethod
    def _encode(
        self,
        builder: PygBuilderBase,
        facts: Sequence[XAtom],
        goals: Sequence[XLiteral],
        actions: Sequence[XAction],
        **kwargs,
    ): ...

    def _build_data(self, builder: PygBuilderBase) -> PygDataT:
        return builder.build()

    def encode_batch(
        self,
        inputs: Sequence[dict],
        *,
        batch_builder: BatchBuilderBase | None = None,
        **kwargs,
    ) -> Batch:
        """
        Encodes a list of inputs into a single PyG Batch.

        Provide an optional batch_builder to append multiple calls into one batch.
        When given, the same builder can be reused across multiple encode_batch calls.
        """
        batch_builder = batch_builder or self._new_batch_builder()
        builder = self._new_builder()
        for item in inputs:
            self._encode_to_builder(builder, **item, **kwargs)
            batch_builder.append(builder)
        batch = batch_builder.build()
        if getattr(batch, "encoder_hash", None) is None:
            batch.encoder_hash = [self._watermark] * batch.num_graphs
        return batch

    def to_networkx(self, data: PygDataT) -> nx.Graph:
        """Reconstructs a NetworkX graph from PyG data for visualization."""
        raise NotImplementedError

    @abstractmethod
    def draw(
        self,
        data: PygDataT,
        *,
        ax=None,
        with_labels: bool = True,
        edge_labels: bool = True,
        node_size: float | None = None,
        node_alpha: float | None = None,
        edge_width: float | None = None,
        edge_alpha: float | None = None,
        label_font_size: float | None = None,
        label_nodes: Iterable[str] | None = None,
        label_node_types: Iterable[str] | None = None,
        label_edges: Iterable[tuple[str, ...]] | None = None,
        **kwargs,
    ):
        """
        Render an encoded state graph with Matplotlib.

        Parameters
        ----------
        data : PygDataT
            A PyG data object previously produced by this encoder.
        ax : matplotlib.axes.Axes | None, optional
            Target axes. If omitted, a new figure and axes are created.
        with_labels : bool, default True
            Whether to annotate nodes with their identifiers.
        edge_labels : bool, default True
            Whether to annotate edges with their ``position`` attribute.
        node_size : float | None, optional
            Override the default node size passed to NetworkX.
        node_alpha : float | None, optional
            Override the transparency (alpha) applied to nodes.
        edge_width : float | None, optional
            Override the default edge width passed to NetworkX.
        edge_alpha : float | None, optional
            Override the transparency (alpha) applied to edges.
        label_font_size : float | None, optional
            Override font size for node/edge labels.
        label_nodes : Iterable[str] | None, optional
            If provided, draw labels only for the specified nodes (regardless of ``with_labels``).
        label_node_types : Iterable[str] | None, optional
            If provided, label every node whose ``type`` attribute matches one of the supplied strings.
        label_edges : Iterable[tuple[str, ...]] | None, optional
            If provided, draw edge labels only for those edges. Accepts 2-tuples (u, v) and, for multigraphs,
            optional 3-tuples (u, v, key).
        **kwargs
            Encoder-specific rendering arguments.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the graph was drawn on.
        """
        ...

    @staticmethod
    def _contained_objects(
        items: Iterable[XAtom | XLiteral | XAction],
    ) -> list[XObject]:
        """
        Collect and lexicographically sort all objects appearing in the given atoms/literals.
        """
        objs = sorted(
            gather_objects(
                [item.atom if isinstance(item, XLiteral) else item for item in items]
            ),
            key=lambda obj: obj.name,
        )
        if len(objs) == 0:
            warnings.warn(
                "Received empty objects sequence. Check if any items have been passed at all."
            )
            return []
        return objs

    def _remove_encoding_artifacts(self, graph: GraphT):
        # In the pyg.utils.from_networkx the graph is converted to a DiGraph
        # In this process it has to be pickled, which is not possible for pymimir objects
        graph_copy = graph.copy()
        for attr in ("state", "goal_level_map"):
            if attr in graph_copy.graph:
                del graph_copy.graph[attr]
        return graph_copy

    def _encoded_by_this(self, data: PygDataT | nx.Graph) -> bool:
        self._ensure_config_hash()
        dhash = getattr(data, "encoder_hash", None)
        if dhash is None and isinstance(data, nx.Graph):
            dhash = data.graph.get("encoder_hash")
        return dhash == self._watermark

    def _build_encoder_watermark(self) -> str:
        data = self._config_payload()
        payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _config_payload(self) -> dict:
        payload = {
            "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
        }
        payload.update(self._config_dict())
        return payload

    def _config_dict(self) -> dict:
        return {}

    @property
    def encoder_hash(self) -> str | None:
        return self._watermark

    def _ensure_config_hash(self) -> None:
        if self._watermark is None:
            self._watermark = self._build_encoder_watermark()

    def as_factory(self) -> EncoderFactory:
        """
        Return a lightweight factory that recreates this encoder without copying the domain.

        Useful for multiprocessing or (de)serialization of encoder configuration.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `as_factory` method."
        )


def check_encoded_by_this(func):
    @functools.wraps(func)
    def wrapper(self, data, *args, **kwargs):
        if not self._encoded_by_this(data):
            raise ValueError("Data must have been encoded by this encoder")
        return func(self, data, *args, **kwargs)

    return wrapper


class EncoderFactory:
    def __init__(
        self, encoder_class: Type[GraphEncoderBase], kwargs: Optional[dict] = None
    ):
        self.encoder_class = encoder_class
        signature = inspect.signature(encoder_class.__init__)
        actual_kwargs = dict()
        for kwarg in kwargs or dict():
            if kwarg not in ("self", "domain") and kwarg in signature.parameters.keys():
                param = signature.parameters[kwarg]
                if (
                    param.default is not inspect.Parameter.empty
                    and kwargs[kwarg] != param.default
                ):
                    actual_kwargs[kwarg] = kwargs[kwarg]
        self.kwargs = actual_kwargs

    def __eq__(self, other):
        if not isinstance(other, EncoderFactory):
            return NotImplemented
        return self.encoder_class == other.encoder_class and self.kwargs == other.kwargs

    def __call__(self, domain: XDomain) -> GraphEncoderBase:
        return self.encoder_class(domain, **self.kwargs)

    def __str__(self):
        return f"{self.encoder_class.__name__}({self.kwargs})"
