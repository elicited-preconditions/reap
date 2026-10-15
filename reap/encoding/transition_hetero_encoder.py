# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from __future__ import annotations

from itertools import chain
from typing import (
    Collection,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Sequence,
)

import torch_geometric as pyg
from torch_geometric.data import HeteroData

from reap.utils.misc import monkeypatch
from reap.xmimir import (
    XAction,
    XAtom,
    XDomain,
    XLiteral,
    XPredicate,
    XProblem,
    XState,
    atom_str_template,
)

from .base_encoder import (
    EncoderFactory,
)
from .hetero_encoder import HGraphEncoder
from .pyg_builder import PygBuilderBase
from .relation_dict import RelationDict


class PredicateEdgeType(NamedTuple):
    src_type: str
    pos: str
    dst_type: str


class XPredicatePatched(XPredicate):  # fake inheritance to mimic interface
    def __init__(self, predicate: XPredicate, new_name: str):
        self._predicate = predicate
        self._name = new_name

    @property
    def name(self):
        return self._name

    def __getattr__(self, item):
        return getattr(self._predicate, item)

    # Forward common special methods explicitly
    def __eq__(self, other):
        # Prefer to compare underlying predicate semantics if needed:
        if not isinstance(other, (XPredicatePatched, XPredicate)):
            return NotImplemented
        return self.semantic_eq(other)

    def __hash__(self):
        return self.semantic_hash()

    def __str__(self):
        return self._name


class XAtomPatched(XAtom):  # fake inheritance to mimic interface
    def __init__(self, atom: XAtom, new_predicate: XPredicate):
        self._atom = atom
        self._predicate = new_predicate

    @property
    def predicate(self):
        return self._predicate

    def __getattr__(self, item):
        return getattr(self._atom, item)

    # Forward common special methods explicitly
    def __eq__(self, other):
        # Prefer to compare underlying atom semantics if needed:
        if not isinstance(other, (XAtomPatched, XAtom)):
            return NotImplemented
        return self.predicate == other.predicate and self.objects == other.objects

    def __hash__(self):
        return self.semantic_hash()

    def __str__(self):
        return atom_str_template.render(predicate=self._predicate, objects=self.objects)


class XLiteralPatched(XLiteral):  # fake inheritance to mimic interface
    def __init__(self, literal: XLiteral, new_atom: XAtom):
        self._literal = literal
        self._atom = new_atom

    @property
    def atom(self):
        return self._atom

    def __getattr__(self, item):
        return getattr(self._literal, item)

    # Forward common special methods explicitly
    def __eq__(self, other):
        # Prefer to compare underlying literal semantics if needed:
        if not isinstance(other, (XLiteralPatched, XLiteral)):
            return NotImplemented
        return self.is_negated == other.is_negated and self.atom == other.atom

    def __hash__(self):
        return hash((self.is_negated, self.atom))


class TransitionHGraphEncoder(HGraphEncoder):
    successor_predicate_suffix = "[suc]"

    def __init__(
        self,
        domain: XDomain,
        *args,
        include_lgan_edges: bool = False,
        lgan_nn_edge_pos: str = "lgan_nn",
        **kwargs,
    ) -> None:
        self.pred_to_succ_pred: Dict[XPredicate, XPredicatePatched] = {
            p: XPredicatePatched(p, p.name + self.successor_predicate_suffix)
            for p in domain.predicates()
        }
        # monkeypatch domain's predicates function for super().__init__ to include successor predicates
        with monkeypatch(
            domain,
            "predicates",
            lambda: tuple(
                chain(
                    self.pred_to_succ_pred.keys(),
                    self.pred_to_succ_pred.values(),
                )
            ),
        ):
            super().__init__(
                domain,
                *args,
                include_lgan_edges=include_lgan_edges,
                lgan_nn_edge_pos=lgan_nn_edge_pos,
                **kwargs,
            )

    def _encode(
        self,
        builder: PygBuilderBase,
        facts: Sequence[XAtom],
        goals: Sequence[XLiteral],
        actions: Sequence[XAction],
        successor: XState | Iterable[XAtom] = None,
        **kwargs,
    ) -> None:
        assert successor is not None, "Successor state must be provided."
        # patch all atoms in successor_facts and successor_subgoal_layers to have successor predicates
        patched_successor_facts = [
            self._patch_atom(atom)
            for atom in (
                successor.atoms(with_statics=False)
                if isinstance(successor, XState)
                else successor
            )
        ]
        # combine facts
        all_facts = list(facts) + patched_successor_facts

        # we are not patching top-level goals, because doubling them would not be needed in the base setting
        super()._encode(
            builder,
            all_facts,
            goals,
            actions,
            **kwargs,
        )

        if self.relation_dict.goal_satisfaction_derivations and goals:
            glm = builder.graph_attrs["goal_level_map"]
            patched_goals = []
            for goal in goals:
                patched_goal = self._patch_literal(goal)
                patched_goals.append(patched_goal)
                if patched_goal not in glm:
                    glm[patched_goal] = glm[goal]
            satisfied_goals = self._goal_satisfaction_map(
                patched_successor_facts, patched_goals
            )
            self._encode_goal_satisfaction(satisfied_goals, builder, goal_level_map=glm)

    def encode(self, *args, **kwargs) -> HeteroData:
        # Call base encode which will call our _encode
        return super().encode(*args, **kwargs)

    def _patch_atom(self, atom: XAtom):
        return XAtomPatched(
            atom,
            self.pred_to_succ_pred[atom.predicate],
        )

    def _patch_literal(self, literal: XLiteral):
        patched_atom = self._patch_atom(literal.atom)
        return XLiteralPatched(
            literal,
            patched_atom,
        )


def get_facts(
    state: XState | Iterable[XAtom], problem: XProblem | None
) -> tuple[set[XAtom], XProblem | None]:
    if isinstance(state, XState):
        problem = problem or state.problem
        successor_facts = set(state.atoms(with_statics=False))
    else:
        successor_facts = set(state)
    return successor_facts, problem


class TransitionEffectsHGraphEncoder(HGraphEncoder):
    """
    An encoder to represent state transitions as heterogeneous graphs with objects and predicates as vertices
    and edges (i, j) whenever a predicate p(..., i, j, ...) holds in the state.
    Additionally, only predicates that change between the two states are included.

    """

    def __init__(
        self,
        *args,
        relation_dict: RelationDict | None = None,
        support_literals: bool = True,
        include_lgan_edges: bool = False,
        lgan_nn_edge_pos: str = "lgan_nn",
        **kwargs,
    ) -> None:
        if relation_dict is not None:
            assert relation_dict.support_literals, (
                f"{self.__class__.__name__} requires support_literals=True in RelationDict."
            )
        else:
            assert support_literals, (
                f"{self.__class__.__name__} requires support_literals=True."
            )
        super().__init__(
            *args,
            relation_dict=relation_dict,
            support_literals=support_literals,
            include_lgan_edges=include_lgan_edges,
            lgan_nn_edge_pos=lgan_nn_edge_pos,
            **kwargs,
        )

    def _encode(
        self,
        builder: PygBuilderBase,
        facts: Sequence[XAtom],
        goals: Sequence[XLiteral],
        actions: Sequence[XAction],
        successor: XState | Iterable[XAtom] = None,
        problem: XProblem | None = None,
        _state: XState | Iterable[XAtom] | None = None,
        **kwargs,
    ) -> None:
        assert successor is not None, "Successor state must be provided."
        base_facts, problem = get_facts(_state or facts, problem)
        successor_facts, problem = get_facts(successor, problem)
        assert problem is not None, (
            "If state and successor is given as an iterable of atoms, problem must be provided."
        )
        successor_literals, added_facts, removed_facts = _compute_change_literals(
            base_facts, successor_facts, problem
        )

        super()._encode(
            builder,
            facts,
            goals,
            actions,
            **kwargs,
        )

        if self.relation_dict.goal_satisfaction_derivations and goals:
            sat_goals = _compute_change_satisfaction(added_facts, removed_facts, goals)
            self._encode_goal_satisfaction(sat_goals, builder)

        self._encode_literals(successor_literals, builder, goal_level_map={})

    def encode(self, *args, **kwargs) -> HeteroData:
        return super().encode(*args, **kwargs)


def _compute_change_satisfaction(
    added_facts: Collection[XAtom],
    removed_facts: Collection[XAtom],
    goals: Sequence[XLiteral],
) -> dict[XLiteral, str]:
    sat_goals = {
        goal: "+"
        for goal in goals
        if any(goal.atom.semantic_eq(fact) for fact in added_facts) == goal.polarity
    } | {
        goal: "-"
        for goal in goals
        if any(goal.atom.semantic_eq(fact) for fact in removed_facts) != goal.polarity
    }
    return sat_goals


def _compute_change_literals(
    facts: set[XAtom], successor_facts: set[XAtom], problem: XProblem
) -> tuple[list[XLiteral], set[XAtom], set[XAtom]]:
    added = {
        atom
        for atom in successor_facts
        if not any(atom.semantic_eq(base_atom) for base_atom in facts)
    }
    removed = {
        atom
        for atom in facts
        if not any(atom.semantic_eq(succ_atom) for succ_atom in successor_facts)
    }
    literals: List[XLiteral] = []
    for atom in added:
        literals.append(XLiteral.register(problem, atom, polarity=True))
    for atom in removed:
        literals.append(XLiteral.register(problem, atom, polarity=False))
    return literals, added, removed
