# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Iterable, Sequence, TypeVar, Union

from .wrappers import *

T = TypeVar("T")


@dataclass
class MinDataPack(Generic[T], ABC):
    """
    Minimal interface for a data pack that can be used to reconstruct the original object.
    """

    def __init__(self):
        self._hash_cache = self._hash()

    @abstractmethod
    def reconstruct(self, *args, **kwargs) -> T:
        """
        Fully reconstruct the original object T from the data pack. May require additional parameters.
        """
        ...

    @abstractmethod
    def _hash(self): ...

    def __hash__(self):
        return self._hash_cache

    @abstractmethod
    def __eq__(self, other): ...


@dataclass
class ActionDataPack(MinDataPack[XAction]):
    """
    Minimal interface for an action data pack that can be used to reconstruct the original action.
    """

    schema: str
    objects: tuple[str, ...]

    def __init__(self, action: XAction):
        self.schema = action.action_schema.name
        self.objects = tuple(map(lambda o: o.name, action.objects))
        super().__init__()

    def __str__(self):
        """
        String representation of the action data pack.
        """
        return f"ActionDataPack({self.schema}, {self.objects})"

    def reconstruct(self, problem: XProblem) -> XAction:
        return XAction.register(problem, self.schema, self.objects)

    def _hash(self):
        return hash((self.schema, self.objects))

    __hash__ = MinDataPack.__hash__

    def __eq__(self, other):
        if not isinstance(other, ActionDataPack):
            return NotImplemented
        return self.schema == other.schema and self.objects == other.objects


@dataclass
class PredicateDataPack(MinDataPack[XPredicate]):
    """
    Minimal interface for a predicate data pack that can be used to reconstruct the original predicate.
    """

    name: str
    arity: int
    category: XCategory

    def __init__(self, predicate: XPredicate):
        self.name = predicate.name
        self.arity = predicate.arity
        self.category = predicate.category
        super().__init__()

    def reconstruct(self, problem: XProblem) -> XPredicate:
        predicate = problem.domain.predicate_lookup(self.name)
        assert predicate.arity == self.arity
        assert predicate.category == self.category
        return predicate

    def _hash(self):
        return hash((self.name, self.arity, self.category))

    __hash__ = MinDataPack.__hash__

    def __eq__(self, other):
        if not isinstance(other, PredicateDataPack):
            return NotImplemented
        return (self.name, self.arity, self.category) == (
            other.name,
            other.arity,
            other.category,
        )


@dataclass
class AtomDataPack(MinDataPack[XAtom]):
    """
    Minimal interface for a ground atom data pack that can be used to reconstruct the original ground atom.
    """

    predicate: str
    objects: tuple[str, ...]

    def __init__(self, atom: XAtom):
        self.predicate = atom.predicate.name
        self.objects = tuple(o.name for o in atom.objects)
        super().__init__()

    def reconstruct(self, problem: XProblem) -> XAtom:
        return XAtom.register(
            problem,
            problem.domain.predicate_lookup(self.predicate),
            map(problem.object_lookup, self.objects),
        )

    def __str__(self):
        return atom_str_template.render(predicate=self.predicate, objects=self.objects)

    def _hash(self):
        return hash((self.predicate, self.objects))

    __hash__ = MinDataPack.__hash__

    def __eq__(self, other):
        if not isinstance(other, AtomDataPack):
            return NotImplemented
        return (self.predicate, self.objects) == (other.predicate, other.objects)


@dataclass
class LiteralDataPack(MinDataPack[XLiteral]):
    """
    Minimal interface for a ground literal data pack that can be used to reconstruct the original ground literal.
    """

    atom: AtomDataPack
    polarity: bool

    def __init__(self, literal: XLiteral):
        self.atom = AtomDataPack(literal.atom)
        self.polarity = literal.polarity
        super().__init__()

    def reconstruct(self, problem: XProblem) -> XLiteral:
        return XLiteral.register(problem, self.atom.reconstruct(problem), self.polarity)

    def __str__(self):
        neg_str = "+" if self.polarity else "-"
        return f"{neg_str}{self.atom}"

    def _hash(self):
        return hash((self.atom, self.polarity))

    __hash__ = MinDataPack.__hash__

    def __eq__(self, other):
        if not isinstance(other, LiteralDataPack):
            return NotImplemented
        return (self.atom, self.polarity) == (other.atom, other.polarity)


@dataclass
class FunctionValueDataPack(MinDataPack[XFunctionValue]):
    """
    Minimal interface for a function value data pack that can be used to reconstruct the original function value.
    """

    function: str
    value: float

    def __init__(self, function_value: XFunctionValue):
        self.function = function_value.function.name
        self.value = function_value.value
        super().__init__()

    def reconstruct(self, problem: XProblem) -> XFunctionValue:
        return XFunctionValue(
            problem.domain.function_lookup(self.function),
            self.value,
        )

    def _hash(self):
        return hash((self.function, self.value))

    __hash__ = MinDataPack.__hash__

    def __eq__(self, other):
        if not isinstance(other, FunctionValueDataPack):
            return NotImplemented
        return (self.function, self.value) == (other.function, other.value)


@dataclass
class StateDataPack(MinDataPack[XState]):
    """
    Minimal interface for a state data pack that can be used to reconstruct the original state.
    """

    fluent_atoms: tuple[AtomDataPack, ...]
    numeric_values: tuple[FunctionValueDataPack, ...]

    def __init__(self, state: XState):
        self.fluent_atoms = tuple(AtomDataPack(atom) for atom in state.fluent_atoms)
        self.numeric_values = tuple(FunctionValueDataPack(nv) for nv in state.numerics)
        super().__init__()

    def reconstruct(self, problem: XProblem) -> XState:
        return XState.register(
            problem,
            (fa.reconstruct(problem) for fa in self.fluent_atoms),
            (nv.reconstruct(problem) for nv in self.numeric_values),
        )

    def _hash(self):
        return hash((self.fluent_atoms, self.numeric_values))

    __hash__ = MinDataPack.__hash__

    def __eq__(self, other):
        if not isinstance(other, StateDataPack):
            return NotImplemented
        return (self.fluent_atoms, self.numeric_values) == (
            other.fluent_atoms,
            other.numeric_values,
        )


@dataclass
class ActionHistoryDataPack(MinDataPack[XState], Sequence[ActionDataPack]):
    """
    Minimal interface for an action history data pack that can be used to reconstruct the original action history.
    """

    actions: list[ActionDataPack]

    def __init__(self, action_history: Iterable[XAction]):
        self.actions = [ActionDataPack(action) for action in action_history]
        super().__init__()

    def __len__(self) -> int:
        """
        Get the number of actions in the action history.
        """
        return len(self.actions)

    def __getitem__(
        self, index: int | slice
    ) -> Union["ActionHistoryDataPack", ActionDataPack]:
        """
        Get the action data pack at the specified index.
        """
        if isinstance(index, slice):
            copied = ActionHistoryDataPack(tuple())
            copied.actions = self.actions[index]
            return copied
        return self.actions[index]

    def __bool__(self):
        """
        Check if the action history is not empty.
        """
        return bool(self.actions)

    def __str__(self):
        """
        String representation of the action history.
        """
        return f"ActionHistoryDataPack({len(self.actions)}: {list(map(str, self.actions))})"

    def _hash(self):
        return hash(tuple(self.actions))

    __hash__ = MinDataPack.__hash__

    def __eq__(self, other):
        if not isinstance(other, ActionHistoryDataPack):
            return NotImplemented
        return self.actions == other.actions

    def reconstruct(self, successor_generator: XSuccessorGenerator) -> XState:
        state = successor_generator.initial_state
        for action_pack in self.actions:
            action = action_pack.reconstruct(successor_generator.problem)
            state, _ = successor_generator.successor(state, action)
        return state

    def reconstruct_sequence(
        self, successor_generator: XSuccessorGenerator
    ) -> list[XTransition]:
        """
        Reconstruct the sequence of states from the action history.
        """
        state = successor_generator.initial_state
        transitions = []
        for action_pack in self.actions:
            action = action_pack.reconstruct(successor_generator.problem)
            transition = XTransition(
                state, action, *successor_generator.successor(state, action)
            )
            transitions.append(transition)
            state = transition.target
        return transitions

    def extend(self, other: "ActionHistoryDataPack") -> None:
        """
        Extend the current action history with another action history.
        """
        self.actions.extend(other.actions)


@dataclass
class TransitionDataPack(MinDataPack[XTransition]):
    """
    Minimal interface for a macro action history data pack that can be used to reconstruct the original macro action history.
    """

    source: StateDataPack
    action: ActionDataPack | tuple[ActionDataPack, ...] | None
    target: StateDataPack
    cost: float | tuple[float, ...] | None

    def __init__(self, transition: XTransition):
        self.source = StateDataPack(transition.source)
        match transition.action:
            case XAction():
                self.action = ActionDataPack(transition.action)
                self.cost = transition.cost
            case tuple():
                self.action = tuple(ActionDataPack(a) for a in transition.action)
                self.cost = tuple(c for c in transition.cost)
            case None:
                self.action = None
                self.cost = None
            case _:
                raise TypeError(
                    f"Unsupported action type in TransitionDataPack: {type(transition.action)}"
                )
        self.target = StateDataPack(transition.target)
        super().__init__()

    def reconstruct(self, problem: XProblem) -> XTransition:
        source = self.source.reconstruct(problem)
        match self.action:
            case ActionDataPack():
                action = self.action.reconstruct(problem)
                cost = self.cost
            case tuple():
                action = tuple(a.reconstruct(problem) for a in self.action)
                cost = self.cost
            case None:
                action = None
                cost = None
            case _:
                raise TypeError(
                    f"Unsupported action type in TransitionDataPack: {type(self.action)}"
                )
        target = self.target.reconstruct(problem)
        return XTransition(source, action, target, cost)

    def _hash(self):
        return hash((self.source, self.action, self.target, self.cost))

    __hash__ = MinDataPack.__hash__

    def __eq__(self, other):
        if not isinstance(other, TransitionDataPack):
            return NotImplemented
        return (self.source, self.action, self.target, self.cost) == (
            other.source,
            other.action,
            other.target,
            other.cost,
        )


class ConjunctiveConditionDataPack(MinDataPack[XConjunctiveCondition]):
    """
    Minimal interface for a conjunctive condition data pack that can be used to reconstruct the original conjunctive condition.
    """

    literals: tuple[LiteralDataPack, ...]

    def __init__(self, condition: XConjunctiveCondition):
        if condition.numeric_constraints:
            raise NotImplementedError(
                "Numeric constraints are not supported in ConjunctiveConditionDataPack at this moment."
            )
        self.literals = tuple(LiteralDataPack(lit) for lit in condition.literals)
        super().__init__()

    def reconstruct(self, problem: XProblem) -> XConjunctiveCondition:
        return XConjunctiveCondition(
            XConjunctiveCondition.register(
                problem,
                (lit.reconstruct(problem) for lit in self.literals),
                (),
            )
        )

    def __str__(self):
        chained_str = " ʌ ".join(map(str, self.literals))
        return f"({chained_str})"

    def _hash(self):
        return hash(self.literals)

    __hash__ = MinDataPack.__hash__

    def __eq__(self, other):
        if not isinstance(other, ConjunctiveConditionDataPack):
            return NotImplemented
        return self.literals == other.literals
