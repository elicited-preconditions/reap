# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from __future__ import annotations

import collections
import itertools
import operator
import sys
import tempfile
from collections import deque
from enum import Enum, auto
from functools import cache, cached_property, lru_cache, reduce, singledispatchmethod
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pymimir.pymimir.advanced.search import (
    ConjunctiveConditionSatisficingBindingGenerator,
)

try:
    # Python 3.11+
    from typing import TypeVarTuple
except ImportError:  # Python 3.10 fallback
    from typing_extensions import TypeVarTuple

import networkx as nx
import numpy as np
from jinja2 import Template
from multimethod import multimethod
from pymimir.wrapper_search_astar import astar_eager
from typing_extensions import Unpack

import reap.xmimir.pymimir_advanced as admir
import reap.xmimir.pymimir_wrappers as wrapmir
from reap.xmimir.hints import *

from .hints import *

atom_str_template = Template(
    "({{ predicate.name if predicate is not string and predicate.name is defined else predicate }}"
    "{% if objects %}"
    "{% for obj in objects %}"
    " {% if obj is string %}{{ obj }}"
    "{% elif obj is mapping %}"
    "{% for k, v in obj.items() %}{{ k }}={{ v }} {% endfor %}"
    "{% elif obj.name is defined %}{{ obj.name }}"
    "{% else %}{{ obj }}{% endif %}"
    "{% endfor %}"
    "{% endif %})"
)


class XCategory(Enum):
    static = 0
    fluent = 1
    derived = 2


_static_types_set = {
    t for t in vars(admir).values() if isinstance(t, type) and "Static" in t.__name__
}
_fluent_types_set = {
    t for t in vars(admir).values() if isinstance(t, type) and "Fluent" in t.__name__
}
_derived_types_set = {
    t
    for t in vars(admir).values()
    if isinstance(t, type) and t.__name__ in ("Derived", "Auxiliary")
}


def lookup_category(typ):
    if typ in _static_types_set:
        return XCategory.static
    elif typ in _fluent_types_set:
        return XCategory.fluent
    elif typ in _derived_types_set:
        return XCategory.derived
    else:
        return None


def _resolve_runtime_class(cls, typ):
    """Best-effort resolution of a typing argument into a runtime class.

    Handles: plain classes, typing.ForwardRef, string forward refs
    (due to `from __future__ import annotations`), and falls back to
    unwrapping typing origins.
    """
    # Fast path: already a real class
    if isinstance(typ, type):
        return typ

    # Resolve typing.ForwardRef or str using the module globals of `cls`
    try:
        from typing import ForwardRef as _FR  # py312 friendly import
    except Exception:
        _FR = None
    g = getattr(sys.modules.get(cls.__module__), "__dict__", {})

    if _FR is not None and isinstance(typ, _FR):
        name = typ.__forward_arg__
        try:
            resolved = eval(name, g, None)
            if isinstance(resolved, type):
                return resolved
        except Exception:
            pass

    if isinstance(typ, str):
        try:
            resolved = eval(typ, g, None)
            if isinstance(resolved, type):
                return resolved
        except Exception:
            pass

    # If this is a typing object, try resolving its origin
    origin = get_origin(typ)
    if isinstance(origin, type):
        return origin

    return None


T = TypeVar("T")


class MimirWrapper(Generic[T]):
    """
    A mixin class to provide a base accessor that checks against hollow-ness of the underlying object.
    If not overridden de by the subclass, the equivalence check as well as the hash of the class
     are determined by the pymimir base object.
    Hollow refers to an instance without a pymimir base. This is useful for some classes
     like Transition where you might want to manually create instances of (source, action, target).
    """

    _base: T | None
    _wrapped_type: type | None = None

    def __init__(self, base: T | None):
        self._base = base

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Try to extract the concrete type arg from the subclass's Generic base.
        for base in getattr(
            cls, "__orig_bases__", ()
        ):  # e.g., MimirWrapper[admir.Predicate]
            origin = get_origin(base)
            if isinstance(origin, type) and issubclass(origin, MimirWrapper):
                args = get_args(base)
                if args:
                    resolved = _resolve_runtime_class(cls, args[0])
                    if isinstance(resolved, type):
                        cls._wrapped_type = resolved
                        break
        # If a concrete `__wrapped_type__` is declared on the subclass, cache it (supports str / ForwardRef)
        alt = getattr(cls, "__wrapped_type__", None)
        resolved = _resolve_runtime_class(cls, alt)
        if isinstance(resolved, type):
            cls._wrapped_type = resolved

    @property
    def base(self) -> T:
        return self._base

    @property
    def advanced(self):
        raise NotImplementedError(
            f"object of type <{self.__class__.__qualname__}> has no property 'advanced'."
        )

    @classmethod
    def registrable(cls) -> bool:
        """
        Does the class have a static 'register' function to create or lookup a new object of this kind?
        """
        return callable(getattr(cls, "register", None))

    @classmethod
    def base_type(cls) -> type[T] | None:
        """Return the concrete wrapped runtime class for this wrapper, if known."""
        return cls._wrapped_type

    @property
    def category(self) -> XCategory | None:
        return lookup_category(type(self.base))

    def __str__(self):
        return str(self.base)

    def __hash__(self):
        return hash(self.base)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if not isinstance(other.base, type(self.base)):
            return NotImplemented
        return self.base == other.base

    def semantic_eq(self, other):
        r"""
        Check for semantic equality of the object.

        This method is used to compare objects that are not the same in memory but are semantically the same.
        Most pymimir objects are mere views onto underlying data. In consequence, two states, two literals, two atoms,
        two actions, etc. may be semantically the same but are created in different repositories and thus may compare
        to unequal.
        Semantic equality should provide a way for the user to identify equality between such objects regardless.

        It should be understood that semantic equality does not ensure that the objects can be used interchangeably,
        particularly not in pymimir functionality.
        """
        raise NotImplementedError(
            f"Semantic equality is not implemented for class {self.__class__.__name__}."
        )

    @staticmethod
    def semantic_eq_sequences(
        container1: Sequence[MimirWrapper],
        container2: Sequence[MimirWrapper],
        *,
        ordered: bool,
    ):
        """
        Check for semantic equality of two containers.
        """
        if len(container1) != len(container2):
            return False

        if not ordered:
            return MimirWrapper.semantic_eq_subset(container1, container2)
        else:
            return all(a.semantic_eq(b) for a, b in zip(container1, container2))

    @staticmethod
    def semantic_eq_subset(
        container1: Sequence[MimirWrapper], container2: Sequence[MimirWrapper]
    ):
        """
        Check that container1 is a semantic subset of container2.

        This means that we check for all x in cont1 that x is semantically equal to at least 1 element in cont2.

        Note:
            An empty container1 is always a subset of container2.
        """
        return all(any(a.semantic_eq(b) for b in container2) for a in container1)


class XPredicate(MimirWrapper[Predicate]):
    def __init__(self, predicate: Predicate):
        super().__init__(predicate)

    @property
    def name(self) -> str:
        return self.base.get_name()

    @property
    def arity(self) -> int:
        return self.base.get_arity()

    @property
    def parameters(self) -> tuple[XParameter, ...]:
        return tuple(map(XParameter, self.base.get_parameters()))

    def semantic_eq(self, other: XPredicate):
        return (
            self.name == other.name
            and self.arity == other.arity
            and self.category == other.category
        )

    def semantic_hash(self):
        return hash((self.category, self.name, self.arity))


class XAtom(MimirWrapper[GroundAtom]):
    """
    The extended atom class.

    Important Notes:
    A pymimir.GroundAtom object is a mere pointer onto a ground atom stored in an pymimir.PDDLRepositories object.
    As such, it is not an independent object but a view onto memory that is kept alive by each pymimir.GroundAtom
    object.

    It is not guaranteed that the same atom is held at the same index in the same repository. Try to rely on
    only a single pymimir.PDDLRepositories object if possible.
    """

    def __init__(self, atom: GroundAtom | wrapmir.GroundAtom):
        if isinstance(atom, wrapmir.GroundAtom):
            atom = atom._advanced_ground_atom
        super().__init__(atom)

    @staticmethod
    def register(problem: XProblem, predicate: XPredicate, objects: Iterable[XObject]):
        """
        Create a new XPredicate from a problem, a predicate and a list of objects.
        """
        return XAtom(
            problem.advanced.get_or_create_ground_atom(
                predicate.base, (o.base for o in objects)
            )
        )

    @property
    def index(self):
        return self.base.get_index()

    @property
    def predicate(self):
        return XPredicate(self.base.get_predicate())

    @cached_property
    def objects(self) -> tuple[XObject, ...]:
        """
        Get the objects of the atom.
        """
        return tuple(map(XObject, self.base.get_objects()))

    def semantic_eq(self, other):
        return self.predicate.semantic_eq(other.predicate) and all(
            a.name == b.name for a, b in zip(self.objects, other.objects)
        )

    def semantic_hash(self):
        return hash(
            (self.predicate.semantic_hash(), tuple(o.name for o in self.objects))
        )

    def __repr__(self):
        return str(self)


class XObject(MimirWrapper[admir.Object]):
    def __init__(self, obj: admir.Object):
        super().__init__(obj)

    @property
    def name(self) -> str:
        return self.base.get_name()

    @property
    def index(self) -> int:
        return self.base.get_index()

    @property
    def type_(self):
        return self.type_bases[0]

    @cached_property
    def type_bases(self) -> tuple[XType, ...]:
        return tuple(map(XType, self.base.get_bases()))

    def __repr__(self):
        return str(self)


class XVariable(MimirWrapper[admir.Variable]):
    def __init__(self, var: admir.Variable):
        super().__init__(var)

    @property
    def name(self) -> str:
        return self.base.get_name()

    @property
    def index(self) -> int:
        return self.base.get_index()

    def __repr__(self):
        return str(self)


class XType(MimirWrapper[admir.Type]):
    def __init__(self, typ: admir.Type):
        super().__init__(typ)

    @property
    def name(self) -> str:
        return self.base.get_name()

    @property
    def index(self) -> int:
        return self.base.get_index()

    @property
    def type_bases(self) -> tuple[XType, ...]:
        return tuple(map(XType, self.base.get_bases()))

    def is_subtype_of(self, type_: XType) -> bool:
        # traverse all bases recursively, check if any matches type_
        bases = self.type_bases
        if not bases:
            return type_ == self
        to_visit = deque(self.type_bases)
        while to_visit:
            current = to_visit.popleft()
            if current == type_:
                return True
            to_visit.extend(current.type_bases)
        return False

    def __str__(self):
        return f"Type({self.name})"

    def __repr__(self):
        return str(self)


class XParameter(MimirWrapper[admir.Parameter]):
    def __init__(self, parameter: admir.Parameter):
        super().__init__(parameter)

    @property
    def index(self) -> int:
        """Index of the parameter in the predicate's parameter list."""
        return self.base.get_index()

    @property
    def variable(self) -> XVariable:
        """The underlying variable of this parameter as an XVariable wrapper."""
        return XVariable(self.base.get_variable())

    @cached_property
    def type_(self):
        return self.type_bases[0]

    @property
    def type_bases(self) -> tuple[XType, ...]:
        return tuple(map(XType, self.base.get_bases()))

    def __repr__(self):
        return str(self)


class XLiteral(MimirWrapper[GroundLiteral]):
    def __init__(self, literal: GroundLiteral | wrapmir.GroundLiteral):
        if isinstance(literal, wrapmir.GroundLiteral):
            literal = literal._advanced_ground_literal
        super().__init__(literal)

    @staticmethod
    def register(problem: XProblem, atom: XAtom, polarity: bool):
        advanced_objects = atom.base.get_objects()
        adv_atom = problem.advanced.get_or_create_atom(
            atom.predicate.base,
            (problem.advanced.get_or_create_term(obj) for obj in advanced_objects),
        )
        adv_literal = problem.advanced.get_or_create_literal(polarity, adv_atom)
        return XLiteral(problem.advanced.ground(adv_literal, advanced_objects))

    @property
    def polarity(self):
        return self.base.get_polarity()

    @property
    def is_negated(self):
        return (
            not self.base.get_polarity()
        )  # in pymimir positive polarity is `not negated`

    @property
    def atom(self) -> XAtom:
        return XAtom(self.base.get_atom())

    def semantic_eq(self, other):
        return self.is_negated == other.is_negated and self.atom.semantic_eq(other.atom)

    def __str__(self):
        return f"{'-' if self.is_negated else '+'}{self.atom}"

    def __repr__(self):
        return str(self)


class XFunctionSkeleton(MimirWrapper[FunctionSkeleton]):
    @property
    def name(self) -> str:
        return self.base.get_name()

    @property
    def index(self) -> int:
        return self.base.get_index()

    @cached_property
    def parameters(self):
        return tuple(map(XVariable, self.base.get_parameters()))


class XLiftedFunction(MimirWrapper[Function]):
    @property
    def skeleton(self) -> XFunctionSkeleton:
        return XFunctionSkeleton(self.base.get_function_skeleton())

    @property
    def name(self) -> str:
        return self.skeleton.name

    @property
    def index(self) -> int:
        return self.base.get_index()

    @cached_property
    def terms(self) -> tuple[XObject | XVariable, ...]:
        return tuple(
            XVariable(t) if isinstance(t, admir.Variable) else XObject(t)
            for t in (term.get() for term in self.base.get_terms())
        )


class XFunction(MimirWrapper[GroundFunction]):
    @property
    def skeleton(self) -> XFunctionSkeleton:
        return XFunctionSkeleton(self.base.get_function_skeleton())

    @property
    def name(self) -> str:
        return self.skeleton.name

    @property
    def index(self) -> int:
        return self.base.get_index()

    @cached_property
    def objects(self) -> tuple[XObject, ...]:
        return tuple(map(XObject, self.base.get_objects()))


class XFunctionValue:
    @multimethod
    def __init__(self, fluent_numeric: GroundFunctionValue):
        self.function = XFunction(fluent_numeric.get_function())
        self.value = fluent_numeric.get_number()

    @multimethod
    def __init__(self, function: GroundFunction, value: float):
        self.function = XFunction(function)
        self.value = value


class XLiftedExprOperator(MimirWrapper[FunctionExpressionOperator]):
    @property
    def index(self) -> int:
        return self.base.get_index()

    @property
    def arity(self) -> int:
        return len(self.args())

    @cache
    def args(
        self, cast: bool = True
    ) -> (
        tuple[XLiftedExprFunction | XLiftedExprOperator | XExprNumber, ...]
        | tuple[admir.FunctionExpression, ...]
    ):
        if cast:
            return tuple(XLiftedExpr.cast(expr) for expr in self._args())
        else:
            return tuple(self._args())

    def _args(self):
        match self.base:
            case admir.FunctionExpressionBinaryOperator():
                return (
                    self.base.get_left_function_expression(),
                    self.base.get_right_function_expression(),
                )
            case admir.FunctionExpressionMultiOperator():
                return (expr for expr in self.base.get_function_expressions())
            case admir.FunctionExpressionMinus():
                return (self.base.get_function_expression(),)
            case _:
                raise TypeError(
                    f"Unknown FunctionExpressionOperator type: {type(self.base)}"
                )

    @property
    def operator_str(self) -> str:
        return f"({self.name}, {self.arity})"

    @property
    def operator(
        self,
    ) -> (
        Callable[[float], float]
        | Callable[[float, float], float]
        | Callable[[Iterable[float]], float]
    ):
        if isinstance(self.base, admir.FunctionExpressionMultiOperator):
            return _MULTIOP_MAP[self.base.get_multi_operator()]
        if isinstance(self.base, admir.FunctionExpressionBinaryOperator):
            return _BINOP_MAP[self.base.get_binary_operator()]
        return operator.neg  # unary minus

    def ground(self, state: XState):
        raise NotImplementedError(
            f"Grounding not implemented yet for {self.__class__.__name__}."
        )


class XLiftedExprFunction(MimirWrapper[FunctionExpressionFunction]):
    @property
    def index(self) -> int:
        return self.base.get_index()

    @property
    def function(self) -> XLiftedFunction:
        return XLiftedFunction(self.base.get_function())


class XExprFunction(MimirWrapper[GroundFunctionExpressionFunction]):
    @property
    def index(self) -> int:
        return self.base.get_index()

    @property
    def function(self) -> XFunction:
        return XFunction(self.base.get_function())

    @property
    def objects(self) -> tuple[XObject, ...]:
        return tuple(map(XObject, self.base.get_function().get_objects()))

    @property
    def terms(self) -> tuple[XObject, ...]:
        return self.objects

    def evaluate(self, state: XState) -> float:
        return state.numeric_value(self.function)


class XExprOperator(MimirWrapper[GroundFunctionExpressionOperator]):
    def __init__(self, func_expr_op: GroundFunctionExpressionOperator):
        super().__init__(func_expr_op)

    @property
    def index(self) -> int:
        return self.base.get_index()

    @cached_property
    def name(self) -> str:
        match self.base:
            case admir.GroundFunctionExpressionBinaryOperator():
                return self.base.get_binary_operator().name
            case admir.GroundFunctionExpressionMultiOperator():
                return self.base.get_multi_operator().name
            case admir.GroundFunctionExpressionMinus():
                return "MINUS"
            case _:
                raise TypeError(
                    f"Unknown GroundFunctionExpressionOperator type: {type(self.base)}"
                )

    @property
    def arity(self) -> int:
        return len(self.args())

    @cache
    def args(
        self, cast: bool = True
    ) -> (
        tuple[XExprFunction | XExprOperator | XExprOperator, ...]
        | tuple[admir.GroundFunctionExpression, ...]
    ):
        if cast:
            return tuple(XExpr.cast(expr) for expr in self._args())
        else:
            return tuple(self._args())

    def _args(self):
        match self.base:
            case admir.GroundFunctionExpressionBinaryOperator():
                return (
                    self.base.get_left_function_expression(),
                    self.base.get_right_function_expression(),
                )
            case admir.GroundFunctionExpressionMultiOperator():
                return (expr for expr in self.base.get_function_expressions())
            case admir.GroundFunctionExpressionMinus():
                return (self.base.get_function_expression(),)
            case _:
                raise TypeError(
                    f"Unknown GroundFunctionExpressionOperator type: {type(self.base)}"
                )

    @property
    def operator_str(self) -> str:
        return f"({self.name}, {self.arity})"

    @property
    def operator(
        self,
    ) -> (
        Callable[[float], float]
        | Callable[[float, float], float]
        | Callable[[Iterable[float]], float]
    ):
        if isinstance(self.base, admir.GroundFunctionExpressionMultiOperator):
            return _MULTIOP_MAP[self.base.get_multi_operator()]
        if isinstance(self.base, admir.GroundFunctionExpressionBinaryOperator):
            return _BINOP_MAP[self.base.get_binary_operator()]
        return operator.neg  # unary minus

    def evaluate(self, state: XState) -> float:
        base = self.base
        if isinstance(base, admir.GroundFunctionExpressionBinaryOperator):
            return self._evaluate_binary(state)
        if isinstance(base, admir.GroundFunctionExpressionMultiOperator):
            return self._evaluate_multi(state)
        if isinstance(base, admir.GroundFunctionExpressionMinus):
            return self._evaluate_unary_minus(state)
        raise TypeError(f"Unknown operator node base type: {type(base)}")

    def _evaluate_binary(self, state: XState) -> float:
        children = self.args(cast=True)
        l = children[0].evaluate(state)
        r = children[1].evaluate(state)
        return float(self.operator(l, r))

    def _evaluate_multi(self, state: XState) -> float:
        vals = [arg.evaluate(state) for arg in self.args(cast=True)]
        return self.operator(vals)

    def _evaluate_unary_minus(self, state: XState) -> float:
        children = self.args(cast=True)
        return -(children[0].evaluate(state))


class XExprNumber(
    MimirWrapper[admir.GroundFunctionExpressionNumber | admir.FunctionExpressionNumber]
):
    @property
    def index(self) -> int:
        return self.base.get_index()

    @property
    def value(self) -> float:
        return self.base.get_number()

    def evaluate(self, state: XState) -> float:
        return self.value


XAnyExpr = XExprFunction | XExprOperator | XExprNumber


TExprUnion = TypeVar("TExprUnion")


class _ExprCasterBase(Generic[TExprUnion]):
    """Generic caster that iterates over a Union of wrapper classes.

    Subclasses must set `cast_outcome_type` to a `typing.Union[...] | None` where the
    last alternative is `None` so that `.__args__[:-1]` iterates only concrete wrapper types.
    Each wrapper type must derive from `MimirWrapper` and implement `base_type()` returning
    the corresponding advanced class to check against.
    """

    # e.g., `XAnyExpr | None` or `XAnyLiftedExpr | None`
    cast_outcome_type: TExprUnion | None

    _cast_map = {
        cls.base_type(): cls for cls in get_args(TExprUnion) if cls is not type(None)
    }

    @classmethod
    def cast(cls, expr, throws=False) -> TExprUnion | None:
        try:
            variant = expr.get()
            return cls._cast_map[variant](variant)
        except (KeyError, TypeError):
            if throws:
                raise TypeError(
                    f"Cannot cast expression of type {type(expr)} to any of {list(cls._cast_map.keys())}."
                )
            return None


XAnyLiftedExpr = XLiftedExprFunction | XLiftedExprOperator | XExprNumber


class XExpr(
    _ExprCasterBase[XAnyExpr],
    MimirWrapper[admir.GroundFunctionExpression],
):
    cast_outcome_type = XAnyExpr | None


class XLiftedExpr(
    _ExprCasterBase[XAnyLiftedExpr], MimirWrapper[admir.FunctionExpression]
):
    cast_outcome_type = XAnyLiftedExpr | None


NCType = TypeVar("NCType", admir.NumericConstraint, admir.GroundNumericConstraint)
XExprVariantType = TypeVar("XExprVariantType", XExpr, XLiftedExpr)
XExprTypes = TypeVarTuple("XExprTypes")

_cmp_precision = 1e-10


def _cmp_eq(a: float, b: float, eps: float = _cmp_precision) -> bool:
    return abs(a - b) <= eps


def _cmp_lt(a: float, b: float, eps: float = _cmp_precision) -> bool:
    return a < b - eps


def _cmp_le(a: float, b: float, eps: float = _cmp_precision) -> bool:
    return a <= b + eps


def _cmp_gt(a: float, b: float, eps: float = _cmp_precision) -> bool:
    return a > b + eps


def _cmp_ge(a: float, b: float, eps: float = _cmp_precision) -> bool:
    return a >= b - eps


def _reduce_add(seq: Iterable[float | int]) -> float:
    return reduce(operator.add, seq, 0.0)


def _reduce_mul(seq: Iterable[float | int]) -> float:
    return reduce(operator.mul, seq, 1.0)


_CMP_MAP = {
    admir.BinaryComparatorEnum.EQUAL: _cmp_eq,
    admir.BinaryComparatorEnum.LESS: _cmp_lt,
    admir.BinaryComparatorEnum.LESS_EQUAL: _cmp_le,
    admir.BinaryComparatorEnum.GREATER: _cmp_gt,
    admir.BinaryComparatorEnum.GREATER_EQUAL: _cmp_ge,
}
_BINOP_MAP = {
    admir.BinaryOperatorEnum.PLUS: operator.add,
    admir.BinaryOperatorEnum.MINUS: operator.sub,
    admir.BinaryOperatorEnum.MUL: operator.mul,
    admir.BinaryOperatorEnum.DIV: operator.truediv,
}
_MULTIOP_MAP = {
    admir.MultiOperatorEnum.PLUS: _reduce_add,
    admir.MultiOperatorEnum.MUL: _reduce_mul,
}


class XNumericConstraintDAG(
    MimirWrapper[NCType], Generic[NCType, XExprVariantType, Unpack[XExprTypes]]
):
    def __init__(self, numeric_constraint: NCType):
        super().__init__(numeric_constraint)
        self.graph: nx.DiGraph = nx.DiGraph()
        self._left_root = self._parse(
            XExprVariantType.cast(self.base.get_left_function_expression(), throws=True)
        )
        self._right_root = self._parse(
            XExprVariantType.cast(
                self.base.get_right_function_expression(), throws=True
            )
        )

    @property
    def comparator(self) -> Callable[[float, float], bool]:
        return _CMP_MAP[self.base.get_binary_comparator()]

    @property
    def comparator_enum(self) -> admir.BinaryComparatorEnum:
        return self.base.get_binary_comparator()

    @property
    def left_root(self):
        return self._left_root

    @property
    def right_root(self):
        return self._right_root

    @property
    def roots(self) -> tuple[Any, Any]:
        return self._left_root, self._right_root

    def leaves(self) -> tuple[Any, ...]:
        return tuple(k for k in self.graph.nodes if self.graph.out_degree(k) == 0)

    def __str__(self) -> str:
        return str(self.base)

    def _parse(self, expr: XAnyExpr) -> Any:
        key = expr
        if key in self.graph:
            return key
        return self._insert_node(expr, key)

    @singledispatchmethod
    def _insert_node(
        self, expr, key: admir.GroundFunctionExpression | admir.FunctionExpression
    ):
        raise NotImplementedError(f"Unsupported expression type {type(expr)}")

    @_insert_node.register
    def _(self, expr: XExprNumber, key: admir.GroundFunctionExpression):
        attrs = {
            "expr": expr,
            "value": expr.value,
            "op": None,
            "function_name": None,
            "function_category": None,
            "terms": (),
        }
        return self._insert_node_impl(key, attrs, children=())

    @_insert_node.register
    def _(self, expr: XExprFunction, key: admir.GroundFunctionExpression):
        cat = lookup_category(type(expr.function.base))
        attrs = {
            "expr": expr,
            "value": None,
            "op": None,
            "function_name": expr.function.name,
            "function_category": cat.name if cat is not None else None,
            "terms": expr.terms,
        }
        return self._insert_node_impl(key, attrs, children=())

    @_insert_node.register
    def _(self, expr: XLiftedFunction, key: admir.GroundFunctionExpression):
        cat = lookup_category(type(expr.function.base))
        attrs = {
            "expr": expr,
            "value": None,
            "op": None,
            "function_name": expr.function.name,
            "function_category": cat.name if cat is not None else None,
            "terms": expr.terms,
        }
        return self._insert_node_impl(key, attrs, children=())

    @_insert_node.register
    def _(self, expr: XExprOperator, key: admir.FunctionExpression):
        child_keys = tuple(self._parse(child) for child in expr.args())
        attrs = {
            "expr": expr,
            "value": None,
            "op": expr.operator_str,
            "function_name": None,
            "function_category": None,
            "terms": (),
        }
        return self._insert_node_impl(key, attrs, children=child_keys)

    @_insert_node.register
    def _(self, expr: XLiftedExprOperator, key: admir.FunctionExpression):
        child_keys = tuple(self._parse(child) for child in expr.args())
        attrs = {
            "expr": expr,
            "value": None,
            "op": expr.operator_str,
            "function_name": None,
            "function_category": None,
            "terms": (),
        }
        return self._insert_node_impl(key, attrs, children=child_keys)

    def _insert_node_impl(
        self,
        expr_key: admir.GroundFunctionExpression | admir.FunctionExpression,
        attrs: Mapping[str, Any],
        *,
        children: tuple[Any, ...],
    ):
        self.graph.add_node(expr_key, **attrs)
        for c in children:
            self.graph.add_edge(expr_key, c)
        return expr_key


class XLiftedNumericConstraint(
    XNumericConstraintDAG[
        admir.NumericConstraint,
        XLiftedExpr,
        XLiftedFunction,
        XLiftedExprFunction,
        XLiftedExprOperator,
        XExprNumber,
    ]
):
    def ground(self, state: XState) -> XNumericConstraint:
        left_expr = self.graph.nodes[self.left_root]["expr"]
        right_expr = self.graph.nodes[self.right_root]["expr"]
        problem = state.problem
        grounded_constraint = problem.advanced.get_or_create_numeric_constraint(
            self.comparator_enum,
            left_expr.base,
            right_expr.base,
            admir.TermList(
                problem.advanced.get_or_create_term(obj) for obj in problem.objects
            ),
        )
        return XNumericConstraint(grounded_constraint)


class XNumericConstraint(
    XNumericConstraintDAG[
        admir.GroundNumericConstraint,
        XExpr,
        XFunction,
        XExprOperator,
        XExprFunction,
        XExprNumber,
    ]
):
    @staticmethod
    def register(
        problem: XProblem,
        comp: admir.BinaryComparatorEnum,
        left_expr: XExprFunction,
        right_expr: XExprFunction,
        bindings: Iterable[XObject],
    ):
        adv_constraint = problem.advanced.get_or_create_numeric_constraint(
            comp, left_expr.base, right_expr.base, (o.base for o in bindings)
        )
        return XNumericConstraint(adv_constraint)

    def _eval_node(self, key: Any, state: XState) -> float:
        node_expr = self.graph.nodes[key]["expr"]
        return node_expr.evaluate(state)

    def holds(self, state: XState) -> bool:
        lhs = self._eval_node(self.left_root, state)
        rhs = self._eval_node(self.right_root, state)
        return self.comparator(lhs, rhs)


class XLiftedConjunctiveCondition(MimirWrapper[wrapmir.ConjunctiveCondition]):
    def __init__(
        self,
        literals: wrapmir.ConjunctiveCondition,
        numerics: Iterable[XLiftedNumericConstraint] = (),
    ):
        super().__init__(literals)
        self._literals = literals
        self._numerics = tuple(numerics)

    @property
    def advanced(self):
        return self.base._advanced_conjunctive_condition

    def ground(
        self,
        state: XState,
        max_groundings: int = -1,
        blacklist: Sequence[XPredicate] | None = None,
        binding_generator: admir.ConjunctiveConditionSatisficingBindingGenerator
        | None = None,
    ) -> list[XConjunctiveCondition]:
        assert len(self._numerics) == 0, "Numeric conditions are not supported yet."
        if max_groundings < 0:
            max_groundings = 1_000_000_000  # Generate all groundings
        problem = state.problem
        adv_problem = problem.advanced
        if binding_generator is None:
            binding_generator = ConjunctiveConditionSatisficingBindingGenerator(
                self.advanced, adv_problem
            )
        result: list[XConjunctiveCondition] = []
        if (blacklist is None) or (len(blacklist) == 0):
            groundings = binding_generator.generate_ground_conjunctions(
                state.base, max_groundings
            )
            for objects, (
                static_ground_literals,
                fluent_ground_literals,
                derived_ground_literals,
            ) in groundings:
                advanced_condition = (
                    adv_problem.get_or_create_ground_conjunctive_condition(
                        static_ground_literals,
                        fluent_ground_literals,
                        derived_ground_literals,
                        admir.GroundNumericConstraintList(),
                    )
                )
                result.append(
                    XConjunctiveCondition(
                        wrapmir.GroundConjunctiveCondition(
                            advanced_condition, problem.base
                        ),
                    )
                )
        else:
            static_blacklist, fluent_blacklist, derived_blacklist = set(), set(), set()
            for x in blacklist:
                match x.category:
                    case XCategory.static:
                        static_blacklist.add(x.base)
                    case XCategory.fluent:
                        fluent_blacklist.add(x.base)
                    case XCategory.derived:
                        derived_blacklist.add(x.base)
                    case _:
                        raise ValueError(
                            f"Found category with unknown value: {x.category}"
                        )
            groundings = binding_generator.generate_ground_conjunctions(
                state.base, max_groundings
            )
            for objects, (
                static_ground_literals,
                fluent_ground_literals,
                derived_ground_literals,
            ) in groundings:
                static_ground_literals = [
                    x
                    for x in static_ground_literals
                    if x.get_atom().get_predicate() not in static_blacklist
                ]
                fluent_ground_literals = [
                    x
                    for x in fluent_ground_literals
                    if x.get_atom().get_predicate() not in fluent_blacklist
                ]
                derived_ground_literals = [
                    x
                    for x in derived_ground_literals
                    if x.get_atom().get_predicate() not in derived_blacklist
                ]
                advanced_condition = (
                    adv_problem.get_or_create_ground_conjunctive_condition(
                        static_ground_literals,
                        fluent_ground_literals,
                        derived_ground_literals,
                        admir.GroundNumericConstraintList(),
                    )
                )
                result.append(
                    XConjunctiveCondition(
                        wrapmir.GroundConjunctiveCondition(
                            advanced_condition, problem.base
                        )
                    )
                )
        return result


class XConjunctiveCondition(
    MimirWrapper[wrapmir.GroundConjunctiveCondition],
    Sequence[Union[XLiteral, XNumericConstraint]],
):
    _numeric_constraints: tuple[XNumericConstraint, ...] = ()

    def __init__(
        self,
        condition: wrapmir.GroundConjunctiveCondition,
        numeric_constraints: Optional[Iterable[XNumericConstraint]] = None,
    ):
        super().__init__(condition)
        self._numeric_constraints = (
            tuple(numeric_constraints) if numeric_constraints is not None else ()
        )
        static_size = len(self.base._static_ground_literals)
        fluent_size = len(self.base._fluent_ground_literals)
        derived_size = len(self.base._derived_ground_literals)
        self._length = len(self.base)
        self._offsets = [0, static_size, static_size + fluent_size, derived_size]

    def __iter__(self) -> Iterator[XLiteral]:
        return iter(self.literals)

    def __len__(self) -> int:
        return len(self.literals)

    def __getitem__(self, index: int) -> XLiteral | XNumericConstraint:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        if index < self._offsets[1]:
            return XLiteral(self.base._static_ground_literals[index])
        elif index < self._offsets[2]:
            return XLiteral(self.base._fluent_ground_literals[index - self._offsets[1]])
        elif index < self._offsets[3]:
            return XLiteral(
                self.base._derived_ground_literals[index - self._offsets[2]]
            )
        else:
            return self.numeric_constraints[index - self._offsets[3]]

    @staticmethod
    def register(
        problem: XProblem,
        literals: Iterable[XLiteral],
        numerics: Iterable[XNumericConstraint] = (),
    ):
        return XConjunctiveCondition(
            wrapmir.GroundConjunctiveCondition.new(
                [wrapmir.GroundLiteral(lit.base) for lit in literals], problem.base
            ),
            numerics,
        )

    @property
    def problem(self) -> XProblem:
        return XProblem(self.base.get_problem())

    @cached_property
    def literals(self) -> tuple[XLiteral, ...]:
        return tuple(map(XLiteral, self.base))

    @property
    def numeric_constraints(self) -> tuple[XNumericConstraint, ...]:
        return self._numeric_constraints

    def holds(self, state: XState) -> bool:
        return all(map(state.literal_holds, self.literals)) and all(
            map(state.numeric_constraint_holds, self.numeric_constraints)
        )

    def lift(self, add_inequalities: bool = False) -> XConjunctiveCondition:
        lifted_literals = self.base.lift(add_inequalities)
        lifted_numerics = tuple(
            nc.lift(add_inequalities) for nc in self.numeric_constraints
        )
        return XLiftedConjunctiveCondition(lifted_literals, lifted_numerics)

    def __str__(self):
        conjunction = " ʌ ".join(
            map(str, chain(self.literals, self.numeric_constraints))
        )
        return f"({conjunction})"


class XDomain(MimirWrapper[wrapmir.Domain]):
    def __init__(
        self,
        domain: wrapmir.Domain | admir.Domain,
        parser: Optional[admir.Parser] = None,
    ):
        if isinstance(domain, admir.Domain):
            # Builds the wrapper by hand. This depends on the internal structure of the pymimir wrappers
            # and breaks if it changes.
            wrapper_domain = object.__new__(wrapmir.Domain)
            wrapper_domain._advanced_parser = (
                parser  # must be given if the wrapper ever needs it
            )
            wrapper_domain._advanced_domain = domain
            wrapper_domain._repositories = domain.get_repositories()
            domain = wrapper_domain
        super().__init__(domain)
        self._predicate_dict = {p.name: p for p in self.predicates()}
        self._action_dict = {a.name: a for a in self.actions}
        self._constant_dict = {c.name: c for c in self.constants}
        self._function_dict = {f.name: f for f in self.functions}

    @staticmethod
    def parse(domain_path: str | Path) -> XDomain:
        domain = wrapmir.Domain(str(domain_path))
        return XDomain(domain)

    @property
    def advanced(self):
        return self.base._advanced_domain

    @property
    def name(self) -> str:
        return self.base.get_name()

    @property
    def filepath(self) -> str:
        return str(self.advanced.get_filepath())

    @cache
    def predicate_lookup(self, name: str) -> XPredicate:
        return self._predicate_dict[name]

    def has_predicate(self, name: str) -> bool:
        return name in self._predicate_dict

    @cache
    def predicates(self, *category: XCategory) -> tuple[XPredicate, ...]:
        if not category:
            category = XCategory.__members__.values()

        iterable = chain.from_iterable(
            getattr(self.advanced, f"get_{cat.name}_predicates")() for cat in category
        )
        return tuple(XPredicate(p) for p in iterable)

    @cached_property
    def constants(self) -> tuple[XObject, ...]:
        return tuple(map(XObject, self.base.get_constants()))

    def constant_lookup(self, name: str):
        return self._constant_dict[name]

    @cached_property
    def types(self) -> tuple[XType, ...]:
        return tuple(map(XType, self.advanced.get_types()))

    @cached_property
    def actions(self) -> tuple[XActionSchema, ...]:
        return tuple(map(XActionSchema, self.advanced.get_actions()))

    @cache
    def action_lookup(self, name: str) -> XActionSchema:
        return self._action_dict[name]

    @cached_property
    def functions(self) -> tuple[XFunction, ...]:
        return tuple(XFunction(p) for p in self.advanced.get_fluent_functions())

    @cache
    def function_lookup(self, name: str) -> XFunction:
        return self._function_dict[name]

    @cached_property
    def requirements(self) -> list[str]:
        return self.base.get_requirements()

    def __str__(self):
        return str(self.base)

    def semantic_eq(self, other):
        if Path(self.filepath).exists() and Path(other.filepath).exists():
            if self.filepath == other.filepath:
                return True
            content1 = Path(self.filepath).read_text()
            content2 = Path(other.filepath).read_text()
            if content1 == content2:
                # two identical files must be semantically equal
                return True
        raise NotImplementedError(
            "No semantic check for domains beyond PDDL file contents."
        )

    def __reduce__(self):
        # when unpickling, python will call: XDomain.parse(self.filepath)
        return self.parse, (self.filepath,)


class XProblem(MimirWrapper[wrapmir.Problem]):
    """
    The extended problem class.

    Each XProblem holds its corresponding PDDLRepositories object to access the ground atoms and literals with.
    """

    @multimethod
    def __init__(self, problem: wrapmir.Problem):
        super().__init__(problem)
        self._object_dict = {o.name: o for o in self.objects}

    @multimethod
    def __init__(self, search_context: admir.SearchContext):
        """
        Builds the problem from a search context. This depends on the internal structure of the pymimir wrappers
        and breaks if it changes.
        """
        problem = search_context.get_problem()
        wrapper_problem = object.__new__(wrapmir.Problem)
        wrapper_problem._advanced_problem = problem
        wrapper_problem._search_context = search_context
        wrapper_problem._domain = XDomain(problem.get_domain(), None).base
        wrapper_problem._static_ground_atom_indices = {
            atom.get_index() for atom in problem.get_static_initial_atoms()
        }
        self.__init__(wrapper_problem)

    @multimethod
    def __init__(self, domain: XDomain, problem_path: str | Path):
        problem = wrapmir.Problem(domain.base, str(problem_path))
        XProblem.__init__(self, problem)

    def __eq__(self, other):
        if not isinstance(other, XProblem):
            return NotImplemented
        return (
            self.advanced == other.advanced
            and self.search_context == other.search_context
        )

    def __hash__(self):
        return hash((self.advanced, self.search_context))

    @property
    def advanced(self):
        return self.base._advanced_problem

    @property
    def search_context(self):
        return self.base._search_context

    @search_context.setter
    def search_context(self, context: admir.SearchContext):
        # replace the private attribute in the base Problem (depends on pymimir internals)
        self.base._search_context = context

    @property
    def repositories(self) -> admir.Repositories:
        """
        Get the repositories of the problem.
        """
        return self.advanced.get_repositories()

    @property
    def state_repository(self) -> admir.StateRepository:
        return self.search_context.get_state_repository()

    @property
    def name(self) -> str:
        return self.base.get_name()

    @property
    def filepath(self) -> str:
        return str(self.advanced.get_filepath())

    @cached_property
    def domain(self):
        return XDomain(self.base.get_domain())

    @cached_property
    def objects(self) -> tuple[XObject, ...]:
        return tuple(map(XObject, self.advanced.get_objects()))

    @cached_property
    def type_hierarchy(self) -> nx.DiGraph:
        dag: nx.DiGraph = nx.DiGraph()

        def add_type_and_bases(t: XType) -> None:
            stack = [t]
            while stack:
                cur = stack.pop()
                if base_types := cur.type_bases:
                    for b in base_types:
                        # parent -> child
                        dag.add_edge(b, cur)
                        stack.append(b)
                else:
                    # root type
                    dag.add_node(cur)

        if self.domain.types:
            for ty in self.domain.types:
                for base_ty in ty.type_bases:
                    dag.add_edge(base_ty, ty)
        else:
            for obj in chain(self.objects, self.domain.constants):
                for t in obj.type_bases:
                    add_type_and_bases(t)

            for pred in self.domain.predicates():
                for param in pred.parameters:
                    for t in param.type_bases:
                        add_type_and_bases(t)
        assert nx.is_directed_acyclic_graph(dag)
        return dag

    @property
    def ground_functions(self) -> tuple[XFunction, ...]:
        aux_value = self.advanced.get_auxiliary_function_value()
        return tuple(
            XFunction(fv.get_function())
            for fv in chain(
                self.advanced.get_fluent_function_values() or (),
                (aux_value,) if aux_value else (),
            )
        )

    @cache
    def object_lookup(self, name: str) -> XObject:
        return self._object_dict[name]

    @cache
    def goal(self) -> XConjunctiveCondition:
        return XConjunctiveCondition(
            self.base.get_goal_condition(), self.numeric_goal()
        )

    @cache
    def literal_goal(self) -> tuple[XLiteral, ...]:
        return tuple(map(XLiteral, self.base.get_goal_condition()))

    @cache
    def numeric_goal(self) -> tuple[XNumericConstraint, ...]:
        return tuple(
            map(XNumericConstraint, self.advanced.get_goal_numeric_constraints())
        )

    @cached_property
    def static_atoms(self) -> tuple[XAtom, ...]:
        return tuple(
            map(
                XAtom,
                chain(
                    self.advanced.get_static_initial_atoms(),
                ),
            )
        )

    @cached_property
    def initial_literals(self) -> tuple[XLiteral, ...]:
        """
        Get the initial literals of the problem definition.
        """
        return tuple(
            map(
                XLiteral,
                chain(
                    self.advanced.get_static_initial_literals(),
                    self.advanced.get_fluent_initial_literals(),
                ),
            ),
        )

    @property
    def initial_fluents(self) -> tuple[XLiteral, ...]:
        """
        Get the initial fluent values of the problem definition.
        """
        return tuple(map(XLiteral, self.advanced.get_fluent_initial_literals()))

    @cached_property
    def initial_atoms(self) -> tuple[XAtom, ...]:
        """
        Get the initial atoms of the problem definition.
        """
        derived_atoms = self.repositories.get_derived_ground_atoms_from_indices(
            tuple(
                self.state_repository.get_or_create_initial_state()[
                    0
                ].get_derived_atoms()
            )
        )
        return tuple(
            map(
                XAtom,
                chain(
                    self.advanced.get_static_initial_atoms(),
                    self.advanced.get_fluent_initial_atoms(),
                    derived_atoms,
                ),
            )
        )

    def atom_count(self, category: XCategory):
        counter = itertools.count()
        get_atom = getattr(self.repositories, f"get_{category.name}_ground_atom")
        while True:
            count = next(counter)
            try:
                get_atom(count)
            except IndexError:
                return count

    @cache
    def all_atoms(self, *category: XCategory) -> tuple[XAtom, ...]:
        """
        Returns all possible atoms of the given category with the problem's constants.
        """
        all_possible_atoms = []
        category = set(category)
        if XCategory.static in category:
            if len(category) == 1:
                return tuple(sorted(self.static_atoms, key=lambda atom: atom.index))
            all_possible_atoms.extend(self.static_atoms)
        type_to_objects = {
            t: [
                obj
                for obj in self.objects
                if obj.type_ == t or obj.type_.is_subtype_of(t)
            ]
            for t in self.type_hierarchy.nodes
        }
        for predicate in self.domain.predicates(*category):
            all_possible_atoms.extend(
                XAtom.register(self, predicate, args)
                for args in itertools.product(
                    *(type_to_objects[param.type_] for param in predicate.parameters)
                )
            )
        all_possible_atoms.sort(key=lambda a: a.index)
        return tuple(all_possible_atoms)

    def __str__(self):
        return (
            f"Problem: {{\n"
            f"name: {self.name} ({Path(self.filepath).absolute()}),\n"
            f"domain: {self.domain.name},\n"
            f"objects: {self.objects},\n"
            f"initial: {' ∧ '.join(str(a) for a in self.initial_atoms)},\n"
            f"goal: {' ∧ '.join(str(l) for l in self.goal())}\n"
            f"}}"
        )

    def semantic_eq(self, other):
        if Path(self.filepath).exists() and Path(other.filepath).exists():
            content1 = Path(self.filepath).read_text()
            content2 = Path(other.filepath).read_text()
            if content1 == content2:
                # two identical files must be semantically equal
                return True

        return (
            self.name == other.name
            and self.domain.semantic_eq(other.domain)
            and len(self.objects) == len(other.objects)
            and all(a.name == b.name for a, b in zip(self.objects, other.objects))
            and self.semantic_eq_sequences(
                self.initial_literals,
                other.initial_literals,
                ordered=False,
            )
            and self.semantic_eq_sequences(
                self.goal(),
                other.goal(),
                ordered=False,
            )
        )


class CustomProblem(XProblem):
    def __init__(
        self,
        problem: XProblem,
        goal: XConjunctiveCondition | None = None,
        initial: tuple[XLiteral, ...] | None = None,
        name: str = "",
        filepath: Path | str = "",
    ):
        super().__init__(problem.base)
        self._goal = goal if goal is not None else super().goal()
        self._initial = (
            initial if initial is not None else tuple(super().initial_literals)
        )
        self._name = (
            name or f"{super().name}_custom_{np.random.default_rng().integers(1000000)}"
        )
        self._filepath = filepath or (
            Path(tempfile.gettempdir())
            / (
                Path(super().filepath).stem
                + f"{hash((self._goal, self._initial, self._name))}.pddl"
            )
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def filepath(self) -> str:
        return self._filepath

    @cache
    def goal(self) -> XConjunctiveCondition:
        return self._goal

    @property
    def initial_literals(self) -> tuple[XLiteral, ...]:
        """
        Get the initial literals of the problem definition.
        """
        return self._initial

    @property
    def initial_atoms(self) -> tuple[XAtom, ...]:
        return tuple(l.atom for l in self._initial)

    def __hash__(self):
        return hash(
            (super().__hash__(), self._goal, self._initial, self._name, self._filepath)
        )


class XActionSchema(MimirWrapper[wrapmir.Action]):
    @property
    def index(self) -> int:
        """
        Get the index of the action schema in the domain.
        """
        return self.base.get_index()

    @property
    def name(self) -> str:
        return self.base.get_name()

    @property
    def arity(self) -> int:
        return self.base.get_arity()

    @property
    def condition(self) -> wrapmir.ConjunctiveCondition:
        return self.base.get_precondition()

    @cached_property
    def effects(self) -> list[wrapmir.ConditionalEffect]:
        return self.base.get_conditional_effect()

    def semantic_eq(self, other):
        return (
            self.name == other.name
            and self.arity == other.arity
            and self.condition == other.condition
            and self.effects == other.effects
        )

    def __str__(self):
        return str(self.base)


class XAction(MimirWrapper[wrapmir.GroundAction]):
    """
    Important Notes:
    A pymimir.GroundAction object is a mere pointer onto a ground action stored in an pymimir.ActionGrounder object.
    As such, it is not an independent object but a view onto memory that is NOT kept alive by pymimir.GroundAction objects

    A side effect is that actions that are semantically the same but are created with different grounders may not equal
    each other, because they are not the same object in memory. Keep this in mind when comparing actions.

    Guideline: Try to use only actions from the same ActionGenerator whenever possible, if comparisons are needed.
    """

    advanced = None

    def __init__(
        self,
        action: wrapmir.GroundAction | admir.GroundAction,
        problem: XProblem | None = None,
    ):
        if isinstance(action, admir.GroundAction):
            assert problem is not None, (
                f"Problem must be provided when passing type {type(admir.GroundAction)}"
            )
            action = wrapmir.GroundAction(action, problem.base)
            self.problem = problem
        else:
            assert isinstance(action, wrapmir.GroundAction), (
                f"Invalid passed action type to XAction: {type(action)}"
            )
            self.problem = XProblem(action.get_problem())
        super().__init__(action)
        self.advanced = self.base._advanced_ground_action

    @staticmethod
    def register(
        problem: XProblem, schema: XActionSchema | str, objects: Iterable[XObject | str]
    ) -> XAction:
        schema = (
            schema
            if isinstance(schema, XActionSchema)
            else problem.domain.action_lookup(schema)
        )
        objects = tuple(
            o if isinstance(o, XObject) else problem.object_lookup(o) for o in objects
        )
        return XAction(
            problem.advanced.ground(
                schema.base,
                admir.ObjectList(o.base for o in objects),
            ),
            problem,
        )

    def to_string(self, for_plan=True):
        return getattr(self.advanced, f"to_string{'_for_plan' if for_plan else ''}")(
            self.problem.advanced
        )

    @property
    def index(self):
        return self.base.get_index()

    @cached_property
    def action_schema(self) -> XActionSchema:
        return XActionSchema(self.advanced.get_action())

    @property
    def name(self) -> str:
        return self.action_schema.name

    def conditions(self) -> wrapmir.GroundConjunctiveCondition:
        return self.base.get_precondition()

    def effects(self) -> list[admir.GroundConditionalEffect]:
        return self.base.get_conditional_effect()

    @cached_property
    def objects(self):
        return tuple(map(XObject, self.advanced.get_objects()))

    def apply(
        self, state: XState, return_cost: bool = False
    ) -> XState | tuple[XState, float]:
        sr = self.problem.search_context.get_state_repository()
        adv_successor, cost = sr.get_or_create_successor_state(
            state.base, self.advanced, 0.0
        )

        successor = XState._fast_create(adv_successor, self.problem)
        if return_cost:
            return successor, cost
        else:
            return successor

    def semantic_eq(self, other):
        return self.action_schema.semantic_eq(other.action_schema) and all(
            a.name == b.name for a, b in zip(self.objects, other.objects)
        )

    def semantic_str(self):
        return str(self)

    def __str__(self):
        return self.to_string()


class StateLabel(Enum):
    goal = auto()
    deadend = auto()
    default = auto()


class XState(MimirWrapper[admir.State]):
    problem: XProblem

    """
    The extended state class.

    Important Notes:
    A pymimir.State object is a mere pointer onto a state stored in the state repository. As such, it is not an
    independent object but a view onto memory that is implicitly kept alive by each pymimir.State object
    (as guaranteed by pymimir binding logic).

    A side effect is that states that are semantically the same but are created in different repositories may not equal
    each other, because they are not the same object in memory. Keep this in mind when comparing states.

    Guideline: Try to use only states from the same repository whenever possible, i.e. use only a single repository.
    """

    def __init__(
        self,
        state: admir.State | wrapmir.State,
        problem: XProblem,
    ):
        if isinstance(state, wrapmir.State):
            state = state._advanced_state
        super().__init__(state)
        self.problem = problem

    @classmethod
    def _fast_create(cls, advanced_state, problem):
        # Bypass __init__ overhead for internal use
        instance = object.__new__(cls)
        instance._base = advanced_state
        instance.problem = problem
        return instance

    @staticmethod
    def register(
        problem: XProblem,
        fluents: Iterable[XAtom],
        numeric: Iterable[XFunctionValue] = (),
    ):
        (
            state,
            metric_value,
        ) = problem.search_context.get_state_repository().get_or_create_state(
            admir.FluentGroundAtomList(f.base for f in fluents),
            tuple(v.value for v in numeric),
        )
        return XState(state, problem)

    def __iter__(self):
        return iter(self.atoms())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.problem == other.problem:
            f = self.fluent_atoms
            y = other.fluent_atoms
            return y == f
        return False

    def __hash__(self):
        return hash(self.base)

    def __repr__(self):
        return f"XState({self.base}, {self.problem})"

    @property
    def static_atoms(self) -> tuple[XAtom, ...]:
        return self.problem.static_atoms

    @cached_property
    def fluent_atoms(self) -> tuple[XAtom, ...]:
        """
        Get the fluent atoms of the state.

        Delayed evaluation is used to avoid unnecessary computation in case this field is never required.
        """
        return tuple(
            map(
                XAtom,
                map(
                    self.problem.repositories.get_fluent_ground_atom,
                    self.base.get_fluent_atoms(),
                ),
            )
        )

    @cached_property
    def derived_atoms(self) -> tuple[XAtom, ...]:
        """
        Get the fluent atoms of the state.

        Delayed evaluation is used to avoid unnecessary computation in case this field is never required.
        """
        return tuple(
            map(
                XAtom,
                self.problem.repositories.get_derived_ground_atoms_from_indices(
                    tuple(self.base.get_derived_atoms())
                ),
            )
        )

    @cached_property
    def numerics(self) -> tuple[XFunctionValue, ...]:
        return tuple(
            itertools.starmap(
                XFunctionValue,
                zip(self.problem.ground_functions, self.base.get_numeric_variables()),
            )
        )

    @cached_property
    def numeric_dict(self) -> dict[XFunction, float]:
        return {fv.function: float(fv.value) for fv in self.numerics}

    def numeric_value(self, function: XFunction) -> float:
        try:
            return self.numeric_dict[function]
        except KeyError as e:
            raise KeyError(
                f"No numeric value found for function {function} in state {str(self)}."
            ) from e

    @property
    def index(self) -> int:
        """
        Get the index of the state in the state space. This index is unique for each state in the state space.
        """
        return self.base.get_index()

    def is_goal(
        self,
        goal: tuple[XLiteral, ...] | XConjunctiveCondition | None = None,
    ) -> bool:
        if goal is None:
            goal = self.problem.goal()
        return all(
            False
            for _ in chain(
                self.unsatisfied_literals(goal),
                self.unsatisfied_numeric_constraints(goal.numeric_constraints),
            )
        )

    @cached_property
    def is_initial(self) -> bool:
        initials = tuple(self.problem.initial_atoms)
        my_atoms = tuple(self.atoms(with_statics=False))
        return all(atom in initials for atom in my_atoms) and len(initials) == len(
            my_atoms
        )

    def atoms(self, with_statics: bool = True) -> Iterable[XAtom]:
        return chain(
            (self.static_atoms if with_statics else tuple()),
            self.fluent_atoms,
            self.derived_atoms,
        )

    def satisfied_literals(self, literals: Iterable[XLiteral]) -> Iterable[XLiteral]:
        return (lit for lit in literals if self.literal_holds(lit))

    def unsatisfied_literals(self, literals: Iterable[XLiteral]) -> Iterable[XLiteral]:
        return (lit for lit in literals if not self.literal_holds(lit))

    def literal_holds(self, literal: XLiteral) -> bool:
        return self.base.literal_holds(literal.base)

    def satisfied_numeric_constraints(
        self, constraints: Iterable[XNumericConstraint]
    ) -> Iterable[XNumericConstraint]:
        return (nc for nc in constraints if self.numeric_constraint_holds(nc))

    def unsatisfied_numeric_constraints(
        self, constraints: Iterable[XNumericConstraint]
    ) -> Iterable[XNumericConstraint]:
        return (nc for nc in constraints if not self.numeric_constraint_holds(nc))

    def numeric_constraint_holds(self, constraint: XNumericConstraint) -> bool:
        return constraint.holds(self)

    def semantic_eq(self, other: XState | Sequence[XAtom]):
        if not self.semantic_eq_subset(self.fluent_atoms, other):
            return False
        if not self.semantic_eq_subset(self.derived_atoms, other):
            return False
        if not self.semantic_eq_subset(self.static_atoms, other):
            return False
        return True

    @cache
    def semantic_str(self) -> str:
        return (
            f"{{"
            f"#fl={len(self.fluent_atoms)}, "  # eq-comp quick-rejection
            f"#dr={len(self.derived_atoms)}}}, "
            f"fl={sorted(map(str, self.fluent_atoms))}, "
            f"dr={sorted(map(str, self.derived_atoms))}"
            f"}}"
        )


class AtomChange(NamedTuple):
    added_fluents: tuple[XAtom, ...]
    added_deriveds: tuple[XAtom, ...]
    removed_fluents: tuple[XAtom, ...]
    removed_deriveds: tuple[XAtom, ...]


class XTransition:
    """
    A transition in a state space.

    Important Notes:
    A transition may be partially specified, i.e. it may only contain a source state and an action.
    In this case, the target state and cost are computed on demand when accessed the first time.
    In turn, if the action is None, the transition needs to have a target and cost specified.

    The cost of the transition reflects the planning problem's cost model. This may differ from what an RL reward
    specifies later on, i.e. it does not necessarily hold that reward = -cost, although this is a common choice.
    """

    source: XState
    action: XAction | tuple[XAction, ...] | None
    target: XState
    cost: float | tuple[float, ...] | None

    def __init__(
        self,
        source: XState,
        action: XAction | Sequence[XAction] | None,
        target: XState | None = None,
        cost: float | Sequence[float] | None = None,
    ):
        if action is None:
            assert (
                target is not None
            ), "If action is None, target and cost must be provided."
        elif target is None or cost is None:
            assert action is not None, (
                "if either target or cost is None, action must be provided."
            )
        self._source = source
        # Optimization: Check for XAction explicitly to avoid slow 'typing.Sequence' checks
        if isinstance(action, XAction) or action is None:
            self._action = action
        else:
            # Assume it's a sequence (list/tuple)
            self._action = tuple(action)
        self._target = target
        self._cost = cost

    def __iter__(self):
        return iter((self.source, self.action, self.target, self.cost))

    def __hash__(self):
        return hash((self.source, self.action, self.target, self.cost))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.source == other.source
            and self.action == other.action
            and self.target == other.target
            and self.cost == other.cost
        )

    def __len__(self):
        if self.action is None or isinstance(self.action, XAction):
            return 1
        return len(self.action)

    def __str__(self):
        src = self.source.index
        tgt = self.target.index

        if isinstance(self.action, XAction):
            act_str = str(self.action)
            cost_str = f"{self._cost}"
        elif isinstance(self.action, Sequence):
            total = sum(self._cost)
            if len(self.action) <= 3:
                parts = ", ".join(f"{c}" for c in self._cost)
                act_str = "; ".join(str(a) for a in self.action)
            else:
                # elide long sequences
                act_str = f"{self.action[0]}; {self.action[1]}; ...; {self.action[-1]}"
                parts = f"{self._cost[0]}, {self._cost[1]}, ..., {self._cost[-1]}"
            cost_str = f"{total} [{parts}]"
        elif self.action is None:
            act_str = "∅"
            cost_str = "?"
        else:
            raise ValueError(f"Unexpected action type in __str__: {type(self.action)}")

        return f"⟦{src} → {tgt}⟧ {act_str} | cost={cost_str}"

    @property
    def source(self):
        return self._source

    @property
    def action(self):
        return self._action

    @property
    def target(self) -> XState:
        if self._target is None:
            self._resolve_target()
        return self._target

    @property
    def cost(self):
        if self._cost is None and self._action is not None:
            self._resolve_target()
        return self._cost

    @cache
    def changed_atoms(self):
        fluents_before = set(self.source.fluent_atoms)
        fluents_after = set(self.target.fluent_atoms)
        derived_before = set(self.source.derived_atoms)
        derived_after = set(self.target.derived_atoms)
        added_fluents = fluents_after - fluents_before
        removed_fluents = fluents_before - fluents_after
        added_derived = derived_after - derived_before
        removed_derived = derived_before - derived_after
        return AtomChange(
            tuple(added_fluents),
            tuple(added_derived),
            tuple(removed_fluents),
            tuple(removed_derived),
        )

    def _resolve_target(self):
        match self.action:
            case None:
                action = self.action
                raise ValueError(
                    f"Cannot determine target state without action. Got {action=}"
                )
            case XAction():
                successor_gen = XSuccessorGenerator(self.source.problem)
                self._target, self._cost = successor_gen.successor(
                    self.source, self.action
                )
            case collections.abc.Sequence():
                successor_gen = XSuccessorGenerator(self.source.problem)
                target = self.source
                cost = []
                for act in self.action:
                    target, transition_cost = successor_gen.successor(target, act)
                    cost.append(transition_cost)
                self._target = target
                self._cost = tuple(cost)

    def to_string(self, detailed: bool = False):
        if not detailed:
            return str(self)
        if isinstance(self.action, XAction):
            action_string = str(self.action)
        elif isinstance(self.action, Sequence):
            action_string = list(map(str, self.action))
        else:
            action_string = "None"
        return (
            f"Transition(\n"
            f"\tfrom:   [{', '.join(map(str, chain(self.source.fluent_atoms, self.source.derived_atoms)))}]\n"
            f"\taction: {action_string}\n"
            f"\tto:     [{', '.join(map(str, chain(self.target.fluent_atoms, self.target.derived_atoms)))}]\n"
            f"\tcost:   {self.cost}\n"
            f")"
        )

    def explain(self) -> str:
        atom_change = self.changed_atoms()
        return (
            f"Transition({self.source.index} -> {self.target.index})\n"
            f"Action: {self.action}\n"
            f"Cost:   {self.cost}\n"
            f"Added Fluent Atoms:    {atom_change.added_fluents}\n"
            f"Added Derived Atoms:   {atom_change.added_deriveds}\n"
            f"Deleted Fluents Atoms: {atom_change.removed_fluents}\n"
            f"Removed Derived Atoms: {atom_change.removed_deriveds}"
        )

    @property
    def is_informed(self):
        return self.action is not None

    @property
    def is_primitive(self):
        return isinstance(self.action, XAction)


class XActionGenerator(MimirWrapper[admir.IApplicableActionGenerator]):
    def __init__(
        self,
        generator: admir.IApplicableActionGenerator,
        problem: XProblem,
    ):
        super().__init__(generator)
        self.problem = problem

    def generate_actions(self, state: XState) -> Generator[XAction, None, None]:
        yield from map(
            lambda a: XAction(a, self.problem),
            self.base.generate_applicable_actions(state.base),
        )

    __hash__ = None


class XSuccessorGenerator(MimirWrapper[admir.SearchContext]):
    problem: XProblem
    action_generator: XActionGenerator

    @multimethod
    def __init__(self, problem: XProblem):
        self.problem = problem
        context = problem.search_context
        super().__init__(context)
        self.action_generator = XActionGenerator(
            context.get_applicable_action_generator(), problem
        )

    @property
    def state_repository(self):
        return self.problem.state_repository

    @property
    def initial_state(self) -> XState:
        return XState(
            self.state_repository.get_or_create_initial_state()[0], self.problem
        )

    def successor(self, state: XState, action: XAction) -> tuple[XState, float]:
        return action.apply(state, return_cost=True)

    def successors(self, state: XState) -> Iterator[XTransition]:
        for action in self.action_generator.generate_actions(state):
            yield XTransition(state, action, *self.successor(state, action))

    def semantic_eq(self, other):
        raise NotImplementedError(
            "Semantic equality is not implemented for successor generators."
        )

    def __str__(self):
        return f"XSuccessorGenerator({self.problem})"

    def __repr__(self):
        return f"<{self.__class__.__qualname__} object at {id(self)}>"

    __hash__ = None


class XSearchResult(MimirWrapper[wrapmir.SearchResult]):
    start: XState

    def __init__(self, result: wrapmir.SearchResult, start: XState):
        super().__init__(result)
        self.start = start

    def __len__(self):
        return len(self.plan or tuple())

    def status(self):
        return self.base.status

    def goal(self):
        return XState(self.base.goal_state, self.start.problem)

    @cached_property
    def plan(self) -> tuple[XAction, ...] | None:
        if self.base.solution is not None:
            plan = self.base.solution
            transitions = []
            state = self.start
            for action in map(XAction, plan):
                next_state, cost = action.apply(state, return_cost=True)
                transitions.append(XTransition(state, action, next_state, cost))
                state = next_state
            return tuple(transitions)
        return None

    @property
    def cost(self) -> float | None:
        return self.base.solution_cost

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.base == other.base and self.start == other.start

    __hash__ = None


class XOptionsBase(MimirWrapper[T]):
    def __init__(self, **options):
        opts = self.base_type()()
        super().__init__(opts)
        for option, value in options.items():
            if hasattr(opts, option):
                setattr(opts, option, value)
            else:
                raise ValueError(
                    f"Unknown option '{option}' for {opts.__class__.__name__}."
                )

    @cache
    def __str__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                f"{attr}={getattr(self.base, attr)}"
                for attr, v in self.base.__class__.__dict__.items()
                if "<property" in str(v)
            )
            + ")"
        )


class XStateSpaceOptions(XOptionsBase[admir.StateSpaceOptions]): ...


class XSearchContextOptions(XOptionsBase[admir.SearchContextOptions]): ...


class XTupleGraphOptions(XOptionsBase[admir.TupleGraphOptions]): ...


class XKnowledgeBaseOptions(XOptionsBase[admir.KnowledgeBaseOptions]): ...


class XParserOptions(XOptionsBase[admir.ParserOptions]): ...


class XStateSpace(MimirWrapper[wrapmir.StateSpaceSampler], Sequence[XState]):
    """
    The extended state space class.

    Note that we implicitly rely on the underlying state space to be created with its own state repository.
    Mixing states from different repositories will lead to different states potentially holding the same index.
    This could mess up a lot of downstream operations relying on this index to be unique for a state, i.e. for a certain
    list of atoms in a problem, and not for a certain emplacement order.
    We also rely on the state space to emplace states in a fixed, deterministic order that always produces the same
    state at the same index.
    """

    @multimethod
    def __init__(self, space: wrapmir.StateSpaceSampler):
        super().__init__(space)
        self._problem = XProblem(self.base.get_problem())
        # take a reference here in case something changes the search context in the problem
        self._search_context = self.problem.search_context
        self._graph: admir.BidirectionalStaticProblemGraph = (
            self.advanced.get_state_space().get_graph()
        )

    @multimethod
    def __init__(self, space: admir.StateSpace):
        problem = XProblem(space.get_search_context())
        space = admir.StateSpaceSampler(space)
        wrapped_space = wrapmir.StateSpaceSampler(space, problem.base)
        XStateSpace.__init__(self, wrapped_space)

    @multimethod
    def __init__(
        self,
        domain_path: str | Path,
        problem_path: str | Path,
        /,
        lifted: bool = True,
        **options,
    ):
        problem = XProblem(XDomain.parse(domain_path), problem_path)
        mode = (
            admir.LiftedOptions(admir.LiftedKPKCOptions())
            if lifted
            else admir.GroundedOptions()
        )
        problem.search_context = admir.SearchContext.create(
            problem.advanced, admir.SearchContextOptions(mode=mode)
        )
        XStateSpace.__init__(
            self,
            problem,
            **options,
        )

    @multimethod
    def __init__(
        self,
        problem: XProblem,
        **options,
    ):
        space = admir.StateSpace.create(
            problem.search_context, XStateSpaceOptions(**options).base
        )
        if space is None:
            raise ValueError(
                "State space creation failed. "
                "This is likely due to the options provided to the constructor. "
                "Please check the options and try again."
            )
        space = admir.StateSpaceSampler(space[0])
        XStateSpace.__init__(self, wrapmir.StateSpaceSampler(space, problem.base))

    def __len__(self):
        return self.base.num_states()

    def __str__(self):
        return (
            f"StateSpace("
            f"#states={len(self)}, "
            f"#transitions={self.total_transition_count}), "
            f"#deadends={self.deadend_count}, "
            f"#goals={self.goal_count}, "
            f"solvable={self.solvable}, "
            f"solution_cost={self.goal_cost(self.initial_state)}"
        )

    @property
    def advanced(self):
        return self.base._advanced_state_space_sampler

    @property
    def advanced_space(self):
        return self.advanced.get_state_space()

    def string(self):
        return str(self.base)

    def __getitem__(self, index: int | slice) -> list[XState] | XState:
        """
        Get the state to the given index.

        Note that this method is not optimized for slicing, as it will always return a list of states.
        Also, we rely on the index to semantically mean the nth state in the repository.
        That is,
            `index == space[state].index`
        """
        if isinstance(index, slice):  # handle slicing
            return [self.get_state(i) for i in range(*index.indices(len(self)))]
        return self.get_state(index)

    def __iter__(self) -> Iterator[XState]:
        return self.states_iter()

    @property
    def problem(self) -> XProblem:
        return self._problem

    @property
    def graph(self) -> admir.BidirectionalStaticProblemGraph:
        return self._graph

    @property
    def search_context(self) -> admir.SearchContext:
        return self._search_context

    @property
    def state_repository(self) -> admir.StateRepository:
        return self.search_context.get_state_repository()

    @cached_property
    def action_generator(self) -> XActionGenerator:
        return XActionGenerator(
            self.search_context.get_applicable_action_generator(), self.problem
        )

    @cached_property
    def successor_generator(self) -> XSuccessorGenerator:
        return XSuccessorGenerator(self.problem)

    @property
    def repositories(self) -> admir.Repositories:
        return self.problem.repositories

    @property
    def solvable(self) -> bool:
        return not self.is_deadend(self.initial_state)

    @property
    def deadend_count(self) -> int:
        return len(self.advanced_space.get_unsolvable_vertices())

    @property
    def goal_count(self) -> int:
        return len(self.advanced_space.get_goal_vertices())

    @property
    def total_transition_count(self) -> int:
        return self.graph.get_num_edges()

    def is_deadend(self, state: XState) -> bool:
        return state.index in self.advanced_space.get_unsolvable_vertices()

    def is_goal(self, state: XState) -> bool:
        return state.index in self.advanced_space.get_goal_vertices()

    def states_iter(self) -> Iterator[XState]:
        return iter(self.get_state(v) for v in range(0, len(self)))

    def goal_states_iter(self) -> Iterator[XState]:
        return iter(
            self.get_state(idx) for idx in self.advanced_space.get_goal_vertices()
        )

    def deadend_states_iter(self) -> Iterator[XState]:
        return iter(
            self.get_state(idx) for idx in self.advanced_space.get_unsolvable_vertices()
        )

    def goal_distance(self, state: XState) -> float:
        return self.advanced.get_steps_to_goal(state.base)

    def goal_cost(self, state: XState) -> float:
        return self.advanced.get_cost_to_goal(state.base)

    def get_state(self, index: int) -> XState:
        return XState(self.advanced.get_state(index), self.problem)

    @cached_property
    def initial_state(self) -> XState:
        return self.get_state(self.advanced_space.get_initial_vertex())

    def forward_transitions(self, state: XState) -> Iterator[XTransition]:
        return iter(
            XTransition(
                source=state,
                action=XAction(admir.get_action_from_edge(edge), self.problem),
                target=XState(
                    admir.get_state_from_vertex(next_state_vertex), self.problem
                ),
                cost=admir.get_action_cost_from_edge(edge),
            )
            for edge, next_state_vertex in zip(
                self.graph.get_forward_adjacent_edges(state.index),
                self.graph.get_forward_adjacent_vertices(state.index),
            )
        )

    def backward_transitions(self, state: XState) -> Iterator[XTransition]:
        return iter(
            XTransition(
                source=XState(
                    admir.get_state_from_vertex(source_state_vertex), self.problem
                ),
                action=XAction(admir.get_action_from_edge(edge), self.problem),
                target=state,
                cost=admir.get_action_cost_from_edge(edge),
            )
            for edge, source_state_vertex in zip(
                self.graph.get_backward_adjacent_edges(state.index),
                self.graph.get_backward_adjacent_vertices(state.index),
            )
        )

    def forward_transition_count(self, state: XState) -> int:
        return self._transition_count(state, "forward")

    def backward_transition_count(self, state: XState) -> int:
        return self._transition_count(state, "backward")

    def _transition_count(self, state: XState, direction: str) -> int:
        return sum(
            1
            for _ in getattr(self.advanced, f"get_{direction}_transitions")(state.base)
        )

    def a_star_search(
        self,
        target: XState,
        start: XState | None = None,
        *,
        heuristic: Mapping[int, float] | None = None,
    ) -> XSearchResult:
        """
        A* from the initial state to `target`.

        Parameters
        ----------
        target: XState,
            The target state to reach.
        start: XState | None,
            The state to start the search from. If None, the initial state of the problem is used.
        heuristic: Mapping[int, float] | None,
            A mapping from state indices to their heuristic costs.
            If None, distances are computed using the base's compute_shortest_backward_distances_from_states method.
        Returns
        -------
        list[Transition],
            A list of transitions leading from the start state to the target state.
        """

        start = start or self.initial_state
        match heuristic:
            case None:

                class OptimalHeuristic(wrapmir.Heuristic):
                    def __init__(self, space):
                        self.values = space.shortest_distances_from_state(
                            target, forward=False
                        )
                        super().__init__()

                    def compute_value(
                        self, state: admir.State, is_goal_state: bool
                    ) -> float:
                        return self.values[state.get_index()]

                    def get_preferred_actions(
                        self,
                    ) -> set[wrapmir.GroundAction]:
                        raise NotImplementedError("This interface is not clear.")

                heuristic = OptimalHeuristic(self)

            case Mapping():

                class CustomHeuristic(wrapmir.Heuristic):
                    def __init__(self, mapping: Mapping[int, float]):
                        self.mapping = mapping
                        super().__init__()

                    def compute_value(
                        self, state: admir.State, is_goal_state: bool
                    ) -> float:
                        return self.mapping.get(state.get_index(), float("inf"))

                    def get_preferred_actions(
                        self,
                    ) -> set[wrapmir.GroundAction]:
                        raise NotImplementedError("This interface is not clear.")

                heuristic = CustomHeuristic(heuristic)
            case _:
                raise ValueError("Heuristic must be a mapping or None.")

        return XSearchResult(
            astar_eager(
                problem=self.problem.base,
                start_state=wrapmir.State(start.base, self.problem.base),
                heuristic=heuristic,
            ),
            start=start,
        )

    @lru_cache(maxsize=100)
    def shortest_distances_from_state(
        self, state: int | XState, forward: bool = True
    ) -> list[float]:
        """
        Compute the shortest forward distances from the given state or state index to all other states in the space.

        Parameters
        ----------
        state: XState | int,
            The state or state index to compute distances from.
        forward: bool,
            Whether to compute forward (True) or backward (False) distances.

        Returns
        -------
        list[float],
            A mapping of shortest forward distances from state to all state indices.
        """
        if isinstance(state, XState):
            state = state.index
        # [1] is the distance map for every state-index in the space
        if forward:
            return self.graph.compute_forward_breadth_first_search([state])[1]
        else:
            return self.graph.compute_backward_breadth_first_search([state])[1]


class XTupleGraph(MimirWrapper[admir.TupleGraph]):
    class TupleNode(NamedTuple):
        distance: int
        states: frozenset[XState]
        atom_tuple: tuple[XAtom, ...]

        def __str__(self):
            state_indices = ", ".join(map(str, self.states))
            atoms = ", ".join(map(str, self.atom_tuple))
            return f"TupleNode(distance={self.distance}, states={{{state_indices}}}, atom-tuple={{{atoms}}})"

        def __repr__(self):
            return str(self)

    def __init__(self, graph: admir.TupleGraph):
        super().__init__(graph)
        self._problem = XProblem(graph.get_state_space().get_search_context())
        self._vertex_indices_by_distance: list[list[int]] = (
            self.base.get_tuple_vertex_indices_grouped_by_distance()
        )
        self.max_distance = len(self._vertex_indices_by_distance) - 1
        self._unprocessed_distances = set(range(len(self._vertex_indices_by_distance)))
        self._atom_tuple_mapping: dict[tuple[XAtom, ...], XTupleGraph.TupleNode] = {}

    def __str__(self):
        return str(self.base)

    @property
    def problem(self) -> XProblem:
        return self._problem

    @property
    def graph(self):
        return self.base.get_graph()

    @cached_property
    def state_space(self) -> XStateSpace:
        return XStateSpace(self.base.get_state_space())

    @property
    def atom_tuple_map(self) -> dict[tuple[XAtom, ...], XTupleGraph.TupleNode]:
        if self._unprocessed_distances:
            # copy to avoid modification during iteration
            for distance in self._unprocessed_distances.copy():
                self.nodes_by_distance(distance)
        return self._atom_tuple_mapping

    def atom_tuples(self) -> Iterable[tuple[XAtom, ...]]:
        return self.atom_tuple_map.keys()

    def nodes(self) -> Iterable[XTupleGraph.TupleNode]:
        return self.atom_tuple_map.values()

    def __iter__(self):
        for distance in range(len(self._vertex_indices_by_distance)):
            yield from self.nodes_by_distance(distance)

    def vertex_indices_by_distance(self, distance: int) -> list[int]:
        return self._vertex_indices_by_distance[distance]

    @cache
    def nodes_by_distance(self, distance: int) -> list[TupleNode]:
        space = self.state_space
        graph = self.graph
        nodes = []
        for tuple_vertex_id in self.vertex_indices_by_distance(distance):
            tuple_vertex = graph.get_vertex(tuple_vertex_id)
            atom_tuple = tuple(
                map(
                    XAtom,
                    self.problem.advanced.get_repositories().get_fluent_ground_atoms_from_indices(
                        admir.get_atom_tuple_from_vertex(tuple_vertex)
                    ),
                )
            )
            states = frozenset(
                space[prob_vertex]
                for prob_vertex in admir.get_problem_vertices_from_graph(tuple_vertex)
            )
            node = XTupleGraph.TupleNode(
                distance=distance, states=states, atom_tuple=atom_tuple
            )
            self._atom_tuple_mapping[atom_tuple] = node
            nodes.append(node)
        self._unprocessed_distances.discard(distance)
        return nodes


class XTupleSpace(XStateSpace):
    """
    A mapping from XState to its associated XTupleGraph, with tuple-graph transitions and distances.
    """

    class TGTransition(XTransition):
        atom_tuple: tuple[XAtom, ...]

        def __init__(self, *args, atom_tuple: tuple[XAtom, ...], **kwargs):
            super().__init__(*args, **kwargs)
            self.atom_tuple = atom_tuple

        def __str__(self):
            base_str = super().__str__()
            atoms = ", ".join(map(str, self.atom_tuple))
            return f"{base_str} | atom-tuple={{{atoms}}}"

    class StateInfo(NamedTuple):
        distance_to_goal: int
        distance_from_initial: int

    def __init__(
        self, tuple_graphs: Sequence[XTupleGraph], options: XTupleGraphOptions
    ):
        if len(tuple_graphs) == 0:
            raise ValueError("At least one tuple graph must be provided.")
        space = tuple_graphs[0].state_space
        if any(
            tg.state_space.advanced_space != space.advanced_space for tg in tuple_graphs
        ):
            raise ValueError("All tuple graphs must belong to the same state space.")
        super().__init__(space)
        self.options = options
        self._graphs = tuple(tuple_graphs)
        self._state_to_graph = dict(zip(self, self._graphs))
        self._fwd_transitions: dict[XState, list[XTupleSpace.TGTransition]] = {}
        self._bkwd_transitions: dict[XState, list[XTupleSpace.TGTransition]] = {}
        self._state_info: dict[XState, XTupleSpace.StateInfo] = {}
        # build all tuple-graph transitions and distances now
        self._compute_transitions()

    def __str__(self):
        return (
            f"TupleGraphSpace("
            f"width={self.options.base.width}, "
            f"#states={len(self)}, "
            f"#transitions={self.total_transition_count}, "
            f"#deadends={self.deadend_count}, "
            f"#goals={self.goal_count}, "
            f"solvable={self.solvable}, "
            f"solution_cost={self.goal_distance(self.initial_state)}"
            f")"
        )

    @staticmethod
    def from_problem(problem: XProblem, **options):
        options["width"] = options.get("width", 1)
        options = XTupleGraphOptions(**options)
        kb = XKnowledgeBase(problem, tuple_graph_options=options.base)
        return kb.get_tuple_space(problem)

    @property
    def tuple_graphs(self) -> tuple[XTupleGraph, ...]:
        return self._graphs

    def get_tuple_graph(self, state: XState | int) -> XTupleGraph:
        return self._state_to_graph[self[state] if isinstance(state, int) else state]

    def _compute_transitions(self):
        """
        Computes tuple-graph transitions between states and augments each transition with
        the action-DAG of *all* minimum-cost primitive paths in the underlying state-space graph.

        Also computes, in terms of tuple-graph jumps (not primitive actions):
          - distance_from_initial: minimal # of TG jumps from the initial state,
          - distance_to_goal: minimal # of TG jumps to *any* goal state.
        """

        tg_forward_transitions: dict[XState, dict[XState, list[XTransition]]] = {
            s: dict() for s in self
        }
        for state, tuple_graph in self._state_to_graph.items():
            transition_dict = tg_forward_transitions[state]
            frontier = deque(
                [(0, [], [], state)]
            )  # (distance, action_list, cost_list, state)
            while frontier:
                dist, actions, costs, frontier_state = frontier.popleft()
                next_dist = dist + 1
                if next_dist > tuple_graph.max_distance:
                    continue
                next_nodes = tuple_graph.nodes_by_distance(next_dist)
                for transition in super().forward_transitions(
                    frontier_state
                ):  # primitive transitions
                    for node in next_nodes:
                        if transition.target in node.states:
                            if transition.target not in transition_dict:
                                transition_dict[transition.target] = []
                            next_actions = actions + [transition.action]
                            next_costs = costs + [transition.cost]
                            transition_dict[transition.target].append(
                                self.TGTransition(
                                    source=state,
                                    action=next_actions,
                                    target=transition.target,
                                    cost=next_costs,
                                    atom_tuple=node.atom_tuple,
                                )
                            )
                            frontier.append(
                                (next_dist, next_actions, next_costs, transition.target)
                            )

        # Materialize local transitions into the object-level forward/backward maps
        # (We intentionally allow multiple TGTransition objects per (source,target)
        #  when they correspond to different atom-tuples/paths.)
        for s, tmap in tg_forward_transitions.items():
            out_list: list[XTupleSpace.TGTransition] = []
            for _, trs in tmap.items():
                out_list.extend(trs)
            self._fwd_transitions[s] = out_list
            # ensure backward list exists
            self._bkwd_transitions[s] = self._bkwd_transitions.get(s, [])

        # Build backward lists from forward lists
        for s, trs in self._fwd_transitions.items():
            for tr in trs:
                self._bkwd_transitions[tr.target].append(tr)

        # Compute TG distances (BFS over TG edges)
        # Build adjacency lists from the recorded TG transitions
        adj_fwd: dict[int, list[int]] = {
            s.index: [tr.target.index for tr in trs]
            for s, trs in self._fwd_transitions.items()
        }
        adj_bwd: dict[int, list[int]] = {s.index: [] for s in self}
        for s, trs in self._fwd_transitions.items():
            for tr in trs:
                adj_bwd[tr.target.index].append(s.index)

        # Distances from initial (in TG jumps)
        init = self.initial_state
        dist_from_init = {init.index: 0}
        q = [init.index]
        while q:
            u = q.pop(0)
            for v in adj_fwd.get(u, []):
                if v not in dist_from_init:
                    dist_from_init[v] = dist_from_init[u] + 1
                    q.append(v)

        # Distances to goal (in TG jumps) via backward BFS from all goal states
        dist_to_goal: dict[int, int] = {}
        q = []
        for g in self.goal_states_iter():
            dist_to_goal[g.index] = 0
            q.append(g.index)
        while q:
            v = q.pop(0)
            for u in adj_bwd.get(v, []):
                if u not in dist_to_goal:
                    dist_to_goal[u] = dist_to_goal[v] + 1
                    q.append(u)

        # Store per-state info
        for s in self:
            d_goal = dist_to_goal.get(s.index, -1)
            d_init = dist_from_init.get(s.index, -1)
            self._state_info[s] = self.StateInfo(
                distance_to_goal=d_goal, distance_from_initial=d_init
            )

    def forward_transitions(self, state: XState):
        return self._fwd_transitions[state]

    def backward_transitions(self, state: XState):
        return self._bkwd_transitions[state]

    @cached_property
    def total_transition_count(self) -> int:
        return sum(len(v) for v in self._fwd_transitions.values())

    def goal_distance(self, state: XState) -> int:
        return self._state_info[state].distance_to_goal

    def distance_from_initial(self, state: XState) -> int:
        return self._state_info[state].distance_from_initial

    def tuple_goal_distance(
        self, state: XState, *atom_tuples: tuple[XAtom, ...]
    ) -> Generator[int, None, None]:
        """Return minimal tuple-jumps from the given atom-tuple (in the TG rooted at `state`) to **any** tuple that includes a goal state."""
        yield from map(
            lambda x: min(x, default=-1),
            self.tuple_goal_jump_distances(state, *atom_tuples),
        )

    def tuple_goal_jump_distances(
        self, state: XState, *atom_tuples: tuple[XAtom, ...]
    ) -> Generator[list[int], None, None]:
        """All distinct tuple-jump distances from `atom_tuple` (in TG rooted at `state`) to any goal-containing tuple.
        Returns an empty tuple if that atom-tuple was not found in the state.
        """
        tuple_map = self._state_to_graph[state].atom_tuple_map
        for atom_tuple in atom_tuples:
            nodes = tuple_map.get(atom_tuple, None)
            if nodes is None:
                yield []
            yield list(self.goal_distance(s) for s in nodes.states)


class XKnowledgeBase(MimirWrapper[admir.KnowledgeBase]):
    def __init__(self, *problems: XProblem, **options):
        if len(problems) == 0:
            raise ValueError("At least one problem must be provided.")
        domain = problems[0].domain
        if any(p.domain != domain for p in problems):
            raise ValueError("All problems must belong to the same domain.")
        gp = admir.GeneralizedProblem.create(
            domain.advanced, admir.ProblemList(p.advanced for p in problems)
        )
        contexts = [p.search_context for p in problems]
        gsc = admir.GeneralizedSearchContext.create(gp, contexts)
        kb_options = XKnowledgeBaseOptions(**options).base
        kb = admir.KnowledgeBase.create(gsc, kb_options)
        super().__init__(kb)
        self.kb_options = kb_options
        self.problems = problems
        self.tuple_graphs = tuple(
            tuple(map(XTupleGraph, prob_graph_list))
            for prob_graph_list in self.base.get_tuple_graphs()
        )
        self.state_spaces = tuple(map(XStateSpace, self.base.get_state_spaces()))
        self._problem_wise = {
            p: (self.state_spaces[i], self.tuple_graphs[i])
            for i, p in enumerate(self.problems)
        }

    def get_tuple_space(self, problem: XProblem) -> XTupleSpace:
        return XTupleSpace(
            self._problem_wise[problem][1], self.kb_options.tuple_graph_options
        )

    def get_state_space(self, problem: XProblem) -> XStateSpace:
        return self._problem_wise[problem][0]


__all__ = [
    "XActionGenerator",
    "XTransition",
    "XTupleGraph",
    "XTupleSpace",
    "XStateSpace",
    "XStateSpaceOptions",
    "XSearchContextOptions",
    "XTupleGraphOptions",
    "XKnowledgeBaseOptions",
    "XKnowledgeBase",
    "XParserOptions",
    "XProblem",
    "XDomain",
    "XState",
    "StateLabel",
    "XLiteral",
    "XConjunctiveCondition",
    "XLiftedConjunctiveCondition",
    "XActionSchema",
    "XExpr",
    "XLiftedExpr",
    "XFunction",
    "XLiftedFunction",
    "XFunctionSkeleton",
    "XExprOperator",
    "XLiftedExprOperator",
    "XExprNumber",
    "XNumericConstraint",
    "XLiftedNumericConstraint",
    "XAtom",
    "XObject",
    "XPredicate",
    "XCategory",
    "XAction",
    "XFunctionValue",
    "XFunction",
    "XSuccessorGenerator",
    "XSearchResult",
    "XNumericConstraint",
    "atom_str_template",
]
