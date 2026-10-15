# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from typing import Union

from .pymimir_advanced import *

Atom = Union[StaticAtom, FluentAtom, DerivedAtom]
GroundAtom = Union[
    StaticGroundAtom,
    FluentGroundAtom,
    DerivedGroundAtom,
]
Literal = Union[
    StaticLiteral,
    FluentLiteral,
    DerivedLiteral,
]
GroundLiteral = Union[
    DerivedGroundLiteral,
    FluentGroundLiteral,
    StaticGroundLiteral,
]
Predicate = Union[
    StaticPredicate,
    FluentPredicate,
    DerivedPredicate,
]
FunctionSkeleton = Union[
    StaticFunctionSkeleton,
    FluentFunctionSkeleton,
    AuxiliaryFunctionSkeleton,
]
GroundFunction = Union[
    StaticGroundFunction,
    FluentGroundFunction,
    AuxiliaryGroundFunction,
]
Function = Union[
    StaticFunction,
    FluentFunction,
    AuxiliaryFunction,
]
GroundFunctionExpressionFunction = Union[
    StaticGroundFunctionExpressionFunction,
    FluentGroundFunctionExpressionFunction,
    AuxiliaryGroundFunctionExpressionFunction,
]
FunctionExpressionFunction = Union[
    StaticFunctionExpressionFunction,
    FluentFunctionExpressionFunction,
    AuxiliaryFunctionExpressionFunction,
]
FunctionExpressionOperator = Union[
    FunctionExpressionBinaryOperator,
    FunctionExpressionMultiOperator,
    FunctionExpressionMinus,
]
GroundFunctionExpressionOperator = Union[
    GroundFunctionExpressionBinaryOperator,
    GroundFunctionExpressionMultiOperator,
    GroundFunctionExpressionMinus,
]
GroundFunctionValue = Union[
    StaticGroundFunctionValue,
    FluentGroundFunctionValue,
    AuxiliaryGroundFunctionValue,
]

__all__ = [
    "Atom",
    "GroundAtom",
    "Literal",
    "GroundLiteral",
    "Predicate",
    "GroundFunctionValue",
    "GroundFunction",
    "FunctionSkeleton",
    "Function",
    "FunctionExpressionFunction",
    "FunctionExpressionOperator",
    "GroundFunctionExpressionFunction",
    "GroundFunctionExpressionOperator",
]
