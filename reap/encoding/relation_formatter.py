# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from __future__ import annotations

from functools import cache, singledispatchmethod
from typing import Protocol, runtime_checkable

from reap.utils.singleton import PickleSafeSingleton
from reap.xmimir import (
    XAction,
    XActionSchema,
    XAtom,
    XLiteral,
    XObject,
    atom_str_template,
)

Node = str


@runtime_checkable
class RelationProto(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def arity(self) -> int: ...


class RelationFormatter(metaclass=PickleSafeSingleton):
    positive_prefix = "[+]"
    negative_prefix = "[-]"
    goal_suffix = "[g]"
    subgoal_suffix = "[sg]"
    subsubgoal_suffix = "[ssg]"
    subsubsubgoal_suffix = "[sssg]"
    goal_satisfied_suffix = "[sat]"
    goal_unsatisfied_suffix = "[unsat]"
    goal_satisfied_added_suffix = "[sat+]"
    goal_satisfied_removed_suffix = "[sat-]"

    goal_level_suffixes = {
        0: goal_suffix,
        1: subgoal_suffix,
        2: subsubgoal_suffix,
        3: subsubsubgoal_suffix,
        None: "",
    }
    polarity_prefixes = {
        True: positive_prefix,
        False: negative_prefix,
        None: "",
    }
    goal_satisfaction_suffixes = {
        True: goal_satisfied_suffix,
        False: goal_unsatisfied_suffix,
        "+": goal_satisfied_added_suffix,
        "-": goal_satisfied_removed_suffix,
        None: "",
    }

    default_nullary_symbol_name = "![nullary_symbol]!"

    @singledispatchmethod
    def __call__(self, item, *args, **kwargs) -> Node | None:
        # fallback for anything that implements RelationProto
        if hasattr(item, "name") and hasattr(item, "arity"):
            goal_level = kwargs.get("goal_level")
            goal_satisfaction = kwargs.get("goal_satisfaction")
            polarity = kwargs.get("polarity")
            return (
                f"{self.polarity_prefixes[polarity]}"
                f"{item.name}"
                f"{self.goal_level_suffixes[goal_level]}"
                f"{self.goal_satisfaction_suffixes[goal_satisfaction]}"
            )
        raise NotImplementedError(
            "__call__ is not implemented for type {}".format(type(item))
        )

    @__call__.register
    def atom(
        self,
        atom: XAtom,
        pos: int | None = None,
        *args,
        **kwargs,
    ) -> Node | None:
        if pos is None:
            return str(atom)
        return f"{atom}:{pos}"

    @__call__.register
    def predicate(
        self,
        predicate: RelationProto,
        *,
        goal_level: int | None = None,
        goal_satisfaction: bool | str | None = None,
        polarity: bool = None,
        **kwargs,
    ) -> Node | None:
        return (
            f"{self.polarity_prefixes[polarity]}"
            f"{predicate.name}"
            f"{self.goal_level_suffixes[goal_level]}"
            f"{self.goal_satisfaction_suffixes[goal_satisfaction]}"
        )

    @cache
    @__call__.register
    def literal(
        self,
        literal: XLiteral,
        pos: int | None = None,
        *args,
        goal_level: int | None = None,
        goal_satisfaction: bool = None,
        **kwargs,
    ) -> Node | None:
        return (
            f"{literal}"
            f"{self.goal_level_suffixes[goal_level]}"
            f"{self.goal_satisfaction_suffixes[goal_satisfaction]}"
            f"{f':{pos}' if pos is not None else ''}"
        )

    @__call__.register
    def action_schema(
        self,
        action: XActionSchema,
        **kwargs,
    ) -> Node | None:
        return action.name

    @cache
    @__call__.register
    def action(
        self,
        action: XAction,
        **kwargs,
    ) -> Node | None:
        # use the same formatting as for atoms
        return atom_str_template.render(predicate=action.name, objects=action.objects)

    @__call__.register
    def object(self, obj: XObject, *args, **kwargs) -> Node | None:
        return obj.name

    @__call__.register
    def str_(self, s: str, *args, **kwargs) -> Node | None:
        return s

    @__call__.register
    def none(self, none: None, *args, **kwargs) -> Node | None:
        return str(None)


relation_formatter = RelationFormatter()
