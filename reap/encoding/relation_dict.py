# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

import itertools
import operator
from typing import Collection, Container, Mapping

from .relation_formatter import Node, RelationProto, relation_formatter


class RelationDict(Mapping[Node, int]):
    default_object_predicate: str = "object"
    default_number_predicate: str = "number"
    default_symbol_ntype: str = "_symbol_"
    default_action_ntype: str = "_action_"

    def __init__(
        self,
        predicates: tuple[RelationProto, ...],
        actions: tuple[RelationProto, ...] = (),
        *,
        max_goal_level: int = 0,
        top_type_predicates: Container[str] = (
            default_object_predicate,
            default_number_predicate,
            default_symbol_ntype,
            default_action_ntype,
        ),
        goal_satisfaction_derivations: Collection[str | bool] | None = None,
        support_literals: bool = False,
    ):
        max_goal_level = max(0, max_goal_level)
        assert max_goal_level <= len(relation_formatter.goal_level_suffixes) - 1, (
            f"max_goal_level exceeded the supported limit ("
            f"{max_goal_level=}, limit={len(relation_formatter.goal_level_suffixes) - 1}."
        )
        regular_predicates = list(
            filter(lambda p: p.name not in top_type_predicates, predicates)
        )
        # all variations of predicates for goal levels and polarities
        all_variations = lambda: itertools.product(
            regular_predicates,
            itertools.chain(range(max_goal_level + 1), [None] * support_literals),
            (True, False),
        )
        # build first all the goal-level literal predicates
        relations = {
            relation_formatter(
                predicate, goal_level=goal_level, polarity=polarity
            ): predicate.arity
            for predicate, goal_level, polarity in all_variations()
        }
        # add regular predicates for atoms and actions
        for predicate in predicates:
            relations[relation_formatter(predicate)] = predicate.arity
        for action in actions:
            relations[relation_formatter(action)] = action.arity
        goal_satisfaction_derivations = self._parse_satisfaction_derivations(
            goal_satisfaction_derivations
        )
        # if requested, add goal-satisfaction variants of all regular predicates
        for goal_sat_deriv in goal_satisfaction_derivations:
            for predicate, goal_level, polarity in all_variations():
                relations[
                    relation_formatter(
                        predicate,
                        goal_level=goal_level,
                        polarity=polarity,
                        goal_satisfaction=goal_sat_deriv,
                    )
                ] = predicate.arity
        self._dict = dict(sorted(relations.items(), key=operator.itemgetter(0)))
        self.max_goal_level = max_goal_level
        self.goal_satisfaction_derivations: set[str] = goal_satisfaction_derivations
        self.base_relations = top_type_predicates
        self.support_literals = support_literals
        self._recompute_associations()

    def update(self, m, /, **kwargs):
        self._dict.update(m, **kwargs)
        self._recompute_associations()

    def __setitem__(self, item, value):
        raise NotImplementedError(
            f"Setting items is not supported for {self.__class__}. Use update() instead."
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(#{len(self)} | {self._dict})"

    def remove(self, *keys: Node):
        for key in keys:
            if key in self:
                del self._dict[key]
        self._recompute_associations()

    def clear(self):
        self._dict.clear()
        self._recompute_associations()

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __contains__(self, item):
        return item in self._dict

    def __eq__(self, other):
        if not isinstance(other, (RelationDict, dict)):
            return NotImplemented
        return self._dict == other

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def _recompute_associations(self):
        max_level = -1
        pred_variations = {}
        pred_associations = {}
        suffixes = set(relation_formatter.goal_level_suffixes.values()) | set(
            relation_formatter.goal_satisfaction_suffixes.values()
        )
        suffixes.remove("")
        prefixes = set(relation_formatter.polarity_prefixes.values())
        prefixes.remove("")
        for key in self.keys():
            prefixes_left = prefixes.copy()
            suffixes_left = suffixes.copy()
            base_key = key
            while prefix_found := next(
                (p for p in prefixes_left if base_key.startswith(p)), False
            ):
                base_key = base_key[len(prefix_found) :]
                prefixes_left.remove(prefix_found)
            while suffix_found := next(
                (s for s in suffixes_left if base_key.endswith(s)), False
            ):
                base_key = base_key[: -len(suffix_found)]
                suffixes_left.remove(suffix_found)
                if suffix_found in relation_formatter.goal_level_suffixes.values():
                    max_level = max(
                        max_level,
                        next(
                            level
                            for level, suf in relation_formatter.goal_level_suffixes.items()
                            if suf == suffix_found
                        ),
                    )
            pred_variations.setdefault(base_key, []).append(key)
            pred_associations[key] = base_key
        self.max_goal_level = max_level
        self.relation_to_variations = pred_variations
        self.variation_to_relation = pred_associations

    @staticmethod
    def extract_pre_and_suffixes(relation: str) -> tuple[str, list[str], list[str]]:
        prefixes = set(relation_formatter.polarity_prefixes.values())
        prefixes.remove("")
        suffixes = set(relation_formatter.goal_level_suffixes.values()) | set(
            relation_formatter.goal_satisfaction_suffixes.values()
        )
        suffixes.remove("")
        base_relation = relation
        found_prefixes = []
        found_suffixes = []
        prefixes_left = prefixes.copy()
        suffixes_left = suffixes.copy()
        while prefix_found := next(
            (p for p in prefixes_left if base_relation.startswith(p)), False
        ):
            base_relation = base_relation[len(prefix_found) :]
            found_prefixes.append(prefix_found)
            prefixes_left.remove(prefix_found)
        while suffix_found := next(
            (s for s in suffixes_left if base_relation.endswith(s)), False
        ):
            base_relation = base_relation[: -len(suffix_found)]
            found_suffixes.append(suffix_found)
            suffixes_left.remove(suffix_found)
        return base_relation, found_prefixes, found_suffixes

    @staticmethod
    def _parse_satisfaction_derivations(
        goal_satisfaction_derivations: Collection[str | bool] | None,
    ):
        out = set()
        for entry in goal_satisfaction_derivations or []:
            if isinstance(entry, str):
                entry = entry.lower().strip()
                if str(True).lower() == entry:
                    out.add(True)
                elif str(False).lower() == entry:
                    out.add(False)
                else:
                    out.add(entry)
            else:
                out.add(entry)
            out.add(None)
        assert all(
            goal_sat_deriv in relation_formatter.goal_satisfaction_suffixes.keys()
            for goal_sat_deriv in out
        ), (
            f"One or more goal satisfaction derivations are not supported: "
            f"{out}. Supported: "
            f"{list(relation_formatter.goal_satisfaction_suffixes.keys())}."
        )
        return out
