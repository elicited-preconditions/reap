"""A string-keyed container of sub-modules that tolerates any key name.

``torch.nn.ModuleDict`` registers its entries as attributes of the module, so a
key that is already the name of an ``nn.Module`` attribute or method (``type``,
``to``, ``clear``, ``forward``, ...) is rejected.  Predicate and relation names
of a planning domain can be exactly such words, hence this wrapper.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Optional

import torch


class PatchedModuleDict(torch.nn.Module):
    """Insertion-ordered mapping from ``str`` keys to ``torch.nn.Module`` values.

    The entries live in one child ``torch.nn.ModuleDict`` named ``module_dict``,
    each under its user key with ``SUFFIX`` appended, so no stored name can
    collide with an attribute of ``nn.Module``.  This layout is part of the
    checkpoint format: a parameter ``w`` of the entry ``key`` appears in a
    ``state_dict`` as ``<prefix>module_dict.<key>____.w``.

    All public accessors work with the user keys (without the suffix).
    ``keys()``, ``values()`` and ``items()`` return lists in insertion order,
    and ``next(d)`` returns the first key.

    See https://github.com/pytorch/pytorch/issues/71203.
    """

    SUFFIX = "____"
    SUFFIX_LENGTH = 4

    def __init__(self, module_map: Optional[Mapping[str, torch.nn.Module]] = None):
        super().__init__()
        self.module_dict = torch.nn.ModuleDict()
        if module_map is None:
            return
        if not isinstance(module_map, Mapping):
            raise TypeError(
                "PatchedModuleDict expects a Mapping of str to nn.Module, "
                f"got {type(module_map).__name__}"
            )
        self.update(module_map)

    # -- key translation -------------------------------------------------

    @classmethod
    def _stored(cls, key: str) -> str:
        return key + cls.SUFFIX

    @classmethod
    def _user(cls, stored_key: str) -> str:
        return stored_key[: -cls.SUFFIX_LENGTH]

    # -- mapping protocol ------------------------------------------------

    def __getitem__(self, key: str) -> torch.nn.Module:
        stored = self._stored(key)
        if stored not in self.module_dict:
            raise KeyError(key)
        return self.module_dict[stored]

    def __setitem__(self, key: str, module: torch.nn.Module) -> None:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(
                f"value for key {key!r} must be a torch.nn.Module, "
                f"got {type(module).__name__}"
            )
        # ModuleDict keeps the position of an existing name on reassignment.
        self.module_dict[self._stored(key)] = module

    def __len__(self) -> int:
        return len(self.module_dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __next__(self) -> str:
        return next(iter(self.keys()))

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._stored(key) in self.module_dict

    def keys(self) -> list[str]:
        return [self._user(stored) for stored in self.module_dict.keys()]

    def values(self) -> list[torch.nn.Module]:
        return list(self.module_dict.values())

    def items(self) -> list[tuple[str, torch.nn.Module]]:
        return [(self._user(stored), m) for stored, m in self.module_dict.items()]

    def update(
        self,
        modules: Mapping[str, torch.nn.Module]
        | Iterable[tuple[str, torch.nn.Module]],
    ) -> None:
        pairs = modules.items() if isinstance(modules, Mapping) else modules
        for key, module in pairs:
            self[key] = module
