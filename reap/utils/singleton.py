# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

from typing import Any, Dict


class PickleSafeSingleton(type):
    _instances: Dict[type, Any] = {}

    def __new__(metacls, name, bases, namespace):
        # Inject a __reduce__ into the class so pickle always returns the singleton
        def __reduce__(self):
            # On unpickle, calling the class returns the existing instance
            return (self.__class__, ())

        # Only add if not already defined
        namespace.setdefault("__reduce__", __reduce__)
        return super().__new__(metacls, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            inst = super().__call__(*args, **kwargs)
            cls._instances[cls] = inst
        return cls._instances[cls]
