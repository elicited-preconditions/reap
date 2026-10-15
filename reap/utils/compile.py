# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

"""
Utilities for selecting methods for compilation with torch.compile through user-defined attrs.

Behavior
--------
- Can be applied to any instance method (not just `forward`, adapt the ).
- Whether compilation is enabled is decided per-instance via an attribute
  named `_compile_<funcname>_enable` (e.g., `_compile_forward_enable`).
- Compile kwargs are resolved per-instance from `_compile_<funcname>_kwargs`
  (a dict), merged on top of any kwargs passed to the decorator.
- Compilation is lazy (on first call) and cached per-instance. The compiled
  unbound function is rebound to the instance via `types.MethodType` and
  stored in `_compiled_<funcname>_cached`. For performance, we also bind it
  to the instance attribute with the original function name.

Notes
-----
If `torch.compile` is unavailable or compilation fails at runtime, the
  original method is executed as a safe fallback.
"""

from __future__ import annotations

import logging
import types
from functools import wraps
from typing import Any, Callable, Dict

import torch

__all__ = ["optional_compile"]

logger = logging.getLogger(__name__)


def optional_compile(
    enable_attr: str | None = None,
    kwargs_attr: str | None = None,
    cache_attr: str | None = None,
    **default_compile_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory to optionally compile an instance method with torch.compile.

    Enablement and kwargs are controlled per-instance using attributes derived
    from the method name:
      - `_compile_<funcname>_enable` (bool): when True, compile on first call.
      - `_compile_<funcname>_kwargs` (dict): extra kwargs passed to torch.compile,
        merged over the decorator-level default_compile_kwargs.

    This is different to class-level decorators, such as @torch.compile, which run once at class definition-time.
    The wrapped callable will check for an already compiled method in the cache and return it.

    Example:
        class MyModule(torch.nn.Module):
            @optional_compile(dynamic=True)
            def forward(self, x):
                ...

        model = MyModule()
        model._compile_forward_enable = True
        model._compile_forward_kwargs = {"fullgraph": False}  # better to handle via instance-args in __init__
        y = model(x)  # compiles forward on first call, then caches it

    If compilation is disabled (or torch.compile is unavailable), the original
    function is executed.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_name = func.__name__
        _enable_attr = enable_attr or f"_compile_{func_name}_enable"
        _kwargs_attr = kwargs_attr or f"_compile_{func_name}_kwargs"
        _cache_attr = cache_attr or f"_compiled_{func_name}_cached"

        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            enabled = bool(getattr(self, _enable_attr, False))
            if not enabled:
                return func(self, *args, **kwargs)

            compiled_bound = getattr(self, _cache_attr, None)
            if compiled_bound is None:
                # merge compile kwargs: decorator defaults < instance overrides
                inst_kwargs = getattr(self, _kwargs_attr, None)
                compile_kwargs: Dict[str, Any] = dict(default_compile_kwargs)
                if isinstance(inst_kwargs, dict):
                    compile_kwargs.update(inst_kwargs)

                try:
                    compiled_unbound = torch.compile(func, **compile_kwargs)
                    compiled_bound = types.MethodType(compiled_unbound, self)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    # Safe fallback on compile failure: cache original to avoid
                    # repeated compile attempts on subsequent calls.
                    logger.warning(
                        "optional_compile: compilation failed for '%s' on %s; using original callable",
                        func_name,
                        type(self).__name__,
                        exc_info=True,
                    )
                    orig_bound = types.MethodType(func, self)
                    setattr(self, _cache_attr, orig_bound)
                    return orig_bound(*args, **kwargs)

                # Cache and also bind to the instance under the original name
                setattr(self, _cache_attr, compiled_bound)
            return compiled_bound(*args, **kwargs)

        return wrapper

    return decorator
