"""Installable package for spark-ds41f cluster control, build, doctor, bench and lab."""

from __future__ import annotations

import functools
import types

from . import config, doctor, cluster, build, bench, cli
from .config import *  # noqa: F401, F403
from .doctor import *  # noqa: F401, F403
from .cluster import *  # noqa: F401, F403
from .build import *  # noqa: F401, F403
from .bench import *  # noqa: F401, F403
from .cli import *  # noqa: F401, F403

_SUBMODULES = (config, doctor, cluster, build, bench, cli)
_MODULE_NAMES = {m.__name__ for m in _SUBMODULES}


def _rebind_function(fn: types.FunctionType, target_globals: dict) -> types.FunctionType:
    rebound = types.FunctionType(
        fn.__code__,
        target_globals,
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    rebound.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    rebound.__doc__ = fn.__doc__
    rebound.__qualname__ = fn.__qualname__
    rebound.__module__ = fn.__module__
    for attr in ("__wrapped__",):
        if hasattr(fn, attr):
            setattr(rebound, attr, getattr(fn, attr))
    return rebound


def _bind_package_namespace() -> None:
    pkg_globals = globals()
    for mod in _SUBMODULES:
        for name, value in list(mod.__dict__.items()):
            if name.startswith("__") and name not in ("__all__",):
                continue
            pkg_globals[name] = value

    rebound_map: dict[object, object] = {}
    for mod in _SUBMODULES:
        for name, value in list(mod.__dict__.items()):
            if isinstance(value, types.FunctionType) and value.__module__ in _MODULE_NAMES:
                rebound = _rebind_function(value, pkg_globals)
                rebound_map[value] = rebound
                pkg_globals[name] = rebound
                setattr(mod, name, rebound)
            elif isinstance(value, functools._lru_cache_wrapper) and getattr(value, "__module__", None) in _MODULE_NAMES:
                inner = _rebind_function(value.__wrapped__, pkg_globals)
                wrapped = functools.cache(inner)
                rebound_map[value] = wrapped
                pkg_globals[name] = wrapped
                setattr(mod, name, wrapped)
            elif isinstance(value, type) and value.__module__ in _MODULE_NAMES:
                for attr_name, attr_val in list(value.__dict__.items()):
                    if isinstance(attr_val, types.FunctionType):
                        setattr(value, attr_name, _rebind_function(attr_val, pkg_globals))

    for name, value in list(pkg_globals.items()):
        if not name.startswith("__") and isinstance(value, dict):
            for k, v in list(value.items()):
                if isinstance(v, types.FunctionType) and v in rebound_map:
                    value[k] = rebound_map[v]


_bind_package_namespace()
