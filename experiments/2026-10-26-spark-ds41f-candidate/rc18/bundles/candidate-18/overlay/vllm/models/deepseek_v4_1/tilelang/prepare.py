# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel preparation for DeepSeek V4.1's TileLang kernel family.

Every TileLang kernel the deployment can launch is declared as a call, exactly as its
call site makes it, by the object that owns it, when that object is built
(``declare(calls, bind)``). After loading, the worker compiles every declared call
before any distributed work, in one thread pool (``prepare_declared``, through
``tilelang.prepare``), freezes TileLang's kernel cache, and lets each owner store its
kernels (``bind_declared``). From then on a call that was not declared raises
``KernelNotPreparedError`` on every rank alike, whatever each rank's disk cache holds,
instead of compiling inside a distributed forward, where the ranks that already have
the kernel would wait in a collective for the one that is compiling.

``calls()`` runs when the worker prepares, after the weights are loaded, so a call may
depend on what loading produced. The call site makes its call through the same key
function, so the declared call and the launched call cannot drift apart. Weight
processing transforms weights only.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable, Iterable
from typing import Any

from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry, VllmJitKernel

# (TileLang JIT function, positional arguments, keyword arguments), exactly as the
# call site makes the call.
TileLangCall = tuple[Any, tuple[Any, ...], dict[str, Any]]


def call(function: Any, *args: Any, **kwargs: Any) -> TileLangCall:
    """One declared call of a ``@tilelang.jit`` function."""
    return function, args, kwargs


class _CallKey:
    """A hashable call (TileLang argument values need not be hashable)."""

    __slots__ = ("function", "args", "kwargs", "_key")

    def __init__(self, function: Any, args: tuple[Any, ...], kwargs: dict) -> None:
        self.function = function
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self._key = (id(function), repr((self.args, sorted(self.kwargs.items()))))

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _CallKey) and self._key == other._key

    def __repr__(self) -> str:
        name = getattr(self.function, "__name__", repr(self.function))
        return f"{name}{self._key[1]}"


Calls = Callable[[], Iterable[TileLangCall]]


class TileLangKernels(VllmJitKernel[_CallKey]):
    """DeepSeek V4.1's declared TileLang calls in the runner's JIT registry.

    A registration is ``(calls, bind, warm)``: ``calls()`` lists the owner's calls
    when the worker prepares; ``warm()`` runs third-party kernels whose keys their
    library computes (TileKernels) once on placeholder inputs; ``bind()`` stores
    the owner's kernels once the cache is frozen.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prepared_calls: weakref.WeakSet[Calls] = weakref.WeakSet()

    def declared_keys(
        self, calls: Calls | None, bind=None, warm=None
    ) -> list[_CallKey]:
        if calls is None:
            return []
        return [_CallKey(function, args, kwargs) for function, args, kwargs in calls()]

    def get_warmup_keys(
        self, calls: Calls | None, bind=None, warm=None
    ) -> list[_CallKey]:
        if calls is None or calls in self._prepared_calls:
            return []
        return self.declared_keys(calls, bind, warm)

    def mark_prepared(self, calls: Calls | None) -> None:
        if calls is not None:
            self._prepared_calls.add(calls)

    def compile(self, compile_key: _CallKey) -> None:
        self.compile_many([compile_key])

    def compile_many(self, compile_keys: Iterable[_CallKey]) -> None:
        import tilelang

        pending = [(key.function, key.args, key.kwargs) for key in compile_keys]
        if not pending:
            return
        tilelang.prepare(
            pending,
            num_workers=compile_workers(),
            progress=PROGRESS.report,
        )


TILELANG = TileLangKernels()


def declare(
    calls: Calls | None = None,
    bind: Callable[[], None] | None = None,
    warm: Callable[[], None] | None = None,
) -> None:
    """Declare an owner's TileLang kernels.

    ``calls()`` lists the calls of the owner's own ``@tilelang.jit`` functions,
    exactly as its call sites make them; they compile in one thread pool.
    ``warm()`` runs kernels whose compile keys their library computes (TileKernels'
    casts, norms, mHC and routing) once, on placeholder inputs of the shapes the
    owner launches, on this rank alone. ``bind()`` stores the owner's kernels.

    Under a runner whose JIT registry collects (model construction and loading),
    ``calls()`` and ``warm()`` run when the worker prepares kernels, after loading,
    and ``bind()`` once they are prepared and the cache is frozen. Elsewhere
    (kernel tests) all three run now.
    """
    if JitWarmupRegistry.active() is None:
        if calls is not None:
            import tilelang

            tilelang.prepare(list(calls()))
        if warm is not None:
            warm()
        if bind is not None:
            bind()
        return
    JitWarmupRegistry.register(TILELANG, calls, bind, warm)


def declare_modules(*models: Any) -> None:
    """Let the modules of ``models`` declare their kernels, after loading.

    Calls ``quant_method.declare_layer_kernels(module)`` on every module's linear
    method that does (methods are often attached after a layer is built, so
    construction is not where they can declare), then ``module.declare_kernels()``
    on every module that defines it. Must run while the runner's JIT registry
    collects.
    """
    seen: set[int] = set()
    modules: list[Any] = []
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            modules.append(module)
            method = getattr(module, "quant_method", None)
            declare_layer = getattr(method, "declare_layer_kernels", None)
            if declare_layer is not None:
                declare_layer(module)
    for module in modules:
        declare_own = getattr(module, "declare_kernels", None)
        if declare_own is not None:
            declare_own()


class _Progress:
    """Kernels prepared so far, for the startup rendezvous's progress reports."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.done = 0
        self.total = 0

    def start(self, total: int) -> None:
        with self._lock:
            self.done, self.total = 0, total

    def report(self, done: int, total: int) -> None:
        with self._lock:
            # tilelang.prepare counts the calls it compiles; keep the larger figures.
            self.done = max(self.done, done)
            self.total = max(self.total, total)

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self.done, self.total


PROGRESS = _Progress()


def declared_calls(registry: JitWarmupRegistry) -> list[TileLangCall]:
    """Every distinct TileLang call declared in ``registry``."""
    calls: dict[_CallKey, TileLangCall] = {}
    for kernel, registrations in registry.registrations():
        if kernel is not TILELANG:
            continue
        for args, kwargs in registrations:
            for key in kernel.declared_keys(*args, **kwargs):
                calls.setdefault(key, (key.function, key.args, key.kwargs))
    return list(calls.values())


def compile_workers() -> int | None:
    """Compile threads: the deployment's compile parallelism (``MAX_JOBS``), since every
    compile holds host memory, which on unified-memory GPUs is the device's; all CPUs
    when it is unset."""
    from vllm import envs

    return int(envs.MAX_JOBS) if envs.MAX_JOBS else None


def prepare_declared(registry: JitWarmupRegistry) -> int:
    """Compile every TileLang call declared in ``registry`` in one thread pool;
    return the number of distinct calls."""
    import tilelang

    calls = declared_calls(registry)
    PROGRESS.start(len(calls))
    tilelang.prepare(calls, num_workers=compile_workers(), progress=PROGRESS.report)
    for kernel, registrations in registry.registrations():
        if kernel is not TILELANG:
            continue
        for args, _ in registrations:
            if args:
                TILELANG.mark_prepared(args[0])
    return len(calls)


def release_host_memory() -> None:
    """Return the compile threads' freed heap to the system (glibc keeps a malloc
    arena per thread), so memory profiling starts from what serving holds."""
    import contextlib
    import ctypes

    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


_SAFE_CELL_ATOMS = (int, float, str, bytes, bool, type(None), type)


def _cell_key(value: Any) -> Any:
    if isinstance(value, _SAFE_CELL_ATOMS):
        return (type(value), value)
    if isinstance(value, tuple):
        return (tuple, tuple(_cell_key(x) for x in value))
    return ("id", id(value))


def _callback_dedup_key(callback: Callable[[], None], deduplicate_closures: bool) -> Any:
    if not deduplicate_closures:
        return ("id", id(callback))
    code = getattr(callback, "__code__", None)
    if code is None or getattr(callback, "__self__", None) is not None:
        return ("id", id(callback))
    closure = getattr(callback, "__closure__", None)
    if not closure:
        return ("code", code)
    try:
        cells = tuple(_cell_key(cell.cell_contents) for cell in closure)
    except (TypeError, ValueError):
        return ("id", id(callback))
    return ("closure", code, cells)


def _callbacks(
    registry: JitWarmupRegistry,
    index: int,
    *,
    deduplicate_closures: bool = False,
) -> list[Callable[[], None]]:
    found: dict[Any, Callable[[], None]] = {}
    for kernel, registrations in registry.registrations():
        if kernel is not TILELANG:
            continue
        for args, _ in registrations:
            callback = args[index] if len(args) > index else None
            if callback is not None:
                key = _callback_dedup_key(callback, deduplicate_closures)
                found.setdefault(key, callback)
    return list(found.values())


def warm_declared(registry: JitWarmupRegistry) -> int:
    """Run every owner's ``warm()`` (before ``freeze``); return how many ran."""
    callbacks = _callbacks(registry, 2, deduplicate_closures=True)
    for warm in callbacks:
        warm()
    return len(callbacks)


def freeze(reason: str) -> None:
    """Refuse every TileLang kernel that is not prepared, from now on."""
    from tilelang.cache import freeze as freeze_cache

    freeze_cache(reason)


_BOUND_CACHE: dict[_CallKey, Any] | None = None


def launch(entry: TileLangCall) -> Any:
    """Return the prepared kernel for ``entry``, reusing bindings within ``bind_declared``."""
    function, args, kwargs = entry
    if _BOUND_CACHE is not None:
        key = _CallKey(function, args, kwargs)
        kernel = _BOUND_CACHE.get(key)
        if kernel is None:
            kernel = function(*args, **kwargs)
            _BOUND_CACHE[key] = kernel
        return kernel
    return function(*args, **kwargs)


def bind_declared(registry: JitWarmupRegistry) -> None:
    """Let every owner store its prepared kernels (after ``freeze``, so a bind that
    asks for an undeclared call fails by name)."""
    global _BOUND_CACHE
    _BOUND_CACHE = {}
    try:
        for bind in _callbacks(registry, 1):
            bind()
    finally:
        _BOUND_CACHE = None
