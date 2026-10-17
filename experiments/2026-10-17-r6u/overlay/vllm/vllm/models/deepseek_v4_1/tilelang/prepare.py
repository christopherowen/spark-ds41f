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

    def get_warmup_keys(
        self, calls: Calls | None, bind=None, warm=None
    ) -> list[_CallKey]:
        if calls is None:
            return []
        return [_CallKey(function, args, kwargs) for function, args, kwargs in calls()]

    def compile(self, compile_key: _CallKey) -> None:
        self.compile_many([compile_key])

    def compile_many(self, compile_keys: Iterable[_CallKey]) -> None:
        import tilelang

        tilelang.prepare(
            [(key.function, key.args, key.kwargs) for key in compile_keys],
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

    Calls ``module.declare_kernels()`` on every module that defines it, and
    ``quant_method.declare_layer_kernels(module)`` on every module's linear method
    that does (methods are often attached after a layer is built, so construction
    is not where they can declare). Must run while the runner's JIT registry
    collects.
    """
    seen: set[int] = set()
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            method = getattr(module, "quant_method", None)
            declare_layer = getattr(method, "declare_layer_kernels", None)
            if declare_layer is not None:
                declare_layer(module)
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
            for key in kernel.get_warmup_keys(*args, **kwargs):
                calls.setdefault(key, (key.function, key.args, key.kwargs))
    return list(calls.values())


def prepare_declared(registry: JitWarmupRegistry) -> int:
    """Compile every TileLang call declared in ``registry`` in one thread pool;
    return the number of distinct calls."""
    import tilelang

    calls = declared_calls(registry)
    PROGRESS.start(len(calls))
    tilelang.prepare(calls, progress=PROGRESS.report)
    return len(calls)


def _callbacks(registry: JitWarmupRegistry, index: int) -> list[Callable[[], None]]:
    found: dict[int, Callable[[], None]] = {}
    for kernel, registrations in registry.registrations():
        if kernel is not TILELANG:
            continue
        for args, _ in registrations:
            callback = args[index] if len(args) > index else None
            if callback is not None:
                found.setdefault(id(callback), callback)
    return list(found.values())


def warm_declared(registry: JitWarmupRegistry) -> int:
    """Run every owner's ``warm()`` (before ``freeze``); return how many ran."""
    callbacks = _callbacks(registry, 2)
    for warm in callbacks:
        warm()
    return len(callbacks)


def freeze(reason: str) -> None:
    """Refuse every TileLang kernel that is not prepared, from now on."""
    from tilelang.cache import freeze as freeze_cache

    freeze_cache(reason)


def bind_declared(registry: JitWarmupRegistry) -> None:
    """Let every owner store its prepared kernels (after ``freeze``, so a bind that
    asks for an undeclared call fails by name)."""
    for bind in _callbacks(registry, 1):
        bind()
