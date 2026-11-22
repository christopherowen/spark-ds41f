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
        pending = [(key.function, key.args, key.kwargs) for key in compile_keys]
        if not pending:
            return
        _prepare_with_warm_index(pending)


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
            head = getattr(module, "lm_head", None)
            processor = getattr(module, "logits_processor", None)
            bind_head = getattr(
                processor, "prepare_vocab_projection", None
            ) or getattr(processor, "prepare_b12x_vocab_projection", None)
            if head is not None and callable(bind_head):
                bind_head(head)
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


def _fast_load_workers(count: int) -> int:
    """Worker threads for warm disk-cache loads (dlopen + JSON params, negligible RAM)."""
    import os

    cpus = os.cpu_count() or 4
    return max(1, min(count, cpus, 8))


def _prepare_index_provenance(cache_dir: str) -> str:
    """Content-addressed digest of all DeepSeek V4.1 TileLang Python sources,
    cache namespace, GPU capability, and codegen-affecting environment vars.

    Any 1-byte source change, TileLang build change, overlay mount, or codegen
    env var change produces a distinct digest and cleanly bypasses old indexes.
    """
    import hashlib
    import os
    from pathlib import Path
    import torch

    h = hashlib.sha256()
    h.update(b"vllm-ds41-prepare-index-v2\0")
    h.update(cache_dir.encode("utf-8"))
    h.update(b"\0")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        h.update(f"sm_{cap[0]}_{cap[1]}\0".encode("utf-8"))

    # Hash all codegen/cache-affecting env vars
    for key in sorted(os.environ):
        if key.startswith(
            (
                "TILELANG_",
                "VLLM_DS41_",
                "VLLM_L2_PREFETCH_",
                "SPARK_DS41F_",
            )
        ):
            if key in ("TILELANG_CACHE_DIR",):
                continue
            h.update(f"{key}={os.environ[key]}\0".encode("utf-8"))

    # Hash every .py file under deepseek_v4_1/ and all tilelang_*.py under v1/worker/gpu/
    roots: list[Path] = [Path(__file__).resolve().parents[1]]
    gpu_worker_dir = Path(__file__).resolve().parents[3] / "v1" / "worker" / "gpu"
    py_files: list[Path] = []
    for root in roots:
        if root.is_file():
            py_files.append(root)
        elif root.is_dir():
            py_files.extend(sorted(root.rglob("*.py")))
    if gpu_worker_dir.is_dir():
        py_files.extend(sorted(gpu_worker_dir.rglob("tilelang*.py")))
    for path in sorted(set(py_files)):
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(path.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()


def _call_index_sig(impl: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    """Stable string key for a declared lazy-mode TileLang call."""
    fn = getattr(impl, "func", None)
    orig = getattr(fn, "orig_func", None) or getattr(fn, "func", fn)
    mod = getattr(orig, "__module__", None)
    qual = getattr(orig, "__qualname__", None)
    if not mod or not qual:
        return None
    try:
        from tilelang.cache.kernel_cache import normalize_pass_configs

        pc = normalize_pass_configs(impl.pass_configs)
        payload = (
            mod,
            qual,
            repr(args),
            repr(sorted(kwargs.items())),
            repr(impl.out_idx),
            repr(impl.execution_backend),
            repr(impl.target),
            repr(impl.target_host),
            repr(sorted(pc.items())) if isinstance(pc, dict) else repr(pc),
            repr(impl.compile_flags),
        )
        return repr(payload)
    except Exception:
        return None


def _prepare_with_warm_index(calls: list[TileLangCall]) -> None:
    """Load warm-cached declared kernels directly from TileLang's disk cache
    without re-elaborating Python AST -> TVM TIRScript (``impl.get_tir()``),
    falling back seamlessly to ``tilelang.prepare()`` on any miss, disabled
    cache, or unsupported output ABI."""
    import concurrent.futures
    import json
    import os
    import time
    from pathlib import Path
    import tilelang
    import tilelang.cache as tl_cache
    from tilelang import env as tl_env
    from tilelang import tvm
    from tilelang.cache.kernel_cache import normalize_pass_configs
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    t_start = time.monotonic()

    if not calls or not tl_env.is_cache_enabled() or tl_cache.frozen() is not None:
        tilelang.prepare(calls, num_workers=compile_workers(), progress=PROGRESS.report)
        return

    try:
        default_kc, default_bc, *_ = tl_cache._resolve_cache_dispatch(
            tl_env.get_default_target(),
            None,
            tl_env.get_default_execution_backend(),
            False,
        )
        cache_dir = default_kc._get_cache_path("")
    except Exception:
        cache_dir = None

    if not cache_dir:
        tilelang.prepare(calls, num_workers=compile_workers(), progress=PROGRESS.report)
        return

    prov = _prepare_index_provenance(cache_dir)
    index_path = Path(cache_dir).parent / f"prepare-index-{prov[:16]}.json"
    index: dict[str, dict[str, Any]] = {}
    if index_path.is_file():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("provenance") == prov:
                entries = loaded.get("entries")
                if isinstance(entries, dict):
                    index = entries
        except Exception:
            index = {}

    fast_jobs: list[tuple[Any, tuple[Any, ...], dict[str, Any], Any, str, dict[str, Any]]] = []
    miss_items: list[tuple[TileLangCall, str | None]] = []
    seen_fast: set[tuple[int, Any]] = set()

    for call_item in calls:
        impl, args, kwargs = call_item
        kw = dict(kwargs)
        kw.update(kw.pop("__tune_params", {}))
        try:
            impl.initialize_jit_mode(*args, **kw)
            if not impl.is_lazy_mode():
                miss_items.append((call_item, None))
                continue
            key = impl.func.parse_args(*args, **kw)[0]
        except Exception:
            miss_items.append((call_item, None))
            continue

        if impl._get_prepared_kernel(key) is not None:
            continue
        if (id(impl), key) in seen_fast:
            continue

        sig = _call_index_sig(impl, args, kw)
        entry = index.get(sig) if sig is not None else None
        # Only fast-load in-place (out_idx=None) or TVM-FFI callee-allocated
        # outputs: non-callee-allocated outputs need PrimFunc dynamic_symbolic_map.
        can_fast = (
            entry is not None
            and isinstance(entry.get("cache_key"), str)
            and (impl.out_idx is None or bool(entry.get("callee_alloc", False)))
        )
        if can_fast:
            seen_fast.add((id(impl), key))
            fast_jobs.append((impl, args, kw, key, sig, entry))
        else:
            miss_items.append((call_item, sig))

    done_count = 0
    total_count = len(fast_jobs) + len(miss_items)
    PROGRESS.start(total_count)

    if fast_jobs:
        tirx = tvm.tirx
        stub_inplace = tirx.PrimFunc([], tirx.Evaluate(0)).with_attr("global_symbol", "main")
        stub_callee = stub_inplace.with_attr("tl.callee_allocated_outputs", 1)

        def _load_one(job):
            impl, args, kw, key, sig, entry = job
            ckey = entry["cache_key"]
            callee_alloc = bool(entry.get("callee_alloc", False))
            tgt = impl.target if impl.target is not None else tl_env.get_default_target()
            eb = (
                impl.execution_backend
                if impl.execution_backend is not None
                else tl_env.get_default_execution_backend()
            )
            kc, bc, *_ = tl_cache._resolve_cache_dispatch(
                tgt, impl.target_host, eb, False
            )
            pc = normalize_pass_configs(impl.pass_configs)
            stub = stub_callee if callee_alloc else stub_inplace
            # Check in-memory cache first
            with kc._lock:
                mem = kc._memory_cache.get(ckey)
            if mem is not None:
                return (impl, args, kw, key, sig, mem)
            k = kc._load_kernel_from_disk(
                ckey, bc, impl.out_idx, pc, impl.compile_flags, stub, False
            )
            if k is None:
                import shutil
                shutil.rmtree(kc._get_cache_path(ckey), ignore_errors=True)
            else:
                with kc._lock:
                    existing = kc._memory_cache.get(ckey)
                    if existing is not None:
                        k = existing
                    else:
                        kc._memory_cache[ckey] = k
            return (impl, args, kw, key, sig, k)

        with concurrent.futures.ThreadPoolExecutor(
            _fast_load_workers(len(fast_jobs)), "tl-prepare-fast"
        ) as ex:
            for impl, args, kw, key, sig, kernel in ex.map(_load_one, fast_jobs):
                if kernel is not None:
                    impl._record_prepared_kernel(key, kernel)
                    done_count += 1
                    PROGRESS.report(done_count, total_count)
                else:
                    # Cache entry vanished or failed integrity check: fall back cleanly
                    miss_items.append(((impl, args, kw), sig))

    if miss_items:
        miss_calls = [item[0] for item in miss_items]
        base_done = done_count

        def _miss_progress(done: int, _tot: int) -> None:
            PROGRESS.report(base_done + done, total_count)

        compiled = tilelang.prepare(
            miss_calls, num_workers=compile_workers(), progress=_miss_progress
        )
        updated = False
        for (_, sig), kernel in zip(miss_items, compiled or ()):
            if sig is None:
                continue
            ckey = getattr(kernel, "_tilelang_cache_key", None)
            if not isinstance(ckey, str):
                continue
            adapter = getattr(kernel, "adapter", None)
            callee_alloc = bool(
                getattr(adapter, "_ffi_callee_allocated_output_abi", False)
            )
            new_entry = {"cache_key": ckey, "callee_alloc": callee_alloc}
            if index.get(sig) != new_entry:
                index[sig] = new_entry
                updated = True
        if updated:
            try:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = index_path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
                tmp_path.write_text(
                    json.dumps(
                        {"provenance": prov, "entries": index},
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                os.replace(tmp_path, index_path)
            except OSError:
                pass

    logger.info(
        "TileLang warm index: %d fast-loaded, %d compiled/missed in %.2f s (index=%s)",
        done_count,
        len(miss_items),
        time.monotonic() - t_start,
        index_path.name,
    )


def prepare_declared(registry: JitWarmupRegistry) -> int:
    """Compile every TileLang call declared in ``registry`` in one thread pool;
    return the number of distinct calls."""
    calls = declared_calls(registry)
    _prepare_with_warm_index(calls)
    for kernel, registrations in registry.registrations():
        if kernel is not TILELANG:
            continue
        for args, _ in registrations:
            if args:
                TILELANG.mark_prepared(args[0])
    return len(calls)


def release_host_memory() -> None:
    """Return the compile threads' freed heap and warmup allocator blocks to the
    system (glibc keeps a malloc arena per thread and libc10 retains mimalloc
    CPU tensor segments), so memory profiling starts from what serving holds."""
    from vllm.utils.mem_utils import reclaim_uma_memory

    reclaim_uma_memory(synchronize=True)


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
