"""Trace every JIT compile in this process, tagged with the vLLM worker's startup phase.

Loaded through PYTHONPATH by the jittrace arms (it changes no computation). Each process
appends JSON lines to $JITTRACE_DIR/<host>-<pid>.jsonl:
  {"event": "phase", "phase": "load_model", "edge": "start"|"end", ...}
  {"event": "compile", "family": "tilelang"|"triton"|"cute", "name": ..., "factory": ...,
   "key": ..., "seconds": ..., "phase": ...}
  {"event": "disk", "family": "tilelang", "name": ..., "phase": ...}   (a disk-cache load)
Modules are patched as they are imported, so nothing loads early.
"""
import functools
import importlib.abc
import json
import os
import socket
import sys
import threading
import time

_DIR = os.environ.get("JITTRACE_DIR")
_lock = threading.Lock()
_local = threading.local()
_state = {"phase": "import"}
_file = None


def _emit(record):
    global _file
    if not _DIR:
        return
    record.update(t=time.time(), pid=os.getpid(), thread=threading.current_thread().name,
                  phase=record.get("phase", _state["phase"]))
    try:
        line = json.dumps(record, default=repr)
        with _lock:
            if _file is None:
                os.makedirs(_DIR, exist_ok=True)
                _file = open(os.path.join(_DIR, f"{socket.gethostname()}-{os.getpid()}.jsonl"), "a", buffering=1)
            _file.write(line + "\n")
    except Exception:  # a trace must never break the process
        pass


def _short(value, limit=600):
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _patch_tilelang_jit(module):
    impl = getattr(module, "JITImpl", None)
    if impl is None or getattr(impl, "_jittrace", False):
        return
    original = impl.__call__

    @functools.wraps(original)
    def call(self, *args, **kwargs):
        func = getattr(self, "func", None)
        name = getattr(func, "__qualname__", None) or getattr(func, "__name__", repr(func))
        outer = getattr(_local, "factory", None)
        _local.factory = (f"{getattr(func, '__module__', '?')}.{name}", (args, kwargs))
        try:
            return original(self, *args, **kwargs)
        finally:
            _local.factory = outer

    impl.__call__ = call
    impl._jittrace = True


def _patch_tilelang_kernel(module):
    kernel = getattr(module, "JITKernel", None)
    if kernel is None or getattr(kernel, "_jittrace", False):
        return
    original = kernel.__init__

    @functools.wraps(original)
    def init(self, func=None, *args, **kwargs):
        if kwargs.get("from_database"):
            return original(self, func, *args, **kwargs)
        start = time.perf_counter()
        try:
            return original(self, func, *args, **kwargs)
        finally:
            factory, key = getattr(_local, "factory", None) or (None, None)
            key = None if key is None else _short(key)
            attrs = getattr(func, "attrs", None)
            name = str(attrs["global_symbol"]) if attrs is not None and "global_symbol" in attrs else repr(func)[:80]
            _emit({"event": "compile", "family": "tilelang", "name": name, "factory": factory,
                   "key": key, "seconds": round(time.perf_counter() - start, 3)})

    kernel.__init__ = init
    kernel._jittrace = True


def _patch_tilelang_cache(module):
    cache = getattr(module, "KernelCache", None)
    if cache is None or getattr(cache, "_jittrace", False) or not hasattr(cache, "_load_kernel_from_disk"):
        return
    original = cache._load_kernel_from_disk

    @functools.wraps(original)
    def load(self, *args, **kwargs):
        kernel = original(self, *args, **kwargs)
        if kernel is not None:
            factory, key = getattr(_local, "factory", None) or (None, None)
            _emit({"event": "disk", "family": "tilelang", "factory": factory,
                   "key": None if key is None else _short(key)})
        return kernel

    cache._load_kernel_from_disk = load
    cache._jittrace = True


def _timed(family, original, name_of):
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _emit({"event": "compile", "family": family, "name": name_of(args, kwargs),
                   "seconds": round(time.perf_counter() - start, 3)})
    wrapper._jittrace = True
    return wrapper


def _triton_name(args, kwargs):
    src = args[0] if args else kwargs.get("src")
    return getattr(src, "name", None) or getattr(getattr(src, "fn", None), "__name__", None) or _short(src, 120)


def _patch_triton(module):
    for attr in ("compile",):
        original = getattr(module, attr, None)
        if original is not None and not getattr(original, "_jittrace", False):
            setattr(module, attr, _timed("triton", original, _triton_name))


def _patch_cute(module):
    import inspect

    original = getattr(module, "compile", None)
    if inspect.isfunction(original) and not getattr(original, "_jittrace", False):
        module.compile = _timed("cute", original, lambda a, k: getattr(a[0], "__name__", _short(a[0], 120)) if a else "?")


def _patch_worker(module):
    worker = getattr(module, "Worker", None)
    if worker is None or getattr(worker, "_jittrace", False):
        return
    for name in ("load_model", "determine_available_memory", "initialize_from_config",
                 "compile_or_warm_up_model"):
        original = getattr(worker, name, None)
        if original is None:
            continue

        def make(original, name):
            @functools.wraps(original)
            def phase(self, *args, **kwargs):
                _state["phase"] = name
                _emit({"event": "phase", "phase": name, "edge": "start"})
                try:
                    return original(self, *args, **kwargs)
                finally:
                    _emit({"event": "phase", "phase": name, "edge": "end"})
                    _state["phase"] = "serving" if name == "compile_or_warm_up_model" else f"after_{name}"
            return phase

        setattr(worker, name, make(original, name))
    worker._jittrace = True


_PATCHES = {
    "tilelang.jit": _patch_tilelang_jit,
    "tilelang.jit.kernel": _patch_tilelang_kernel,
    "tilelang.cache.kernel_cache": _patch_tilelang_cache,
    "triton.runtime.jit": _patch_triton,
    "triton.compiler.compiler": _patch_triton,
    "cutlass.cute": _patch_cute,
    "vllm.v1.worker.gpu_worker": _patch_worker,
}


class _Finder(importlib.abc.MetaPathFinder):
    """Run a patch right after its module executes."""

    def find_spec(self, name, path, target=None):
        patch = _PATCHES.get(name)
        if patch is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            try:
                spec = finder.find_spec(name, path, target)
            except Exception:
                continue
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        original = loader.exec_module

        def exec_module(module, _original=original, _patch=patch):
            _original(module)
            try:
                _patch(module)
            except Exception as error:  # a trace must never break the process
                _emit({"event": "trace_error", "module": name, "error": repr(error)})

        loader.exec_module = exec_module
        return spec


if _DIR:
    sys.meta_path.insert(0, _Finder())
    _emit({"event": "process", "argv": sys.argv[:3]})
