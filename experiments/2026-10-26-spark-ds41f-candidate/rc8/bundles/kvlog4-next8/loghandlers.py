"""pytest plugin: around each test whose node id contains one of LOGSTATE_MATCH's parts,
the root logger's level and each of its handlers' class, id, level and filters (pytest
reuses its two capture handlers for the whole session), and whether a record logged on
kv_cache_utils' logger reaches a probe handler on the root."""

import logging
import os

MATCH = [part for part in os.environ.get("LOGSTATE_MATCH", "").split(",") if part]


class _Probe(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.seen = []

    def emit(self, record):
        self.seen.append(record.getMessage()[:60])


def _handlers() -> str:
    root = logging.getLogger()
    parts = [f"root level={root.level}"]
    for handler in root.handlers:
        filters = [getattr(f, "__qualname__", type(f).__qualname__) for f in handler.filters]
        parts.append(f"{type(handler).__name__}#{id(handler) % 100000} level={handler.level} "
                     f"filters={filters}")
    return "; ".join(parts)


def _reaches_root() -> str:
    probe = _Probe()
    root = logging.getLogger()
    root.addHandler(probe)
    vllm = logging.getLogger("vllm")
    old = vllm.propagate
    vllm.propagate = True
    try:
        logging.getLogger("vllm.v1.core.kv_cache_utils").warning("loghandlers probe record")
    finally:
        vllm.propagate = old
        root.removeHandler(probe)
    return f"probe record reached the root: {probe.seen == ['loghandlers probe record']}"


def _matches(item) -> bool:
    return any(part in item.nodeid for part in MATCH)


def pytest_runtest_call(item):
    if _matches(item):
        print(f"\nHANDLERS during {item.nodeid}: {_handlers()}; {_reaches_root()}", flush=True)


def pytest_runtest_teardown(item):
    if _matches(item):
        print(f"\nHANDLERS after {item.nodeid}: {_handlers()}", flush=True)
