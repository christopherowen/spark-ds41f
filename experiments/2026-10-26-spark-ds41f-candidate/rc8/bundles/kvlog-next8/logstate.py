"""pytest plugin: the vllm loggers' state before and after each test whose node id contains
one of LOGSTATE_MATCH's comma-separated parts (candidate 8's chunk-5 diagnosis)."""

import logging
import os

MATCH = [part for part in os.environ.get("LOGSTATE_MATCH", "").split(",") if part]
NAMES = ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.kv_cache_utils")


def _state() -> str:
    parts = [f"logging.disable={logging.root.manager.disable}",
             f"root handlers={[type(h).__name__ for h in logging.root.handlers]}"]
    for name in NAMES:
        logger = logging.getLogger(name)
        parts.append(f"{name}: level={logger.level} effective={logger.getEffectiveLevel()} "
                     f"propagate={logger.propagate} disabled={logger.disabled} "
                     f"handlers={[type(h).__name__ for h in logger.handlers]}")
    return "; ".join(parts)


def _matches(item) -> bool:
    return any(part in item.nodeid for part in MATCH)


def pytest_runtest_setup(item):
    if _matches(item):
        print(f"\nLOGSTATE before {item.nodeid}: {_state()}", flush=True)


def pytest_runtest_teardown(item):
    if _matches(item):
        print(f"\nLOGSTATE after {item.nodeid}: {_state()}", flush=True)
