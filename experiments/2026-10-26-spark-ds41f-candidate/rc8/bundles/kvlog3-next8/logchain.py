"""pytest plugin: around each test whose node id contains one of LOGSTATE_MATCH's parts,
kv_cache_utils' logger's parent chain (object ids, propagate, handlers), each link against
logging.getLogger(its name), and the root logger objects."""

import logging
import os

MATCH = [part for part in os.environ.get("LOGSTATE_MATCH", "").split(",") if part]


def _chain() -> str:
    links = []
    logger = logging.getLogger("vllm.v1.core.kv_cache_utils")
    while logger is not None:
        named = logging.getLogger(logger.name) if logger.name != "root" else logging.getLogger()
        links.append(f"{logger.name}#{id(logger) % 100000} is_getLogger={logger is named} "
                     f"propagate={logger.propagate} "
                     f"handlers={[type(h).__name__ for h in logger.handlers]}")
        logger = logger.parent
    roots = (f"logging.root#{id(logging.root) % 100000} getLogger()#{id(logging.getLogger()) % 100000} "
             f"Logger.root#{id(logging.Logger.root) % 100000} "
             f"manager.root#{id(logging.Logger.manager.root) % 100000}")
    return " -> ".join(links) + f" | {roots}"


def _matches(item) -> bool:
    return any(part in item.nodeid for part in MATCH)


def pytest_runtest_call(item):
    if _matches(item):
        print(f"\nCHAIN during {item.nodeid}: {_chain()}", flush=True)


def pytest_runtest_teardown(item):
    if _matches(item):
        print(f"\nCHAIN after {item.nodeid}: {_chain()}", flush=True)
