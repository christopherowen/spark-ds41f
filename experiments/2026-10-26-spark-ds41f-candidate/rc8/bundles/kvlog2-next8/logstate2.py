"""pytest plugin: around each test whose node id contains one of LOGSTATE_MATCH's parts,
whether kv_cache_utils emits a record at all (a handler on its own logger), whether its
spec classes are the ones kv_cache_interface exports now (a reloaded module breaks
isinstance), and the global state test_custom_quant's fixtures touch."""

import logging
import os
import sys

MATCH = [part for part in os.environ.get("LOGSTATE_MATCH", "").split(",") if part]
SPECS = ("MLAAttentionSpec", "SlidingWindowSpec", "MambaSpec", "UniformTypeKVCacheSpecs",
         "SlidingWindowMLASpec", "ChunkedLocalAttentionSpec", "HiddenStateCacheSpec")


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(f"{record.levelname} {record.name}: {record.getMessage()[:120]}")


_handler = _Records()


def _matches(item) -> bool:
    return any(part in item.nodeid for part in MATCH)


def _identity() -> str:
    utils = sys.modules.get("vllm.v1.core.kv_cache_utils")
    interface = sys.modules.get("vllm.v1.kv_cache_interface")
    if utils is None or interface is None:
        return "modules not imported yet"
    same = {name: getattr(utils, name, None) is getattr(interface, name, None) for name in SPECS}
    loggers = utils.logger is logging.getLogger("vllm.v1.core.kv_cache_utils")
    copies = sorted(name for name, module in sys.modules.items()
                    if name.endswith("kv_cache_interface") or name.endswith("kv_cache_utils"))
    return f"same classes {same}; logger is getLogger: {loggers}; modules {copies}"


def _globals() -> str:
    parts = []
    try:
        from vllm.config import get_current_vllm_config_or_none
        config = get_current_vllm_config_or_none()
        parts.append(f"current vllm config set: {config is not None}")
    except Exception as error:  # the accessor's name differs between trees
        parts.append(f"current vllm config: {type(error).__name__}")
    try:
        import torch.distributed as dist
        parts.append(f"torch.distributed initialized: {dist.is_initialized()}")
    except Exception as error:
        parts.append(f"torch.distributed: {type(error).__name__}")
    return "; ".join(parts)


def pytest_runtest_setup(item):
    if _matches(item):
        _handler.messages.clear()
        logging.getLogger("vllm.v1.core.kv_cache_utils").addHandler(_handler)
        print(f"\nPROBE before {item.nodeid}: {_identity()}; {_globals()}", flush=True)


def pytest_runtest_teardown(item):
    if _matches(item):
        logging.getLogger("vllm.v1.core.kv_cache_utils").removeHandler(_handler)
        print(f"\nPROBE after {item.nodeid}: kv_cache_utils records {_handler.messages}; "
              f"{_globals()}", flush=True)
