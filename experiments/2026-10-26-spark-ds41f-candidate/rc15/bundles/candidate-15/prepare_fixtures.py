"""pytest plugin: fixtures and TileLang cache seeding for Candidate 15 tests."""
import contextlib
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

import pytest
import torch

if "tblib" not in sys.modules:
    tblib = types.ModuleType("tblib")
    pickling_support = types.ModuleType("tblib.pickling_support")
    pickling_support.install = lambda *args, **kwargs: None
    tblib.pickling_support = pickling_support
    sys.modules["tblib"] = tblib
    sys.modules["tblib.pickling_support"] = pickling_support

Path("/c/direct-io-test").mkdir(parents=True, exist_ok=True)

# Seed writable TileLang cache dir with symlinks to pre-built 1030 kernels
tl_cache = Path(os.environ.get("TILELANG_CACHE_DIR", "/c/tilelang-candidate-15"))
tl_cache.mkdir(parents=True, exist_ok=True)
ro_cache = Path("/cache/spark-ds41f/jit/tilelang")
if ro_cache.is_dir():
    for entry in ro_cache.iterdir():
        target = tl_cache / entry.name
        if not target.exists():
            with contextlib.suppress(OSError):
                target.symlink_to(entry)


@pytest.fixture
def default_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config


@pytest.fixture(autouse=True)
def reset_default_torch_device():
    yield
    torch.set_default_device(None)


@pytest.fixture
def dist_init():
    from vllm.config import (
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    config = (
        contextlib.nullcontext()
        if get_current_vllm_config_or_none() is not None
        else set_current_vllm_config(VllmConfig())
    )
    try:
        with config:
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


@pytest.fixture
def workspace_init():
    from vllm.platforms import current_platform
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device(f"{current_platform.device_type}:0")
    init_workspace_manager(device)
    yield
    with patch("vllm.v1.worker.workspace._manager", None):
        pass
