"""pytest plugin: fixtures used by the startup-weight-arena unit tests without
tests/conftest.py, which imports tblib (absent from the serving image)."""
import contextlib
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest
import torch


Path("/c/direct-io-test").mkdir(parents=True, exist_ok=True)


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
