"""pytest plugin: the vLLM conftest fixtures and hooks r6u's tests use, without
tests/conftest.py, which imports tblib (absent from the serving image)."""
import contextlib
import os
import tempfile

import pytest
import torch


@pytest.fixture
def default_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config


@pytest.fixture(autouse=True)
def reset_default_torch_device():
    """tests/kernels/conftest.py: restore the factory default device after every test."""
    yield
    torch.set_default_device(None)


@pytest.fixture
def dist_init():
    """tests/conftest.py's single-rank distributed environment (the Engram model
    tests)."""
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
