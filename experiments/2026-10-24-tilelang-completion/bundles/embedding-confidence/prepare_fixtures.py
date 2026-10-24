"""pytest plugin: the vLLM conftest fixtures and hooks r6u's tests use, without
tests/conftest.py, which imports tblib (absent from the serving image)."""
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
