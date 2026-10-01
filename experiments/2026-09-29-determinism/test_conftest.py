"""pytest fixture for the vLLM unit tests run in fresh containers (run57 onward): a default VllmConfig."""
import pytest
@pytest.fixture
def default_vllm_config():
    """Set a default VllmConfig for tests that directly test CustomOps or pathways
    that use get_current_vllm_config() outside of a full engine context.
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))  # the GPU test still runs on cuda; only the config defaults to CPU
    with set_current_vllm_config(config):
        yield config
