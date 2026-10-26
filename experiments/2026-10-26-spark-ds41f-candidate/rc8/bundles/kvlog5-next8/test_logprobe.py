"""Probe tests for candidate 8's chunk-5 diagnosis: each does one part of what
tests/quantization/test_register_quantization_config.py::test_custom_quant does, so a
later caplog_vllm test shows which part breaks its capture. Mounted into the image's
tests/ so that tests/conftest.py's fixtures apply."""

import pytest


def test_only_dist_init(dist_init):
    pass


def test_only_workspace_init(workspace_init):
    pass


def test_only_model_load():
    from tests.quantization.utils import load_model_without_vllm_runner

    # Offline, the hub lookup fails, as it does for test_custom_quant.
    with pytest.raises(Exception):
        load_model_without_vllm_runner("meta-llama/Llama-3.2-1B-Instruct")
