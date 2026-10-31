# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek V4.1 compressor owes nothing to B12X preparation.

B12X prepares its units' plans while memory profiling, a distributed forward, so a
B12X unit would compile its CuTe kernel there on a rank whose cache is cold. The
compressor offers none; binding its declared TileLang kernels allocates the serving
storage before profiling instead.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
from vllm.models.deepseek_v4_1.tilelang import prepare

TOKENS, REQUESTS = 64, 4


def _compressor(ratio: int) -> DeepseekCompressor:
    compressor = DeepseekCompressor.__new__(DeepseekCompressor)  # no weights or TP
    torch.nn.Module.__init__(compressor)
    compressor.prefix, compressor.compress_ratio = "layers.0.compressor", ratio
    compressor.capacity, compressor.requests = TOKENS, REQUESTS
    compressor.fused_wkv_wgate = SimpleNamespace(weight=torch.empty(512 * ratio, 32))
    compressor._values = compressor._latent = compressor._emitted = None
    compressor._slots = compressor._gates = compressor._pending_values = None
    compressor._pending_gates = compressor._pending_tags = None
    compressor._local_ids = compressor._row_ids = None
    return compressor


def _bind(monkeypatch, compressor: DeepseekCompressor) -> None:
    declared = {}
    monkeypatch.setattr(
        prepare,
        "declare",
        lambda calls=None, bind=None, warm=None: declared.update(bind=bind),
    )
    compressor.declare_kernels()
    declared["bind"]()


@pytest.mark.parametrize("ratio", [1, 2])
def test_offers_b12x_preparation_nothing(ratio):
    compressor = _compressor(ratio)
    assert getattr(compressor, "b12x_preparation_provider", None) is None
    assert not hasattr(compressor, "get_b12x_preparation_units")


def test_binding_allocates_the_serving_storage(monkeypatch):
    compressor = _compressor(2)
    _bind(monkeypatch, compressor)
    assert compressor._values.shape == (TOKENS, 512)
    assert compressor._values.dtype == torch.float32
    assert compressor._gates.shape == (TOKENS, 512)
    assert compressor._latent.shape == (TOKENS, 512)
    assert compressor._latent.dtype == torch.bfloat16
    assert compressor._emitted.shape == compressor._slots.shape == (TOKENS,)
    assert compressor._pending_values.shape == (REQUESTS, 512)
    assert compressor._pending_tags.shape == compressor._local_ids.shape == (REQUESTS,)
    assert compressor._row_ids.tolist() == list(range(REQUESTS))


def test_norm_weight_loads_unchanged_without_a_norm_module(monkeypatch):
    # The TileLang compression applies the norm's weight itself, so no norm module
    # (and no preparation unit) is built; the checkpoint loads the same parameter.
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=TOKENS, max_num_seqs=REQUESTS
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(rms_norm_eps=1e-6)),
    )
    compressor = DeepseekCompressor(config, 1, 32, 512, prefix="layers.0.compressor")
    weight = dict(compressor.named_parameters())["norm.weight"]
    assert weight is compressor.norm.weight
    assert weight.shape == (512,) and weight.dtype == torch.float32
    assert not weight.requires_grad and not hasattr(weight, "weight_loader")
    assert type(compressor.norm) is torch.nn.Module
    assert not hasattr(compressor.norm, "b12x_preparation_provider")


def test_forwarding_before_binding_fails_by_name():
    compressor = _compressor(1)
    with pytest.raises(RuntimeError, match="bound"):
        compressor.forward(torch.empty(1, 32), metadata=None)


def test_sequential_compressor_layers_share_capacity_scratch(monkeypatch):
    """Multiple compressor layers of the same geometry and ratio share one set
    of transient serving scratch buffers instead of duplicating them per layer."""
    first, second = _compressor(2), _compressor(2)
    other_ratio = _compressor(1)
    _bind(monkeypatch, first)
    _bind(monkeypatch, second)
    _bind(monkeypatch, other_ratio)
    for attr in (
        "_values",
        "_latent",
        "_emitted",
        "_slots",
        "_row_ids",
        "_gates",
        "_pending_values",
        "_pending_gates",
        "_pending_tags",
        "_local_ids",
    ):
        assert getattr(first, attr) is getattr(second, attr)
    assert first._values is not other_ratio._values
