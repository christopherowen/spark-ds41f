# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Under TileLang the DeepSeek V4.1 compressor owes nothing to B12X preparation.

B12X prepares its units' plans while memory profiling, a distributed forward, so a
B12X unit would compile its CuTe kernel there on a rank whose cache is cold. The
TileLang compressor offers none; binding its declared kernels allocates the serving
storage before profiling instead.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1 import compressor as compressor_module
from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.utils.b12x import PreparationResourceUnavailableError

TOKENS, REQUESTS = 64, 4


def _compressor(monkeypatch, ratio: int) -> DeepseekCompressor:
    monkeypatch.setattr(compressor_module.kernels, "uses_tilelang", lambda: True)
    compressor = DeepseekCompressor.__new__(DeepseekCompressor)  # no weights or TP
    torch.nn.Module.__init__(compressor)
    compressor.prefix, compressor.compress_ratio = "layers.0.compressor", ratio
    compressor.capacity, compressor.requests = TOKENS, REQUESTS
    compressor.fused_wkv_wgate = SimpleNamespace(weight=torch.empty(512 * ratio, 32))
    compressor._plan = None
    compressor._values = compressor._latent = compressor._emitted = None
    compressor._slots = compressor._gates = compressor._pending_values = None
    compressor._pending_gates = compressor._pending_tags = None
    compressor._local_ids = compressor._row_ids = None
    compressor._primer_starts = None
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
def test_no_b12x_unit_and_no_b12x_plan(monkeypatch, ratio):
    compressor = _compressor(monkeypatch, ratio)
    workload = SimpleNamespace(max_tokens=TOKENS, token_counts=(1, TOKENS))
    assert compressor.get_b12x_preparation_units(compressor, workload) == ()
    assert compressor._plan is None


def test_binding_allocates_the_serving_storage(monkeypatch):
    compressor = _compressor(monkeypatch, 2)
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
    # B12X's priming metadata is not needed.
    assert compressor._primer_starts is None


@pytest.mark.parametrize("tilelang", [False, True])
def test_norm_weight_loads_unchanged_and_only_b12x_builds_a_norm(monkeypatch, tilelang):
    # The TileLang compression applies the norm's weight itself, so no B12X norm
    # (and no preparation unit) is built; the checkpoint loads the same parameter.
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.deepseek_v4_1 import b12x_layers

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(compressor_module.kernels, "uses_tilelang", lambda: tilelang)
    monkeypatch.setattr(b12x_layers, "_capacity", lambda: TOKENS)
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
    if tilelang:
        assert type(compressor.norm) is torch.nn.Module
        assert not hasattr(compressor.norm, "b12x_preparation_provider")
    else:
        assert isinstance(compressor.norm, b12x_layers.B12xRMSNorm)


def test_forwarding_before_binding_fails_by_name(monkeypatch):
    compressor = _compressor(monkeypatch, 1)
    with pytest.raises(PreparationResourceUnavailableError, match="bound"):
        compressor.forward(torch.empty(1, 32), metadata=None)
