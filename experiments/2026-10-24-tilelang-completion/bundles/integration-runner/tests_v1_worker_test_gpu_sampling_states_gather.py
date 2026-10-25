# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sampler's per-row top-k and top-p (``SamplingStates.get_top_k_top_p``) through
the runner's TileLang row gather, against the torch indexing it replaces: the same
values, bit for bit, by int64 and int32 request indices (a repeated and a negative,
wrapping, index included), and only the gathers ``SamplingStates.register_warmups``
declares are launched."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
pytest.importorskip("tilelang")

from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import prepare  # noqa: E402
from vllm.v1.worker.gpu import tilelang_inputs as tli  # noqa: E402
from vllm.v1.worker.gpu.sample import sampler as sampler_module  # noqa: E402
from vllm.v1.worker.gpu.sample.states import SamplingStates  # noqa: E402

DEVICE = torch.device("cuda")
VOCAB = 129280
MAX_REQS = 32


def _states(seed: int) -> SamplingStates:
    """Every slot a request: top-k from 1 to the vocabulary (-1: all of it), top-p
    in (0, 1]; slot 0 uses both."""
    rng = np.random.default_rng(seed)
    states = SamplingStates(MAX_REQS, VOCAB)
    for slot in range(MAX_REQS):
        top_k = 7 if slot == 0 else int(rng.choice([-1, 1, 7, 50, VOCAB]))
        top_p = (
            0.5 if slot == 0 else float(rng.choice([1.0, 0.9, rng.uniform(0.01, 1)]))
        )
        states.add_request(slot, SamplingParams(top_k=top_k, top_p=top_p))
    states.apply_staged_writes()
    torch.accelerator.synchronize()
    return states


def _rows(num_rows: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """The logits rows' request slots: a request's rows together, a repeated slot,
    and -1 (the last slot, as torch's indexing wraps it)."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    rows = torch.randint(0, MAX_REQS, (num_rows,), generator=gen, device=DEVICE)
    if num_rows > 2:
        rows[1] = rows[0]
        rows[-1] = -1
    return rows.to(dtype)


@pytest.mark.parametrize("num_rows", [1, 6, 16, 96, 192])
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
def test_top_k_top_p_gather_is_torch_indexing(num_rows, index_dtype):
    states = _states(num_rows)
    rows = _rows(num_rows, index_dtype, num_rows)
    slots = np.arange(MAX_REQS)
    top_k, top_p = states.get_top_k_top_p(rows, slots)
    torch.accelerator.synchronize()
    assert top_k is not None and top_p is not None
    expected_k = states.top_k.gpu[rows]
    expected_p = states.top_p.gpu[rows]
    assert top_k.dtype == expected_k.dtype == torch.int32
    assert top_p.dtype == expected_p.dtype == torch.float32
    assert top_k.shape == top_p.shape == (num_rows,)
    assert top_k.is_contiguous() and top_p.is_contiguous()
    assert torch.equal(top_k, expected_k)
    assert torch.equal(top_p.view(torch.int32), expected_p.view(torch.int32))


def test_top_k_top_p_gathers_only_what_the_batch_uses():
    """A batch whose requests leave top-k (or top-p) at its default gathers
    nothing for it, as before."""
    states = SamplingStates(MAX_REQS, VOCAB)
    states.add_request(0, SamplingParams(top_k=5))
    states.add_request(1, SamplingParams(top_p=0.5))
    states.add_request(2, SamplingParams())
    states.apply_staged_writes()
    rows = torch.tensor([0, 0, 2], dtype=torch.int64, device=DEVICE)
    top_k, top_p = states.get_top_k_top_p(rows, np.array([0, 2]))
    assert top_p is None and top_k is not None
    assert top_k.tolist() == [5, 5, VOCAB]
    rows = torch.tensor([1, 2], dtype=torch.int64, device=DEVICE)
    top_k, top_p = states.get_top_k_top_p(rows, np.array([1, 2]))
    assert top_k is None and top_p is not None
    assert top_p.tolist() == [0.5, 1.0]
    assert states.get_top_k_top_p(rows, np.array([2])) == (None, None)


def test_the_gathers_launched_are_the_declared_ones(monkeypatch):
    """``SamplingStates.register_warmups`` declares the gathers the sampler can
    launch (int32 top-k and FP32 top-p, by int64 or int32 indices); the launches
    use exactly those keys, so a frozen cache serves them."""
    states = _states(3)
    registry = JitWarmupRegistry(None)
    with registry.activate():
        states.register_warmups()
    declared = {
        (args[0], args[1])
        for function, args, _ in prepare.declared_calls(registry)
        if function is tli.gather_rows_kernel
    }
    assert declared == {
        (index, dtype) for index in ("int32", "int64") for dtype in ("int32", "float32")
    }
    prepare.prepare_declared(registry)
    launched = set()
    original = tli._kernel

    def record(factory, *key):
        launched.add((id(factory), key))
        return original(factory, *key)

    monkeypatch.setattr(tli, "_kernel", record)
    for index_dtype in (torch.int64, torch.int32):
        states.get_top_k_top_p(_rows(6, index_dtype, 5), np.arange(MAX_REQS))
    torch.accelerator.synchronize()
    assert launched == {(id(tli.gather_rows_kernel), key) for key in declared}


def test_sampler_declares_the_states_gathers(monkeypatch):
    """The sampler's declarations include its sampling states' gathers."""
    called = []
    for name in (
        "register_min_p_warmup",
        "register_sampling_warmups",
        "register_topk_scores_warmup",
    ):
        monkeypatch.setattr(sampler_module, name, lambda *a, **k: None)
    monkeypatch.setattr(
        sampler_module.tilelang_topk_topp,
        "register_top_k_top_p",
        lambda *a, **k: None,
    )
    owner = SimpleNamespace(
        sampling_states=SimpleNamespace(
            max_num_reqs=MAX_REQS,
            register_warmups=lambda: called.append("gathers"),
        ),
        num_speculative_tokens=1,
        use_fp64_gumbel=False,
        compute_nans=False,
        trace_replay_state=None,
        logprobs_mode="raw_logprobs",
        **{
            name: SimpleNamespace(register_warmups=lambda *a, **k: None)
            for name in (
                "penalties_state",
                "logit_bias_state",
                "bad_words_state",
                "thinking_budget_state",
                "logprob_token_ids_state",
            )
        },
    )
    sampler_module.Sampler.register_warmups(
        owner,
        logits=((torch.float32, VOCAB),),
        processed_logits_strides=(VOCAB,),
        vocab_size=VOCAB,
    )
    assert called == ["gathers"]
