# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where the GPU worker registers a step's JIT kernels while preparing.

A step's Triton kernels specialize on the KV cache groups' block tables, whose
packing reads the KV cache layout the engine core sets before memory profiling.
The worker asks the runner for them in ``_prepare_kernels`` (inside memory
profiling, so after the layout is set), while the registry collects, and only
then compiles the registry.
"""

import contextlib
from types import SimpleNamespace

import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.v1.worker.gpu_worker import Worker


class _Registry:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    @contextlib.contextmanager
    def activate(self):
        self._trace.append("collect")
        yield
        self._trace.append("collected")

    def warmup(self) -> None:
        self._trace.append("compile")


def _worker(monkeypatch, runner, family: str = "b12x") -> Worker:
    worker = Worker.__new__(Worker)  # no GPU, no init: only preparation is used
    worker.model_runner = runner
    worker.vllm_config = None
    worker._startup_heartbeat = None
    worker.observability_config = SimpleNamespace(
        jit_monitor_mode="warn", jit_monitor_verbose=False
    )
    monkeypatch.setattr(
        gpu_worker_module, "set_current_vllm_config", lambda _: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        gpu_worker_module.envs, "VLLM_DS41_KERNEL_BACKEND", family, raising=False
    )
    return worker


def test_step_kernels_register_while_collecting_before_the_registry_compiles(
    monkeypatch,
):
    trace: list[str] = []
    runner = SimpleNamespace(
        jit_warmup_registry=_Registry(trace),
        declare_step_kernels=lambda: trace.append("step kernels"),
    )
    _worker(monkeypatch, runner)._prepare_kernels()
    assert trace == ["collect", "step kernels", "collected", "compile"]


def test_a_runner_without_step_kernels_still_compiles_its_registry(monkeypatch):
    trace: list[str] = []
    runner = SimpleNamespace(jit_warmup_registry=_Registry(trace))
    _worker(monkeypatch, runner)._prepare_kernels()
    assert trace == ["compile"]


def test_tilelang_kernels_compile_and_freeze_before_binding(monkeypatch):
    """TileLang: declare from the loaded model, compile, warm, add the step's
    kernels, compile the registry, return the compile heap, freeze, bind, and
    only then watch for compiles."""
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.utils import jit_monitor

    trace: list[str] = []
    registry = _Registry(trace)
    target, drafter = object(), object()

    def declare_modules(model, draft):
        assert (model, draft) == (target, drafter)
        trace.append("declare")

    def prepare_declared(active):
        assert active is registry
        trace.append("prepare")
        return 3

    record = lambda name: lambda *args, **kwargs: trace.append(name)  # noqa: E731
    monkeypatch.setattr(prepare, "declare_modules", declare_modules)
    monkeypatch.setattr(prepare, "prepare_declared", prepare_declared)
    monkeypatch.setattr(prepare, "warm_declared", record("warm"))
    monkeypatch.setattr(prepare, "release_host_memory", record("release"))
    monkeypatch.setattr(prepare, "freeze", record("freeze"))
    monkeypatch.setattr(prepare, "bind_declared", record("bind"))
    monkeypatch.setattr(jit_monitor, "activate", record("monitor"))
    runner = SimpleNamespace(
        jit_warmup_registry=registry,
        model=target,
        speculator=SimpleNamespace(model=drafter),
        declare_step_kernels=lambda: trace.append("step kernels"),
    )
    _worker(monkeypatch, runner, "tilelang")._prepare_kernels()
    assert trace == [
        "collect", "declare", "collected", "prepare", "warm",
        "collect", "step kernels", "collected", "compile",
        "release", "freeze", "bind", "monitor",
    ]  # fmt: skip


def test_adaptive_verification_kernels_are_declared_before_the_manager_exists(
    monkeypatch,
):
    """Preparation runs inside memory profiling, before ``initialize_kv_cache``
    builds the adaptive verification manager: the speculator's flag, not the
    manager, decides that its kernels (the draft budget among them) are declared."""
    import torch

    import vllm.v1.core.kv_cache_utils as kv_cache_utils
    import vllm.v1.worker.gpu.model_runner as runner_module
    from vllm.v1.worker.gpu import tilelang_inputs

    declared: list = []
    monkeypatch.setattr(kv_cache_utils, "get_kv_cache_groups", lambda *_: [])
    for name in (
        "add_kv_sharing_layers_to_kv_cache_groups",
        "get_shared_kv_cache_layers",
        "select_kernel_block_sizes",
        "declare_input_batch_kernels",
    ):
        monkeypatch.setattr(runner_module, name, lambda *_, **__: None)
    monkeypatch.setattr(runner_module, "dead_rows_enabled", lambda: False)
    monkeypatch.setattr(runner_module.envs, "VLLM_MOE_SKIP_PADDING", False)
    monkeypatch.setattr(tilelang_inputs, "declare", declared.append)
    monkeypatch.setattr(tilelang_inputs, "_call", lambda factory, *key: (factory, key))
    steps = 5
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(use_request_boundary_checkpoints=False),
        get_kv_cache_spec=dict,
        _register_kv_cache_warmups=lambda *_: None,
        speculator=SimpleNamespace(enable_adaptive_verification=True),
        adaptive_verification=None,
        batch_sharder=None,
        num_speculative_steps=steps,
        draft_tokens_handler=SimpleNamespace(track_consumed_drafts=False),
        req_states=SimpleNamespace(
            next_prefill_tokens=torch.zeros(4, dtype=torch.int32),
            draft_tokens=torch.zeros(16, steps, dtype=torch.int32),
            register_warmups=lambda: None,
        ),
        model_state=SimpleNamespace(num_new_sampled_tokens_per_step=1),
        is_last_pp_rank=False,
        is_pooling_model=False,
    )
    runner_module.GPUModelRunner.declare_step_kernels(runner)
    calls = [call for calls in declared for call in calls()]
    budget = (tilelang_inputs.assign_draft_budget_kernel, (steps, "int64"))
    assert budget in calls
    assert (tilelang_inputs.fill_rows_kernel, ("int64", "float32")) in calls


def test_sink_declares_ranks_for_dense_alternate_rows(monkeypatch):
    """The sink ranks its alternates in dense copies of their rows: at TP3 the
    logits' padded row stride (129408) is not the vocabulary (129280), so the
    dense stride needs its own declaration."""
    import torch

    from vllm.v1.worker.gpu.sample import tilelang_logprob, tilelang_sampling
    from vllm.v1.worker.gpu.sample.sink import SinkSampler

    declared: list = []
    monkeypatch.setattr(tilelang_sampling, "declare", lambda calls: None)
    monkeypatch.setattr(
        tilelang_logprob,
        "register_ranks",
        lambda logits, vocab_size: declared.append((list(logits), vocab_size)),
    )
    vocab, padded = 129280, 129408
    SinkSampler.register_warmups(
        SimpleNamespace(num_alternates=4, rejection=True),
        logits=((torch.bfloat16, padded),),
        processed_logits_strides=(vocab, padded),
        sampled_widths=(1, 6),
        bin_counts_stride=vocab,
        vocab_size=vocab,
    )
    assert declared == [([(torch.bfloat16, vocab), (torch.float32, vocab)], vocab)]


def test_dead_row_caps_are_declared_before_the_manager_exists(monkeypatch):
    """Verification caps its counts at the dead-row cut inside the resampled-token
    insertion; whether a step can have dead rows comes from the speculator's flag
    and the environment, so the capped insertion is declared during preparation,
    before ``initialize_kv_cache`` builds the adaptive verification manager."""
    import torch

    import vllm.v1.core.kv_cache_utils as kv_cache_utils
    import vllm.v1.worker.gpu.model_runner as runner_module
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.v1.worker.gpu import tilelang_inputs
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    monkeypatch.setattr(kv_cache_utils, "get_kv_cache_groups", lambda *_: [])
    for name in (
        "add_kv_sharing_layers_to_kv_cache_groups",
        "get_shared_kv_cache_layers",
        "select_kernel_block_sizes",
        "declare_input_batch_kernels",
        "register_input_batch_warmups",
    ):
        monkeypatch.setattr(runner_module, name, lambda *_, **__: None)
    monkeypatch.setattr(runner_module.envs, "VLLM_MOE_SKIP_PADDING", False)
    monkeypatch.setattr(tilelang_inputs, "declare", lambda calls: None)
    monkeypatch.setattr(Sampler, "register_warmups", lambda self, **_: None)
    declared: list[dict] = []
    monkeypatch.setattr(
        RejectionSampler,
        "register_warmups",
        lambda self, **kwargs: declared.append(kwargs),
    )
    head = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    head.num_embeddings_padded = 129408
    processor = LogitsProcessor.__new__(LogitsProcessor)
    processor.org_vocab_size = 129280
    processor.head_dtype = torch.bfloat16
    language_model = SimpleNamespace(lm_head=head, logits_processor=processor)
    steps = 5
    for enabled in (True, False):
        monkeypatch.setattr(runner_module, "dead_rows_enabled", lambda e=enabled: e)
        runner = SimpleNamespace(
            vllm_config=SimpleNamespace(use_request_boundary_checkpoints=False),
            get_kv_cache_spec=dict,
            _register_kv_cache_warmups=lambda *_: None,
            speculator=SimpleNamespace(
                enable_adaptive_verification=True, draft_logits=None
            ),
            adaptive_verification=None,  # built later, by initialize_kv_cache
            batch_sharder=None,
            num_speculative_steps=steps,
            draft_tokens_handler=SimpleNamespace(track_consumed_drafts=False),
            req_states=SimpleNamespace(
                next_prefill_tokens=torch.zeros(4, dtype=torch.int32),
                draft_tokens=torch.zeros(16, steps, dtype=torch.int32),
                register_warmups=lambda: None,
            ),
            model_state=SimpleNamespace(num_new_sampled_tokens_per_step=1),
            is_last_pp_rank=True,
            is_pooling_model=False,
            sampler=Sampler.__new__(Sampler),
            rejection_sampler=RejectionSampler.__new__(RejectionSampler),
            model_config=SimpleNamespace(get_hidden_size=lambda: 4096),
            dtype=torch.bfloat16,
            _language_model=lambda: language_model,
            prompt_logprobs_worker=None,
            structured_outputs_worker=None,
            sink=None,
            _greedy_verification_available=lambda: False,
        )
        runner_module.GPUModelRunner.declare_step_kernels(runner)
        assert declared.pop()["dead_rows"] is enabled
    assert not declared
