# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 TileLang kernels are declared into the runner's JIT registry."""

import tilelang

from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.models.deepseek_v4_1.tilelang import prepare


def _gemm(n, k, block_M=16):  # stands in for a @tilelang.jit function
    raise AssertionError("declaring must not compile")


def _rope(heads, dim):
    raise AssertionError("declaring must not compile")


def test_declared_calls_are_collected_once_each():
    registry = JitWarmupRegistry(vllm_config=None)
    with registry.activate():
        prepare.declare(lambda: [prepare.call(_gemm, 1024, 512, block_M=16)])
        # A second owner declaring the same call (another layer of the same shape).
        prepare.declare(
            lambda: [
                prepare.call(_gemm, 1024, 512, block_M=16),
                prepare.call(_gemm, 1024, 512, block_M=32),
                prepare.call(_rope, 8, 512),
            ]
        )
    calls = sorted(
        (fn.__name__, args, tuple(sorted(kw.items())))
        for fn, args, kw in prepare.declared_calls(registry)
    )
    assert calls == [
        ("_gemm", (1024, 512), (("block_M", 16),)),
        ("_gemm", (1024, 512), (("block_M", 32),)),
        ("_rope", (8, 512), ()),
    ]


def test_calls_are_listed_and_kernels_bound_when_preparing():
    registry = JitWarmupRegistry(vllm_config=None)
    events = []

    def calls():
        events.append("listed")
        return [prepare.call(_rope, 4, 128)]

    with registry.activate():
        prepare.declare(calls, bind=lambda: events.append("bound"))
    assert events == []
    prepare.declared_calls(registry)
    prepare.bind_declared(registry)
    assert events == ["listed", "bound"]


def test_standalone_declaration_prepares_and_binds_now(monkeypatch):
    prepared = []
    monkeypatch.setattr(tilelang, "prepare", lambda calls, **kw: prepared.extend(calls))
    bound = []
    prepare.declare(lambda: [prepare.call(_rope, 2, 64)], bind=lambda: bound.append(1))
    assert prepared == [(_rope, (2, 64), {})] and bound == [1]


def test_prepare_declared_skips_recompilation_in_registry_warmup(monkeypatch):
    batches = []
    monkeypatch.setattr(
        tilelang, "prepare", lambda calls, **kw: batches.append(list(calls))
    )
    registry = JitWarmupRegistry(vllm_config=None)
    calls_invocations = 0

    def first_calls():
        nonlocal calls_invocations
        calls_invocations += 1
        return [prepare.call(_gemm, 1024, 512, block_M=16)]

    with registry.activate():
        prepare.declare(first_calls)
    assert prepare.prepare_declared(registry) == 1
    assert calls_invocations == 1
    assert batches == [[(_gemm, (1024, 512), {"block_M": 16})]]

    # Registering step kernels after prepare_declared and calling registry.warmup()
    # must only compile the newly registered step kernels.
    with registry.activate():
        prepare.declare(lambda: [prepare.call(_rope, 8, 128)])
    registry.warmup()
    assert calls_invocations == 1
    assert batches == [
        [(_gemm, (1024, 512), {"block_M": 16})],
        [(_rope, (8, 128), {})],
    ]


def test_warm_declared_deduplicates_equivalent_closures():
    registry = JitWarmupRegistry(vllm_config=None)
    warmed = []

    def make_layer_warm(hidden: int):
        return lambda: warmed.append(hidden)

    with registry.activate():
        for _ in range(8):
            prepare.declare(warm=make_layer_warm(3072))
        for _ in range(4):
            prepare.declare(warm=make_layer_warm(1536))

    assert prepare.warm_declared(registry) == 2
    assert warmed == [3072, 1536]


def test_bind_declared_reuses_launched_kernels():
    registry = JitWarmupRegistry(vllm_config=None)
    invocations = 0

    def fake_jit(n, k, block_M=16):
        nonlocal invocations
        invocations += 1
        return ("kernel", n, k, block_M)

    bound = []
    with registry.activate():
        for _ in range(6):
            entry = prepare.call(fake_jit, 1024, 512, block_M=32)
            prepare.declare(
                lambda e=entry: [e],
                bind=lambda e=entry: bound.append(prepare.launch(e)),
            )

    prepare.bind_declared(registry)
    assert invocations == 1
    assert len(bound) == 6
    assert all(k == ("kernel", 1024, 512, 32) for k in bound)


def test_declare_modules_declares_child_linear_methods_before_parent_modules():
    import torch.nn as nn

    class _Method:
        def declare_layer_kernels(self, layer: nn.Module) -> None:
            layer.tilelang_shards = 4

    class _Child(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = _Method()

    class _Parent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = _Child()
            self.observed_shards = None

        def declare_kernels(self) -> None:
            self.observed_shards = self.gate.tilelang_shards

    parent = _Parent()
    registry = JitWarmupRegistry(vllm_config=None)
    with registry.activate():
        prepare.declare_modules(parent)
    assert parent.observed_shards == 4
