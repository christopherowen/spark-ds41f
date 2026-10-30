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


def test_registry_warmup_skips_calls_already_prepared_by_prepare_declared(monkeypatch):
    batches = []
    monkeypatch.setattr(
        tilelang, "prepare", lambda calls, **kw: batches.append(list(calls))
    )
    registry = JitWarmupRegistry(vllm_config=None)
    with registry.activate():
        prepare.declare(lambda: [prepare.call(_gemm, 1024, 512, block_M=16)])
    assert prepare.prepare_declared(registry) == 1
    with registry.activate():
        prepare.declare(lambda: [prepare.call(_rope, 8, 512)])
    registry.warmup()
    assert batches == [
        [(_gemm, (1024, 512), {"block_M": 16})],
        [(_rope, (8, 512), {})],
    ]
