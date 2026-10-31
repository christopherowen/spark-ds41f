# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill sequence parallelism: row layout, threshold and CPU collectives.

The collectives are simulated in one process, either by computing every
rank's contribution first or by running the ranks in lockstep threads.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v4_1 import sp_prefill

WORLD_SIZES = (2, 3, 4)
# T mod 3 = 0, 1, 2 around the 4096-row prefill chunk, plus small edge cases.
TOKEN_COUNTS = (1, 2, 4, 7, 4095, 4096, 4097, 4098)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
def test_row_layout_partitions_real_rows(world_size, num_tokens):
    layouts = [
        sp_prefill.SPRows(num_tokens, world_size, rank) for rank in range(world_size)
    ]
    padded = layouts[0].padded_rows
    local = layouts[0].local_rows
    assert padded % world_size == 0
    assert 0 <= padded - num_tokens < world_size
    assert local * world_size == padded
    covered = []
    for rank, layout in enumerate(layouts):
        assert (layout.padded_rows, layout.local_rows) == (padded, local)
        assert layout.start == rank * local
        assert 0 <= layout.real_rows <= local
        covered.extend(range(layout.start, layout.start + layout.real_rows))
        if layout.real_rows < local:
            # Only trailing ranks own padding, and nothing real follows it.
            assert all(later.real_rows == 0 for later in layouts[rank + 1 :])
    assert covered == list(range(num_tokens))
    assert sum(local - layout.real_rows for layout in layouts) == padded - num_tokens


@pytest.mark.parametrize("num_tokens", (4096, 4097, 4098))
def test_row_layout_for_the_prefill_chunk_at_tp3(num_tokens):
    last = sp_prefill.SPRows(num_tokens, 3, 2)
    assert last.padded_rows == 4098
    assert last.local_rows == 1366
    assert last.start == 2732
    assert last.real_rows == num_tokens - 2732


def test_plan_rows_threshold():
    assert sp_prefill.plan_rows(4096, 0, 3, 0) is None
    assert sp_prefill.plan_rows(4096, 2048, 1, 0) is None
    assert sp_prefill.plan_rows(2047, 2048, 3, 0) is None
    assert sp_prefill.plan_rows(2048, 2048, 3, 1) == sp_prefill.SPRows(2048, 3, 1)
    with pytest.raises(ValueError):
        sp_prefill.plan_rows(4096, 2048, 3, 3)


def _config(
    *,
    tp=3,
    pp=1,
    capture=48,
    seqs=8,
    spec_tokens=5,
    boundary=False,
    hidden=5120,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            get_hidden_size=lambda: hidden, dtype=torch.bfloat16
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        ),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=capture),
        scheduler_config=SimpleNamespace(max_num_seqs=seqs),
        max_decode_rows=seqs * (1 + (spec_tokens or 0)),
        speculative_config=(
            None
            if spec_tokens is None
            else SimpleNamespace(num_speculative_tokens=spec_tokens)
        ),
        use_request_boundary_checkpoints=boundary,
    )


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("limit", (1, 10_239, 10_240, 2 * 1024 * 1024, 2_088_960))
def test_rows_above_bytes_is_the_first_scattering_forward(world_size, limit):
    rows = sp_prefill.rows_above_bytes(limit, 10_240, world_size)
    for num_tokens in range(max(1, rows - 2 * world_size), rows + 2 * world_size):
        padded = sp_prefill.SPRows(num_tokens, world_size, 0).padded_rows
        assert (padded * 10_240 > limit) == (num_tokens >= rows), num_tokens


def test_resolve_min_rows_follows_the_one_shot_all_reduce(monkeypatch):
    # A 2 MiB RoCE all-reduce takes 204 rows of 5120 bf16; T = 205 pads to a
    # 207-row message at TP 3, the first one it refuses.
    monkeypatch.setattr(
        sp_prefill, "_prefill_sp_capacity_bytes", lambda: 2 * 1024 * 1024
    )
    assert sp_prefill.resolve_min_rows(_config()) == 205
    assert sp_prefill.resolve_min_rows(_config(tp=2)) == 205
    assert sp_prefill.resolve_min_rows(_config(hidden=4096)) == 256
    # The decode sizes still bound it from below.
    assert sp_prefill.resolve_min_rows(_config(capture=512)) == 513


def test_resolve_min_rows(monkeypatch):
    monkeypatch.setattr(sp_prefill, "_prefill_sp_capacity_bytes", lambda: 0)
    # Without a one-shot all-reduce every reduce-scatter scatters: SP starts
    # above the decode sizes, graphs and max_num_seqs * (1 + k).
    assert sp_prefill.resolve_min_rows(_config()) == 49
    assert sp_prefill.resolve_min_rows(_config(capture=512)) == 513
    assert sp_prefill.resolve_min_rows(_config(capture=None, spec_tokens=None)) == 9
    for disabled in (
        _config(tp=1),
        _config(pp=2),
        _config(boundary=True),
    ):
        assert sp_prefill.resolve_min_rows(disabled) == 0
    assert sp_prefill.resolve_min_rows(_config(), static_sequence_parallel=True) == 0


def _simulate_reduce_scatter(monkeypatch, layouts, contributions):
    """Install a reduce-scatter that returns each rank's slice of the sum."""
    total = sum(contributions)
    calls = iter(range(len(layouts)))

    def reduce_scatter(padded, dim):
        rank = next(calls)
        assert dim == 0 and padded is contributions[rank]
        return total.chunk(len(layouts))[rank].clone()

    monkeypatch.setattr(
        sp_prefill, "tensor_model_parallel_reduce_scatter", reduce_scatter
    )


@pytest.mark.parametrize("num_tokens", (9, 10, 11))
def test_reduce_scatter_then_gather_round_trips(monkeypatch, num_tokens):
    world = 3
    full = torch.arange(num_tokens * 2, dtype=torch.float32).reshape(num_tokens, 2)
    layouts = [sp_prefill.SPRows(num_tokens, world, rank) for rank in range(world)]
    # Partial sums whose exact total is `full`, as TP row-parallel outputs.
    contributions = []
    for layout, weight in zip(layouts, (1.0, 2.0, -2.0)):
        padded = layout.empty_padded((2,), torch.float32, torch.device("cpu"))
        assert padded.shape == (layout.padded_rows, 2)
        assert not padded[num_tokens:].any()
        padded[:num_tokens] = full * weight
        contributions.append(padded)
    _simulate_reduce_scatter(monkeypatch, layouts, contributions)
    local = [
        layout.reduce_scatter(padded) for layout, padded in zip(layouts, contributions)
    ]
    for layout, rows in zip(layouts, local):
        assert rows.shape == (layout.local_rows, 2)
        real = layout.real_rows
        torch.testing.assert_close(
            rows[:real], full[layout.start : layout.start + real], rtol=0, atol=0
        )
        assert not rows[real:].any()

    def all_gather(x, dim):
        assert dim == 0
        return torch.cat(local, dim=0)

    monkeypatch.setattr(sp_prefill, "tensor_model_parallel_all_gather", all_gather)
    for layout, rows in zip(layouts, local):
        torch.testing.assert_close(layout.gather(rows), full, rtol=0, atol=0)
    with pytest.raises(ValueError):
        layouts[0].gather(full)
    with pytest.raises(ValueError):
        layouts[0].reduce_scatter(torch.zeros(layouts[0].padded_rows + 1, 2))


def test_embedding_reduce_scatter_has_one_owner_per_token(monkeypatch):
    from vllm.model_executor.layers import vocab_parallel_embedding as embedding

    vocab, width, world = 12, 3, 3
    shard = vocab // world
    table = torch.arange(vocab * width, dtype=torch.float32).reshape(vocab, width)
    table += 1
    ids = torch.tensor([0, 11, 5, 4, 7, 3, 9])
    padded_rows = 9
    buffers = []

    def reduce_scatter(padded, dim):
        assert dim == 0
        buffers.append(padded.clone())
        return padded

    monkeypatch.setattr(
        embedding, "tensor_model_parallel_reduce_scatter", reduce_scatter
    )
    for rank in range(world):
        rows = table[rank * shard : (rank + 1) * shard]
        layer = SimpleNamespace(
            tp_size=world,
            parallel_group=None,
            use_fused_embedding=False,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=rank * shard,
                org_vocab_end_index=(rank + 1) * shard,
                num_org_vocab_padding=0,
                added_vocab_start_index=vocab,
                added_vocab_end_index=vocab,
            ),
            quant_method=SimpleNamespace(
                embedding=lambda _, local_ids, rows=rows: rows[local_ids].clone()
            ),
        )
        embedding.VocabParallelEmbedding.forward_reduce_scatter(layer, ids, padded_rows)
    expected = torch.zeros(padded_rows, width)
    expected[: ids.numel()] = table[ids]
    torch.testing.assert_close(sum(buffers), expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        embedding.VocabParallelEmbedding.forward_reduce_scatter(layer, ids, 8)


@pytest.mark.parametrize("num_tokens", (9, 10, 11))
def test_local_slice_zero_pads_the_last_rank(num_tokens):
    full = torch.arange(num_tokens * 2, dtype=torch.float32).reshape(num_tokens, 2)
    full += 1
    for rank in range(3):
        layout = sp_prefill.SPRows(num_tokens, 3, rank)
        local = layout.local_slice(full)
        real = layout.real_rows
        assert local.shape == (layout.local_rows, 2)
        torch.testing.assert_close(
            local[:real], full[layout.start : layout.start + real], rtol=0, atol=0
        )
        assert not local[real:].any()
    with pytest.raises(ValueError):
        sp_prefill.SPRows(num_tokens, 3, 0).local_slice(full[1:])


def test_activate_scopes_the_rows():
    outer, inner = sp_prefill.SPRows(9, 3, 0), sp_prefill.SPRows(12, 3, 1)
    assert sp_prefill.current() is None
    with sp_prefill.activate(outer):
        assert sp_prefill.current() is outer
        with pytest.raises(RuntimeError), sp_prefill.activate(inner):
            assert sp_prefill.current() is inner
            raise RuntimeError
        assert sp_prefill.current() is outer
    assert sp_prefill.current() is None


def test_gather_many_groups_pynccl_or_falls_back(monkeypatch):
    layout = sp_prefill.SPRows(10, 3, 2)  # padded=12, local=4, real=2
    assert layout.gather_many() == ()

    a = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    b = torch.arange(12, dtype=torch.int64).reshape(4, 3)

    # Fallback path when pynccl is disabled or absent.
    monkeypatch.setattr(
        sp_prefill,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(pynccl_comm=None)
        ),
    )
    monkeypatch.setattr(
        sp_prefill,
        "tensor_model_parallel_all_gather",
        lambda x, dim: x.repeat(3, 1),
    )
    out_a, out_b = layout.gather_many(a, b, padded_rows=(64,))
    assert out_a.shape == (64, 2)
    torch.testing.assert_close(out_a[:10], a.repeat(3, 1)[:10], rtol=0, atol=0)
    assert out_b.shape == (10, 3)
    torch.testing.assert_close(out_b, b.repeat(3, 1)[:10], rtol=0, atol=0)

    # Grouped PyNCCL path.
    events = []

    class _Comm:
        disabled = False

        def group_start(self):
            events.append("group_start")

        def all_gather(self, full, local):
            events.append(("all_gather", tuple(full.shape), tuple(local.shape)))
            full.copy_(local.repeat(3, *([1] * (local.ndim - 1))))

        def group_end(self):
            events.append("group_end")

    monkeypatch.setattr(sp_prefill, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        sp_prefill,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(pynccl_comm=_Comm())
        ),
    )
    out_a2, out_b2 = layout.gather_many(a, b, padded_rows=(64,))
    assert events == [
        "group_start",
        ("all_gather", (12, 2), (4, 2)),
        ("all_gather", (12, 3), (4, 3)),
        "group_end",
    ]
    assert out_a2.shape == (64, 2)
    torch.testing.assert_close(out_a2[:10], out_a[:10], rtol=0, atol=0)
    torch.testing.assert_close(out_b2, out_b, rtol=0, atol=0)
    with pytest.raises(ValueError):
        layout.gather_many(torch.zeros(3, 2))


@pytest.mark.parametrize(
    "num_tokens,expected_q_rows", [(10, 64), (70, 128), (131, 131)]
)
def test_gather_mxfp8_gathers_cast_and_extra_tensors(
    monkeypatch, num_tokens, expected_q_rows
):
    from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows

    world = 3
    layouts = [sp_prefill.SPRows(num_tokens, world, rank) for rank in range(world)]
    local = layouts[0].local_rows
    local_q = max(local, 128)
    full_vals = torch.arange(layouts[0].padded_rows * 64, dtype=torch.uint8).reshape(
        layouts[0].padded_rows, 64
    )
    full_scales = torch.arange(layouts[0].padded_rows * 2, dtype=torch.uint8).reshape(
        layouts[0].padded_rows, 2
    )
    full_weights = torch.arange(
        layouts[0].padded_rows * 6, dtype=torch.float32
    ).reshape(layouts[0].padded_rows, 6)
    full_ids = torch.arange(layouts[0].padded_rows * 6, dtype=torch.int64).reshape(
        layouts[0].padded_rows, 6
    )

    rank_0 = layouts[0]
    vals_0 = torch.zeros((local_q, 64), dtype=torch.uint8)
    scales_0 = torch.zeros((local_q, 2), dtype=torch.uint8)
    vals_0[:local] = full_vals[:local]
    scales_0[:local] = full_scales[:local]
    x_q = MXFP8Rows(
        rows=torch.empty((local, 64), dtype=torch.bfloat16),
        values=vals_0,
        scales=scales_0,
    )

    calls = []

    def fake_gather_many(self, *tensors, padded_rows=()):
        calls.append((len(tensors), padded_rows))
        sources = (full_vals, full_scales, full_weights, full_ids)[: len(tensors)]
        outs = []
        for idx, (t, src) in enumerate(zip(tensors, sources)):
            assert t.shape[0] == local
            rows = padded_rows[idx] if idx < len(padded_rows) else self.num_tokens
            if rows > self.num_tokens:
                buf = src.new_zeros((rows, *src.shape[1:]))
                buf[: self.num_tokens] = src[: self.num_tokens]
                outs.append(buf)
            else:
                outs.append(src[: self.num_tokens])
        return tuple(outs)

    monkeypatch.setattr(sp_prefill.SPRows, "gather_many", fake_gather_many)
    gathered_q, weights, ids = rank_0.gather_mxfp8(
        x_q, full_weights[:local], full_ids[:local]
    )
    assert calls == [(4, (expected_q_rows, expected_q_rows))]
    assert gathered_q.rows.shape == (num_tokens, 64)
    assert gathered_q.rows.dtype == torch.bfloat16
    assert gathered_q.values.shape == (expected_q_rows, 64)
    assert gathered_q.scales.shape == (expected_q_rows, 2)
    torch.testing.assert_close(
        gathered_q.values[:num_tokens], full_vals[:num_tokens], rtol=0, atol=0
    )
    torch.testing.assert_close(
        gathered_q.scales[:num_tokens], full_scales[:num_tokens], rtol=0, atol=0
    )
    if expected_q_rows > num_tokens:
        assert not gathered_q.values[num_tokens:].any()
        assert not gathered_q.scales[num_tokens:].any()
    torch.testing.assert_close(weights, full_weights[:num_tokens], rtol=0, atol=0)
    torch.testing.assert_close(ids, full_ids[:num_tokens], rtol=0, atol=0)
    with pytest.raises(ValueError):
        rank_0.gather_mxfp8(
            MXFP8Rows(
                rows=torch.empty((local + 1, 64), dtype=torch.bfloat16),
                values=vals_0,
                scales=scales_0,
            )
        )


def test_sp_moe_inputs_routes_local_rows_and_skips_duplicate_cast():
    from vllm.models.deepseek_v4_1 import kernels
    from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows, provided_mxfp8
    from vllm.models.deepseek_v4_1.tilelang.router import provided_routes

    layout = sp_prefill.SPRows(10, 3, 0)  # local_rows = 4
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    x_q = MXFP8Rows(
        rows=x,
        values=torch.ones((128, 64), dtype=torch.uint8),
        scales=torch.ones((128, 2), dtype=torch.uint8),
    )
    input_ids = torch.arange(10, dtype=torch.int64)
    local_w = torch.ones((4, 6), dtype=torch.float32)
    local_ids = torch.zeros((4, 6), dtype=torch.int64)
    full_q = MXFP8Rows(
        rows=torch.empty((10, 64), dtype=torch.bfloat16),
        values=torch.ones((128, 64), dtype=torch.uint8),
        scales=torch.ones((128, 2), dtype=torch.uint8),
    )
    full_w = torch.ones((10, 6), dtype=torch.float32)
    full_ids = torch.zeros((10, 6), dtype=torch.int64)
    seen = []

    class _FakeSPRows:
        def gather(self, _):
            raise AssertionError("BF16 gather should not run when x_q is present")

        def gather_mxfp8(self, q, *extra):
            seen.append(("gather_mxfp8", q is x_q, len(extra)))
            return (full_q, full_w, full_ids) if extra else (full_q,)

    fake_sp_rows = _FakeSPRows()

    def route_local(local_x, sp_rows, *, input_ids=None):
        seen.append(("route_local", local_x is x, sp_rows is fake_sp_rows, input_ids))
        return local_w, local_ids

    mhc = SimpleNamespace(
        cast_rows=lambda _: (_ for _ in ()).throw(
            AssertionError("mhc.cast_rows should not run when x_q is present")
        )
    )
    ffn = SimpleNamespace(
        experts=SimpleNamespace(router=SimpleNamespace(route_local=route_local))
    )
    with kernels.sp_moe_inputs(fake_sp_rows, x, x_q, ffn, input_ids, mhc) as rows:
        assert rows is full_q.rows
        assert provided_mxfp8(rows) is full_q
        assert provided_routes(rows) == (full_w, full_ids)
    with pytest.raises(RuntimeError):
        provided_mxfp8(full_q.rows)
    assert provided_routes(full_q.rows) is None
    assert seen == [
        ("route_local", True, True, input_ids),
        ("gather_mxfp8", True, 2),
    ]

    # Without route_local, gathers MXFP8 only.
    seen.clear()
    ffn_no_router = SimpleNamespace(experts=SimpleNamespace(router=object()))
    with kernels.sp_moe_inputs(
        _FakeSPRows(), x, x_q, ffn_no_router, input_ids, mhc
    ) as rows:
        assert rows is full_q.rows
        assert provided_mxfp8(rows) is full_q
        assert provided_routes(rows) is None
    with pytest.raises(RuntimeError):
        provided_mxfp8(full_q.rows)
    assert seen == [("gather_mxfp8", True, 0)]

    # When x_q is None (CPU stand-in tests), falls back to BF16 gather + cast_rows.
    fallback_rows = torch.randn(10, 64, dtype=torch.bfloat16)
    cast_called = []
    sp_fallback = SimpleNamespace(gather=lambda local_x: fallback_rows)
    mhc_fallback = SimpleNamespace(
        cast_rows=lambda r: (cast_called.append(r is fallback_rows), None)[1]
    )
    with kernels.sp_moe_inputs(
        sp_fallback, x, None, ffn, input_ids, mhc_fallback
    ) as rows:
        assert rows is fallback_rows
    assert cast_called == [True]


def test_moe_reduce_scatter_sums_shared_and_routed_before_the_scatter(monkeypatch):
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    events = []

    def reduce_scatter(states, dim):
        events.append(("reduce_scatter", states.clone(), dim))
        return states.chunk(3)[1]

    monkeypatch.setattr(
        moe_runner, "tensor_model_parallel_reduce_scatter", reduce_scatter
    )
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(
            is_sequence_parallel=False,
            skip_final_all_reduce=False,
            in_dtype=torch.float32,
        ),
        router=object(),
        _l2_prefetch_pre_reduce_hook=lambda rows: events.append(("hook", rows)),
    )
    shared = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    fused = torch.full((10, 2), 0.5)
    output = moe_runner.MoERunner._reduce_scatter_final_output(
        runner, shared, fused, 12, None, False
    )
    assert events[0] == ("hook", 10)
    _, states, dim = events[1]
    assert dim == 0 and states.shape == (12, 2)
    torch.testing.assert_close(states[:10], shared + fused, rtol=0, atol=0)
    assert not states[10:].any()
    torch.testing.assert_close(output, states[4:8], rtol=0, atol=0)
    with pytest.raises(NotImplementedError):
        moe_runner.MoERunner._reduce_scatter_final_output(
            runner, shared, fused, 12, None, True
        )


def _hook_runner(events):
    """A runner's attributes for ``_reduce_scatter_final_output`` with the model's
    reduce-scatter hook installed (it records its call and fires ``then``)."""

    def reduce(write, rows, padded_rows, row_shape, dtype, device, then):
        events.append(("reduce", write, rows, padded_rows, tuple(row_shape), dtype))
        then()
        return "parts"

    return dict(
        moe_config=SimpleNamespace(
            is_sequence_parallel=False,
            skip_final_all_reduce=False,
            in_dtype=torch.bfloat16,
        ),
        router=object(),
        _l2_prefetch_pre_reduce_hook=lambda rows: events.append(("hook", rows)),
        _sp_project_reduce_scatter_hook=reduce,
    )


def test_moe_reduce_scatter_hook_writes_with_the_deferred_combine():
    """The model's hook gets the experts' deferred combine as its writer, and
    nothing reads the unwritten output; without a deferred combine it refuses."""
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    events = []
    runner = SimpleNamespace(**_hook_runner(events))
    unwritten = torch.empty((10, 2), dtype=torch.bfloat16)

    def combine(start, stop, out):
        raise AssertionError("only the hook calls the combine")

    output = moe_runner.MoERunner._reduce_scatter_final_output(
        runner, None, unwritten, 12, None, False, combine
    )
    assert output == "parts"
    assert events == [
        ("reduce", combine, 10, 12, (2,), torch.bfloat16),
        ("hook", 10),
    ]
    with pytest.raises(NotImplementedError):
        moe_runner.MoERunner._reduce_scatter_final_output(
            runner, None, unwritten, 12, None, False
        )


def test_moe_runner_defers_a_folded_combine_to_the_hook(monkeypatch):
    """A folded SP forward with the hook installed asks the experts to defer their
    combine inside the MoE op, hands that combine to the hook, and resets the
    experts' call state after the op; without the hook nothing is deferred."""
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    monkeypatch.setattr(
        moe_runner, "tensor_model_parallel_reduce_scatter", lambda states, dim: "rs"
    )
    runner_cls = moe_runner.MoERunner
    events = []
    experts = SimpleNamespace(
        fold_shared=False, combine_addend=None, defer_combine=False,
        deferred_combine=None,
    )  # fmt: skip
    seen = []

    def combine(start, stop, out):
        raise AssertionError("only the hook calls the combine")

    def entry(hidden, logits, shared_in, ids, name, unpadded, dtype):
        seen.append((experts.fold_shared, experts.defer_combine))
        if experts.defer_combine:
            experts.deferred_combine = combine
        experts.combine_addend = "shared rows"
        return torch.zeros_like(hidden), torch.empty_like(hidden)

    class Runner:
        forward = runner_cls.forward
        apply_routed_input_transform = runner_cls.apply_routed_input_transform
        _maybe_pad_hidden_states = runner_cls._maybe_pad_hidden_states
        _maybe_reduce_routed_output_before_transform = (
            runner_cls._maybe_reduce_routed_output_before_transform
        )
        _maybe_reduce_shared_expert_output = (
            runner_cls._maybe_reduce_shared_expert_output
        )
        _maybe_apply_routed_scale_to_output = (
            runner_cls._maybe_apply_routed_scale_to_output
        )
        apply_routed_output_transform = runner_cls.apply_routed_output_transform
        _reduce_scatter_final_output = runner_cls._reduce_scatter_final_output

    runner = Runner()
    for name, value in _hook_runner(events).items():
        setattr(runner, name, value)
    runner.moe_config.hidden_dim = 2
    runner.routed_input_transform = runner.routed_output_transform = None
    runner.routed_scaling_factor = 1.0
    runner._shared_experts = object()
    runner._quant_method = SimpleNamespace(
        skip_forward_padding=True, has_unpadded_output=False, output_dtype=None
    )
    runner._fused_output_is_reduced = False
    runner._shared_output_fold = lambda shared_in: experts
    runner._forward_entry = entry
    runner._encode_layer_name = lambda: "layer"
    hidden = torch.ones((10, 2), dtype=torch.bfloat16)
    assert runner.forward(hidden, hidden, reduce_scatter_rows=12) == "parts"
    assert seen == [(True, True)]
    assert events == [
        ("reduce", combine, 10, 12, (2,), torch.bfloat16),
        ("hook", 10),
    ]
    assert vars(experts) == dict(
        fold_shared=False, combine_addend=None, defer_combine=False,
        deferred_combine=None,
    )  # fmt: skip
    del runner._sp_project_reduce_scatter_hook
    assert runner.forward(hidden, hidden, reduce_scatter_rows=12) == "rs"
    assert seen[-1] == (True, False)


def test_tilelang_experts_defer_only_a_folded_combine(monkeypatch):
    """With ``defer_combine`` the TileLang experts run their plan without an output
    (everything but the combine), leave ``output`` unwritten and keep the plan's
    deferred combine; an unfolded call is never deferred."""
    from vllm.model_executor.layers.fused_moe import tilelang as tilelang_experts
    from vllm.model_executor.layers.fused_moe.tilelang import TileLangExperts

    # The tokens' cast comes from the mHC entry; the stand-in plan reads none.
    monkeypatch.setattr(tilelang_experts, "_provided_mxfp8", lambda rows: rows)
    runs = []

    def run(x, ids, weights, gemm_weights, out, workspace, shared):
        runs.append((out, shared))
        return "combine" if out is None else out

    experts = SimpleNamespace(
        _plan=SimpleNamespace(run=run),
        _tails=(None, None),
        _bases=(None, None),
        _codes_2bit=(None, None),
        _narrow=(None, None),
        w1_scale=None,
        w2_scale=None,
        fold_shared=True,
        combine_addend="shared rows",
        defer_combine=True,
        deferred_combine=None,
    )
    output = torch.full((4, 2), float("nan"), dtype=torch.bfloat16)
    call = dict(
        output=output,
        hidden_states=torch.zeros((4, 2), dtype=torch.bfloat16),
        w1=None,
        w2=None,
        topk_weights=torch.zeros((4, 6)),
        topk_ids=torch.zeros((4, 6), dtype=torch.int32),
        activation=None,
        global_num_experts=32,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=torch.zeros(16, dtype=torch.bfloat16),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    TileLangExperts.apply(experts, **call)
    assert runs == [(None, "shared rows")]
    assert experts.deferred_combine == "combine"
    assert output.isnan().all()
    experts.defer_combine, experts.deferred_combine = False, None
    TileLangExperts.apply(experts, **call)
    assert runs[-1] == (output, "shared rows") and experts.deferred_combine is None
    experts.fold_shared, experts.defer_combine = False, True
    with pytest.raises(AssertionError):
        TileLangExperts.apply(experts, **call)


def test_engram_gates_only_the_real_local_rows(monkeypatch):
    # Engram's geometry and token map are b12x's; its gate is TileLang's.
    pytest.importorskip("b12x")
    pytest.importorskip("tilelang")
    from vllm.models.deepseek_v4_1.common import engram as engram_ops
    from vllm.models.deepseek_v4_1.tilelang import columns
    from vllm.models.deepseek_v4_1.tilelang import engram as tl_engram
    from vllm.models.deepseek_v4_1.tilelang.engram_hash import IMAGE_SENTINEL

    tokens, streams, width = 10, 2, 4

    def gate(state, kv, norm_weights, token_ids, out, eps, hc_mult):
        assert kv.shape[0] == state.shape[0] == out.shape[0]
        assert token_ids.shape == (state.shape[0],)
        text = (token_ids != IMAGE_SENTINEL)[:, None]
        return out.copy_(state + kv[:, : state.shape[1]] * text)

    # The TileLang gate and pass-through copy, stubbed as torch.
    monkeypatch.setattr(tl_engram, "engram_gate", gate)
    monkeypatch.setattr(columns, "copy_columns", lambda src, dst: dst.copy_(src))
    monkeypatch.setattr(engram_ops, "tensor_model_parallel_all_reduce", lambda x: x)
    staged = torch.randn(tokens, 6)
    engram = SimpleNamespace(
        embed_tokens=SimpleNamespace(table_memory="device", columns=None),
        overlap_epochs=None,
        staged_rows=staged,
        wkv=lambda rows: rows.repeat(1, 2),
        l2_prefetch_after_projection=None,
        norm_weights=None,
        eps=1e-6,
        hc_mult=streams,
    )
    hidden = torch.randn(tokens, streams, width)
    hash_ids = torch.zeros(tokens, 24, dtype=torch.int64)
    token_ids = torch.where(torch.arange(tokens) % 3 != 1, 7, IMAGE_SENTINEL)
    full = engram_ops.Engram.forward(engram, hidden, hash_ids, token_ids)
    for rank in range(3):
        layout = sp_prefill.SPRows(tokens, 3, rank)
        local = layout.local_slice(hidden)
        output = engram_ops.Engram.forward(
            engram, local, hash_ids, token_ids, sp_rows=layout
        )
        real = layout.real_rows
        assert output.shape == local.shape
        torch.testing.assert_close(
            output[:real], full[layout.start : layout.start + real], rtol=0, atol=0
        )
        torch.testing.assert_close(output[real:], local[real:], rtol=0, atol=0)


# --- Whole-model dataflow on lockstep CPU ranks -----------------------------

# Rank shares of every TP partial sum; the shares sum to one.
_SHARES = (0.5, 0.25, 0.25)


class _World:
    """Lockstep all-gather and reduce-scatter for ranks run in threads."""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=60)
        self.slots = [None] * size
        self.local = threading.local()

    @property
    def rank(self):
        return self.local.rank

    def _exchange(self, x):
        self.slots[self.rank] = x
        self.barrier.wait()
        values = list(self.slots)
        self.barrier.wait()
        return values

    def all_gather(self, x, dim):
        assert dim == 0
        return torch.cat(self._exchange(x), dim=0)

    def reduce_scatter(self, x, dim):
        assert dim == 0
        return sum(self._exchange(x)).chunk(self.size)[self.rank].clone()

    def run(self, fn):
        def rank_main(rank):
            self.local.rank = rank
            return fn(rank)

        with ThreadPoolExecutor(self.size) as pool:
            futures = [pool.submit(rank_main, rank) for rank in range(self.size)]
            return [future.result() for future in futures]


class _Mix:
    """Row-wise stand-in for the native mHC kernels."""

    def collapse(self, hidden, mix=None):
        if mix is None:
            return hidden.mean(dim=1)
        return (hidden * mix[:, :, None]).sum(dim=1)

    def post(self, x, residual, post, comb):
        return residual + x[:, None, :] * 0.25

    def pre(self, residual, fn, scale, base, norm, pre, **previous):
        if previous.get("previous_output") is not None:
            residual = self.post(previous["previous_output"], residual, None, None)
        if residual.ndim == 2:
            residual = residual[:, None, :].expand(-1, 4, -1).clone()
        rows = residual.shape[0]
        mix = residual.new_zeros((rows, 4))
        mix[:, 0] = 0.75
        mix[:, 1] = 0.25
        return (
            residual,
            residual.new_full((rows, 4), 0.25),
            torch.eye(4).expand(rows, 4, 4),
            residual[:, 0, :] * 0.5 + residual[:, 1, :] * 0.5,
            mix,
            None,  # no MXFP8 cast: the stand-in projections take any rows
        )

    def post_pre(self, x, residual, post, comb, fn, scale, base, norm, pre):
        return self.pre(self.post(x, residual, post, comb), fn, scale, base, norm, pre)

    def cast_rows(self, rows):
        return None


def _partial(world, rows, result):
    """This rank's zero-padded TP share of ``result``, reduce-scattered."""
    padded = result.new_zeros((rows, *result.shape[1:]))
    padded[: result.shape[0]] = result * _SHARES[world.rank]
    return world.reduce_scatter(padded, 0)


class _Attention(nn.Module):
    """Honors the SP contract of DeepseekV4Attention._forward."""

    def __init__(self, index, world, local_rows):
        super().__init__()
        self.index, self.world, self.local_rows = index, world, local_rows

    def forward(self, positions, hidden, scaling):
        rows = sp_prefill.current()
        if rows is not None:
            self.local_rows.append(hidden.shape[0])
            hidden = rows.gather(hidden)
        # The cross-row term shows whether attention saw every row.
        result = (
            hidden * 0.5
            + hidden.mean(dim=0)
            + positions[:, None].float() * 0.01
            + self.index
        )
        if rows is None:
            return result
        return _partial(self.world, rows.padded_rows, result)


class _Experts(nn.Module):
    def __init__(self, world):
        super().__init__()
        self.world = world

    def forward(self, hidden, ids, *, reduce_scatter_rows=None):
        result = hidden * 0.25 + hidden.sum(dim=0) * 0.01 + ids[:, None].float() * 0.02
        if reduce_scatter_rows is None:
            return result
        return _partial(self.world, reduce_scatter_rows, result)


class _Layer(nn.Module):
    def __init__(self, index, world, local_rows):
        super().__init__()
        self.engram = None
        self.hc_mult = 4
        self.mhc = _Mix()
        self.attn = _Attention(index, world, local_rows)
        self.ffn = _Experts(world)
        for name in (
            "hc_attn_fn_broadcast",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_attn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
            "hc_ffn_base",
        ):
            setattr(self, name, torch.ones(1))
        self.attn_norm = self.ffn_norm = SimpleNamespace(weight=torch.ones(2))

    def forward(self, *args, **kwargs):
        from vllm.models.deepseek_v4_1.nvidia import model as native

        return native.DeepseekV4DecoderLayer.forward(self, *args, **kwargs)


class _Embedding:
    """Vocabulary-parallel embedding: rank r owns ids with id % N == r."""

    def __init__(self, table, world):
        self.table, self.world = table, world

    def forward_reduce_scatter(self, ids, padded_rows):
        owned = (ids % self.world.size == self.world.rank)[:, None]
        padded = self.table.new_zeros((padded_rows, self.table.shape[1]))
        padded[: ids.numel()] = self.table[ids] * owned
        return self.world.reduce_scatter(padded, 0)


@pytest.mark.parametrize("num_tokens", (9, 10, 11))
@pytest.mark.parametrize("use_embeds", (False, True))
def test_model_forward_matches_full_rows(monkeypatch, num_tokens, use_embeds):
    pytest.importorskip("b12x")
    from vllm.models.deepseek_v4_1.nvidia import model as native

    world = _World(3)
    monkeypatch.setattr(
        native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(native, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(native, "get_tensor_model_parallel_rank", lambda: world.rank)
    monkeypatch.setattr(native.l2_prefetch, "ENABLED", False)
    monkeypatch.setattr(
        sp_prefill, "tensor_model_parallel_all_gather", world.all_gather
    )
    monkeypatch.setattr(
        sp_prefill, "tensor_model_parallel_reduce_scatter", world.reduce_scatter
    )
    # Float64 keeps the reordered TP partial sums well inside the tolerance.
    table = torch.randn(16, 2, dtype=torch.float64)
    ids = (torch.arange(num_tokens) * 7) % 16
    positions = torch.arange(100, 100 + num_tokens)

    def run(min_rows, local_rows):
        target = SimpleNamespace(
            embed_input_ids=lambda input_ids: table[input_ids],
            embed_tokens=_Embedding(table, world),
            prefill_sp_min_rows=min_rows,
            prefill_sp_unreduced=False,
            disk_engram=False,
            engram_hash=None,
            start_layer=0,
            end_layer=3,
            ced_decoder_start=None,
            layers=nn.ModuleList(_Layer(i, world, local_rows) for i in range(3)),
            aux_hidden_state_layers={2, 3},
            # L2 prefetch is on by default on SM121; its plans are built.
            _l2pf_ready=True,
            # The sink detector's column buffer: none, the detector is off here.
            _sink_columns=None,
            # Graph output buffers: none, nothing is captured here.
            _graph_out=lambda index, rows: None,
            # GLP steering: none, unset here.
            glp=None,
            norm=lambda hidden: hidden * 2,
        )
        output, aux = native.DeepseekV4Model.forward(
            target,
            ids,
            positions,
            None,
            inputs_embeds=table[ids] if use_embeds else None,
        )
        return output, aux

    expected, expected_aux = run(0, [])
    local_rows = [[] for _ in range(3)]
    results = world.run(lambda rank: run(4, local_rows[rank]))
    layout = sp_prefill.SPRows(num_tokens, 3, 0)
    for rank, (output, aux) in enumerate(results):
        # Attention received only this rank's rows in every layer.
        assert local_rows[rank] == [layout.local_rows] * 3
        torch.testing.assert_close(output, expected, rtol=1e-9, atol=1e-9)
        assert len(aux) == len(expected_aux) == 2
        for actual, value in zip(aux, expected_aux):
            torch.testing.assert_close(actual, value, rtol=1e-9, atol=1e-9)


def test_dspark_tp_defaults_enabled_for_multi_rank(monkeypatch):
    from vllm.models.deepseek_v4_1.nvidia import dspark

    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    assert not dspark._vocab_parallel_greedy()
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 4)
    assert dspark._vocab_parallel_greedy()
    assert not hasattr(dspark, "_use_sequence_parallel")
