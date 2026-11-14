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

    ffn = SimpleNamespace(
        experts=SimpleNamespace(router=SimpleNamespace(route_local=route_local))
    )
    with kernels.sp_moe_inputs(fake_sp_rows, x, x_q, ffn, input_ids) as rows:
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

    # Neither missing input is a request for an alternate execution path.
    with pytest.raises(ValueError, match="producer-provided MXFP8"):
        with kernels.sp_moe_inputs(fake_sp_rows, x, None, ffn, input_ids):
            pytest.fail("missing cast entered the MoE body")
    ffn_no_router = SimpleNamespace(experts=SimpleNamespace(router=object()))
    with pytest.raises(ValueError, match="router with route_local"):
        with kernels.sp_moe_inputs(fake_sp_rows, x, x_q, ffn_no_router, input_ids):
            pytest.fail("missing router entered the MoE body")


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
    # The gate is TileLang's.
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
        normalized = residual[:, 0, :] * 0.5 + residual[:, 1, :] * 0.5
        return (
            residual,
            residual.new_full((rows, 4), 0.25),
            torch.eye(4).expand(rows, 4, 4),
            normalized,
            mix,
            self.cast_rows(normalized),
        )

    def post_pre(self, x, residual, post, comb, fn, scale, base, norm, pre):
        return self.pre(self.post(x, residual, post, comb), fn, scale, base, norm, pre)

    def cast_rows(self, rows):
        from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows

        # The CPU projections read real-valued rows. Carry the same cast
        # interface as mHC; actual packed-byte math has separate kernel tests.
        return MXFP8Rows(
            rows,
            torch.zeros_like(rows, dtype=torch.uint8),
            torch.zeros((rows.shape[0], 1), dtype=torch.uint8),
        )


def _simulated_mxfp8_gather(layout, x_q, *extra):
    # Match the grouped cast-and-route gather contract, while CPU stand-in
    # projections consume the gathered real-valued rows instead of GPU bytes.
    rows, *gathered = layout.gather_many(x_q.rows, *extra)
    return (_Mix().cast_rows(rows), *gathered)


def _cpu_route_local(x, layout, *, input_ids=None):
    return (
        torch.ones((x.shape[0], 1), dtype=torch.float32),
        torch.zeros((x.shape[0], 1), dtype=torch.int64),
    )


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
        self.experts = SimpleNamespace(
            router=SimpleNamespace(route_local=_cpu_route_local)
        )

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
    from vllm.models.deepseek_v4_1.nvidia import model as native

    monkeypatch.setattr(
        sp_prefill.SPRows, "gather_mxfp8", _simulated_mxfp8_gather
    )
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


def test_dspark_fused_greedy_enabled_at_every_tp_size(monkeypatch):
    from vllm.models.deepseek_v4_1.nvidia import dspark

    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    assert dspark._vocab_parallel_greedy()
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 4)
    assert dspark._vocab_parallel_greedy()
    assert not hasattr(dspark, "_use_sequence_parallel")


def test_model_forward_sp_sink_columns_with_rank_order_parts(monkeypatch):
    from vllm.models.deepseek_v4_1.nvidia import model as native
    from vllm.models.deepseek_v4_1.tilelang.collectives import RankOrderParts

    monkeypatch.setattr(
        sp_prefill.SPRows, "gather_mxfp8", _simulated_mxfp8_gather
    )
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
    table = torch.randn(16, 2, dtype=torch.float32)
    ids = torch.arange(9)

    def run(_rank):
        layers = nn.ModuleList(_Layer(i, world, []) for i in range(2))
        orig_ffn = layers[-1].ffn.forward
        layers[-1].ffn.forward = lambda hidden, i, **kw: RankOrderParts(
            orig_ffn(hidden, i, **kw).unsqueeze(0)
        )

        def post_collapse_norm(
            x, residual, post, comb, pre, weight, eps, *, columns=None, column=0, out=None
        ):
            if isinstance(x, RankOrderParts):
                x = x.packed.sum(dim=0)
            recon = layers[-1].mhc.post(x, residual, post, comb)
            if columns is not None:
                columns.copy_(recon[:, :, column].float())
            return layers[-1].mhc.collapse(recon, pre) * 2

        layers[-1].mhc.post_collapse_norm = post_collapse_norm
        sink = torch.zeros((16, 4), dtype=torch.float32)
        target = SimpleNamespace(
            embed_input_ids=lambda input_ids: table[input_ids],
            embed_tokens=_Embedding(table, world),
            prefill_sp_min_rows=4,
            prefill_sp_unreduced=False,
            disk_engram=False,
            engram_hash=None,
            start_layer=0,
            end_layer=2,
            ced_decoder_start=None,
            layers=layers,
            aux_hidden_state_layers=set(),
            _l2pf_ready=True,
            _sink_columns=sink,
            _sink_column=0,
            hc_mult=4,
            rms_norm_eps=1e-6,
            _graph_out=lambda index, rows: None,
            glp=None,
            norm=SimpleNamespace(weight=torch.ones(2)),
        )
        out = native.DeepseekV4Model.forward(target, ids, ids, None)
        return out, sink[:9].clone()

    results = world.run(run)
    for out, sink in results:
        assert out.shape == (9, 2)
        assert sink.abs().sum() > 0



def test_engram_slices():
    assert sp_prefill.engram_slices(2048) == 1
    assert sp_prefill.engram_slices(6143) == 1
    assert sp_prefill.engram_slices(6144) == 2
    assert sp_prefill.engram_slices(8192) == 2
    assert sp_prefill.engram_slices(16384) == 2


class _RowExchangeFabric:
    """Lockstep multi-rank PyNcclCommunicator harness for CPU threads."""

    def __init__(self, world: int):
        import collections

        self.world = world
        self.barrier = threading.Barrier(world, timeout=30)
        self.ops: list[list] = [[] for _ in range(world)]
        self.pairs: set[frozenset] = set()
        self._collections = collections

    def comm(self, rank: int, events: list | None = None):
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        comm = PyNcclCommunicator.__new__(PyNcclCommunicator)
        comm.disabled, comm.world_size, comm.rank = False, self.world, rank
        comm.group_start = lambda: None
        comm.send = lambda t, peer, stream: self.ops[rank].append(("send", peer, t))
        comm.recv = lambda t, peer, stream: self.ops[rank].append(("recv", peer, t))
        comm.group_end = lambda: self._end(rank)
        orig_a2a = comm.all_to_all_rows

        def tracked_a2a(output, input_, rows, stream=None, *, copy_self=True):
            assert not copy_self, "project_exchange_rows must pass copy_self=False"
            lo = output.storage_offset() // output.stride(1)
            hi = lo + output.shape[1]
            if events is not None:
                events.append(("exchange", lo, hi))
            return orig_a2a(output, input_, rows, stream=stream or object(), copy_self=copy_self)

        comm.all_to_all_rows = tracked_a2a
        return comm

    def _end(self, rank: int) -> None:
        self.barrier.wait()
        if rank == 0:
            sends = self._collections.defaultdict(self._collections.deque)
            for source, ops in enumerate(self.ops):
                for kind, peer, tensor in ops:
                    if kind == "send":
                        sends[(source, peer)].append(tensor.clone())
            for destination, ops in enumerate(self.ops):
                for kind, peer, tensor in ops:
                    if kind == "recv":
                        sent = sends[(peer, destination)].popleft()
                        assert sent.shape == tensor.shape, (peer, destination)
                        tensor.copy_(sent)
                        self.pairs.add(frozenset((peer, destination)))
            assert not any(sends.values()), "a send nobody received"
            self.ops = [[] for _ in range(self.world)]
        self.barrier.wait()


@pytest.mark.parametrize("topology", [None, "direct"])
@pytest.mark.parametrize("world", [3, 4])
@pytest.mark.parametrize("short", [0, 3, 9, 10, 25])
@pytest.mark.parametrize("slices", [1, 2, 3])
def test_project_exchange_rows_bit_identical_and_overlapped(
    monkeypatch, topology, world, short, slices
):
    """``SPRows.project_exchange_rows`` projects only real rows, exchanges each slice's
    peer blocks before projecting this rank's own block directly into ``out[rank]``
    with ``copy_self=False``, runs ``then()`` after the last projection, and yields
    bit-identical BF16 blocks to a full-row projection + ``all_to_all_rows`` across
    TP3 and TP4 (both ring4 relay and direct topologies, including padded/empty tail
    blocks)."""
    from vllm.distributed.device_communicators.pynccl import ring4_relay

    if topology is None:
        monkeypatch.delenv("SPARKNET_ROCE_TOPOLOGY", raising=False)
    else:
        monkeypatch.setenv("SPARKNET_ROCE_TOPOLOGY", topology)
    ring4_relay.cache_clear()

    rows_per_rank = 10
    total = max(1, world * rows_per_rank - short)
    block_cols = 6
    gen = torch.Generator().manual_seed(100 + world * 17 + short * 3 + slices)
    inputs = [
        torch.randn((total, block_cols), generator=gen, dtype=torch.bfloat16)
        for _ in range(world)
    ]
    fabric = _RowExchangeFabric(world)
    local_comm = threading.local()
    monkeypatch.setattr(
        sp_prefill,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(
                pynccl_comm=local_comm.comm, b12x_ar_comm=None
            )
        ),
    )

    outputs: list[torch.Tensor | None] = [None] * world
    events_by_rank: list[list] = [[] for _ in range(world)]
    then_at: list[int] = [-1] * world
    errors = []

    def run(rank: int) -> None:
        try:
            events = events_by_rank[rank]
            local_comm.comm = fabric.comm(rank, events)
            sp = sp_prefill.SPRows(num_tokens=total, world_size=world, rank=rank)

            def project(start: int, stop: int, out: torch.Tensor) -> None:
                events.append(("project", start, stop))
                out.copy_(inputs[rank][start:stop])

            outputs[rank] = sp.project_exchange_rows(
                project,
                (block_cols,),
                torch.bfloat16,
                torch.device("cpu"),
                slices=slices,
                then=lambda: then_at.__setitem__(rank, len(events)),
            )
        except BaseException as exc:
            errors.append((rank, exc))
            fabric.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ring4_relay.cache_clear()
    assert not errors, errors

    for me in range(world):
        sp = sp_prefill.SPRows(num_tokens=total, world_size=world, rank=me)
        local = sp.local_rows
        real = sp.real_rows
        got = outputs[me]
        assert got is not None and got.dtype == torch.bfloat16
        assert got.shape == (world, local, block_cols)
        for source in range(world):
            expected = inputs[source][me * local : me * local + real]
            assert torch.equal(got[source, :real].view(torch.int16), expected.view(torch.int16))
        events = events_by_rank[me]
        calls = [(s, e) for kind, s, e in events if kind == "project"]
        projected = sorted(i for s, e in calls for i in range(s, e))
        assert projected == list(range(total))
        assert then_at[me] == len(events)
        for kind, lo, hi in events:
            if kind != "exchange":
                continue
            at = events.index((kind, lo, hi))
            for owner in range(world):
                start = owner * local + lo
                stop = min(owner * local + hi, total)
                if stop > start:
                    idx = events.index(("project", start, stop))
                    assert (idx > at) == (owner == me), (me, owner, lo, hi)


@pytest.mark.parametrize("world", [3, 4])
@pytest.mark.parametrize("tokens", [5, 10, 12])
@pytest.mark.parametrize("blocked_wkv", [False, True])
@pytest.mark.parametrize("with_previous_output", [False, True])
def test_engram_sp_forward_bit_identical_blocked_and_replicated(
    monkeypatch, world, tokens, blocked_wkv, with_previous_output
):
    """Under ``sp_rows``, ``Engram.forward`` produces bit-identical BF16 outputs to
    the full-row non-SP forward across TP3 and TP4 in both blocked column-parallel
    mode (``_PaddedColumnParallelLinear`` via ``project_exchange_rows``) and
    replicated mode, with and without ``previous_output=RankOrderParts``, including
    padded and wholly empty tail ranks and ``IMAGE_SENTINEL`` tokens."""
    from vllm.models.deepseek_v4_1.common import engram as engram_ops
    from vllm.models.deepseek_v4_1.tilelang import columns
    from vllm.models.deepseek_v4_1.tilelang import engram as tl_engram
    from vllm.models.deepseek_v4_1.tilelang import mhc as tl_mhc
    from vllm.models.deepseek_v4_1.tilelang.collectives import RankOrderParts
    from vllm.models.deepseek_v4_1.tilelang.engram_hash import IMAGE_SENTINEL

    streams, width = 2, 4
    in_cols = 6
    block_cols = 4
    logical_cols = world * block_cols

    def gate_ref(state, kv_rows, token_ids):
        text = (token_ids != IMAGE_SENTINEL)[:, None].to(state.dtype)
        return (state.float() + kv_rows[:, : state.shape[1]].float() * text.float()).to(
            torch.bfloat16
        )

    def fake_gate(state, kv, norm_weights, token_ids, out, eps, hc_mult):
        assert state.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
        out.copy_(gate_ref(state, kv, token_ids))

    def fake_gate_blocked(state, kv, norm_weights, token_ids, out, eps, hc_mult):
        assert state.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
        n = state.shape[0]
        flat_kv = (
            kv[:, :n, :]
            .permute(1, 0, 2)
            .contiguous()
            .view(n, world * block_cols)[:, :logical_cols]
        )
        out.copy_(gate_ref(state, flat_kv, token_ids))

    def fake_mhc_post(prev, res, post_m, res_m, out):
        if prev.ndim == 3:
            prev = prev.float().sum(dim=0).to(torch.bfloat16)
        recon = (
            res.float() * res_m[:, :, 0:1]
            + prev[:, None, :].float() * post_m[:, :, None]
        ).to(torch.bfloat16)
        out.copy_(recon)

    def fake_post_gate(
        prev, res, post_m, res_m, kv, norm_weights, token_ids, out, eps, hc_mult
    ):
        recon = torch.empty_like(res)
        fake_mhc_post(prev, res, post_m, res_m, recon)
        fake_gate(
            recon.flatten(1),
            kv,
            norm_weights,
            token_ids,
            out.view(-1, streams * width),
            eps,
            hc_mult,
        )

    def fake_post_gate_blocked(
        prev, res, post_m, res_m, kv, norm_weights, token_ids, out, eps, hc_mult
    ):
        recon = torch.empty_like(res)
        fake_mhc_post(prev, res, post_m, res_m, recon)
        fake_gate_blocked(
            recon.flatten(1),
            kv,
            norm_weights,
            token_ids,
            out.view(-1, streams * width),
            eps,
            hc_mult,
        )

    monkeypatch.setattr(tl_engram, "engram_gate", fake_gate)
    monkeypatch.setattr(tl_engram, "engram_gate_blocked", fake_gate_blocked)
    monkeypatch.setattr(tl_engram, "engram_post_gate", fake_post_gate)
    monkeypatch.setattr(tl_engram, "engram_post_gate_blocked", fake_post_gate_blocked)
    monkeypatch.setattr(tl_mhc, "_mhc_post", fake_mhc_post)
    monkeypatch.setattr(columns, "copy_columns", lambda src, dst: dst.copy_(src))
    monkeypatch.setattr(engram_ops, "tensor_model_parallel_all_reduce", lambda x: x)
    monkeypatch.setattr(sp_prefill, "ENGRAM_SLICE_TOKENS", 4)

    gen = torch.Generator().manual_seed(200 + world * 13 + tokens)
    staged = torch.randn((tokens, in_cols), generator=gen, dtype=torch.bfloat16)
    weight_full = torch.randn(
        (logical_cols, in_cols), generator=gen, dtype=torch.bfloat16
    )
    hidden = torch.randn((tokens, streams, width), generator=gen, dtype=torch.bfloat16)
    prev_a = torch.randn((tokens, width), generator=gen, dtype=torch.bfloat16)
    prev_b = torch.randn((tokens, width), generator=gen, dtype=torch.bfloat16)
    prev_sum = (prev_a.float() + prev_b.float()).to(torch.bfloat16)
    post_mix = torch.rand((tokens, streams), generator=gen, dtype=torch.float32)
    res_mix = torch.rand((tokens, streams, streams), generator=gen, dtype=torch.float32)
    hash_ids = torch.zeros((tokens, 24), dtype=torch.int64)
    token_ids = torch.where(
        torch.arange(tokens) % 3 != 1,
        torch.tensor(7, dtype=torch.int32),
        torch.tensor(IMAGE_SENTINEL, dtype=torch.int32),
    )

    ref_engram = SimpleNamespace(
        embed_tokens=SimpleNamespace(table_memory="device", columns=None),
        staged_rows=staged,
        wkv=lambda r: (r.float() @ weight_full.float().T).to(torch.bfloat16),
        l2_prefetch_after_projection=None,
        norm_weights=None,
        eps=1e-6,
        hc_mult=streams,
    )
    if with_previous_output:
        full = engram_ops.Engram.forward(
            ref_engram,
            hidden,
            hash_ids,
            token_ids,
            previous_output=prev_sum,
            post_mix=post_mix,
            res_mix=res_mix,
        )
    else:
        full = engram_ops.Engram.forward(ref_engram, hidden, hash_ids, token_ids)

    if not blocked_wkv:
        for rank in range(world):
            layout = sp_prefill.SPRows(tokens, world, rank)
            wkv_rows_seen = []
            prefetch_seen = []

            def rep_wkv(r, seen=wkv_rows_seen):
                seen.append(r.shape[0])
                return (r.float() @ weight_full.float().T).to(torch.bfloat16)

            engram = SimpleNamespace(
                embed_tokens=SimpleNamespace(table_memory="device", columns=None),
                staged_rows=staged,
                wkv=rep_wkv,
                l2_prefetch_after_projection=lambda n, ps=prefetch_seen: ps.append(n),
                norm_weights=None,
                eps=1e-6,
                hc_mult=streams,
            )
            local_h = layout.local_slice(hidden)
            if with_previous_output:
                local_parts = RankOrderParts(
                    torch.stack(
                        (layout.local_slice(prev_a), layout.local_slice(prev_b))
                    )
                )
                local_post = layout.local_slice(post_mix)
                local_res = layout.local_slice(res_mix)
                out = engram_ops.Engram.forward(
                    engram,
                    local_h,
                    hash_ids,
                    token_ids,
                    sp_rows=layout,
                    previous_output=local_parts,
                    post_mix=local_post,
                    res_mix=local_res,
                )
            else:
                out = engram_ops.Engram.forward(
                    engram, local_h, hash_ids, token_ids, sp_rows=layout
                )
            real = layout.real_rows
            assert out.dtype == torch.bfloat16 and out.shape == local_h.shape
            assert torch.equal(
                out[:real].view(torch.int16),
                full[layout.start : layout.start + real].view(torch.int16),
            )
            if real > 0:
                assert wkv_rows_seen == [real]
                assert prefetch_seen == [real]
            else:
                assert wkv_rows_seen == []
                assert prefetch_seen == [0]
        return

    # Blocked column-parallel mode across lockstep ranks.
    fabric = _RowExchangeFabric(world)
    local_comm = threading.local()
    monkeypatch.setattr(
        sp_prefill,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(
                pynccl_comm=local_comm.comm, b12x_ar_comm=None
            )
        ),
    )

    class _FakeBlockedWKV(engram_ops._PaddedColumnParallelLinear):
        def __init__(self, rank: int):
            nn.Module.__init__(self)
            self.output_size_per_partition = block_cols
            self.logical_output_size = logical_cols
            self.padded_output_size = logical_cols
            self.weight = nn.Parameter(
                weight_full[rank * block_cols : (rank + 1) * block_cols],
                requires_grad=False,
            )
            self.direct_into_calls = 0

        def local_product(self, input_, out=None):
            prod = (input_.float() @ self.weight.float().T).to(torch.bfloat16)
            if out is not None:
                self.direct_into_calls += 1
                out.copy_(prod)
                return out
            return prod

    outputs: list[torch.Tensor | None] = [None] * world
    direct_calls: list[int] = [0] * world
    prefetch_calls: list[list[int]] = [[] for _ in range(world)]
    errors = []

    def run_blocked(rank: int) -> None:
        try:
            local_comm.comm = fabric.comm(rank)
            layout = sp_prefill.SPRows(tokens, world, rank)
            wkv = _FakeBlockedWKV(rank)
            engram = SimpleNamespace(
                embed_tokens=SimpleNamespace(table_memory="device", columns=None),
                staged_rows=staged,
                wkv=wkv,
                l2_prefetch_after_projection=lambda n, r=rank: prefetch_calls[r].append(
                    n
                ),
                norm_weights=None,
                eps=1e-6,
                hc_mult=streams,
            )
            local_h = layout.local_slice(hidden)
            if with_previous_output:
                local_parts = RankOrderParts(
                    torch.stack(
                        (layout.local_slice(prev_a), layout.local_slice(prev_b))
                    )
                )
                local_post = layout.local_slice(post_mix)
                local_res = layout.local_slice(res_mix)
                out = engram_ops.Engram.forward(
                    engram,
                    local_h,
                    hash_ids,
                    token_ids,
                    sp_rows=layout,
                    previous_output=local_parts,
                    post_mix=local_post,
                    res_mix=local_res,
                )
            else:
                out = engram_ops.Engram.forward(
                    engram, local_h, hash_ids, token_ids, sp_rows=layout
                )
            outputs[rank] = out
            direct_calls[rank] = wkv.direct_into_calls
        except BaseException as exc:
            errors.append((rank, exc))
            fabric.barrier.abort()

    threads = [threading.Thread(target=run_blocked, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors

    for rank in range(world):
        layout = sp_prefill.SPRows(tokens, world, rank)
        real = layout.real_rows
        out = outputs[rank]
        assert out is not None and out.dtype == torch.bfloat16
        assert torch.equal(
            out[:real].view(torch.int16),
            full[layout.start : layout.start + real].view(torch.int16),
        )
        assert direct_calls[rank] > 0
        assert prefetch_calls[rank] == [tokens]
        if real < layout.local_rows:
            local_h = layout.local_slice(hidden)
            if with_previous_output:
                expected_pad = torch.empty_like(local_h[real:])
                fake_mhc_post(
                    layout.local_slice(prev_sum)[real:],
                    local_h[real:],
                    layout.local_slice(post_mix)[real:],
                    layout.local_slice(res_mix)[real:],
                    expected_pad,
                )
                assert torch.equal(
                    out[real:].view(torch.int16), expected_pad.view(torch.int16)
                )
            else:
                assert torch.equal(
                    out[real:].view(torch.int16), local_h[real:].view(torch.int16)
                )


@pytest.mark.parametrize("world", [3, 4, 5])
@pytest.mark.parametrize("tokens", [9, 7, 5])
def test_interleaved_sp_rows_matches_interleaved_rows_slice(
    monkeypatch, world: int, tokens: int
) -> None:
    """Exchanging gathered disk hash columns by row block and interleaving only
    each rank's local rows matches slicing the full ``interleaved_rows`` result
    bit for bit across TP3, TP4, and TP5 (including ``24 % tp != 0`` and empty
    trailing ranks)."""
    from vllm.models.deepseek_v4_1.common import engram as engram_ops

    max_cols = (24 + world - 1) // world
    gen = torch.Generator().manual_seed(700 + 31 * world + tokens)
    rank_locals = [
        torch.randn((tokens, max_cols * 256), generator=gen, dtype=torch.bfloat16)
        for _ in range(world)
    ]

    # Reference: full interleaved_rows via dim=-1 all-gather.
    gathered_flat = torch.cat(
        [loc.view(-1, 256) for loc in rank_locals], dim=-1
    )
    full_rows = gathered_flat.view(tokens, -1)[:, : 24 * 256].contiguous()

    def fake_all_to_all_rows(tensor: torch.Tensor, local_rows: int) -> torch.Tensor:
        raise AssertionError("handled per-rank via exchange_rows")

    for rank in range(world):
        layout = sp_prefill.SPRows(tokens, world, rank)
        real = layout.real_rows

        def exchange_for_rank(
            _self, _tensor: torch.Tensor, r: int = rank, l: int = layout.local_rows
        ) -> torch.Tensor:
            out = torch.zeros(
                (world, l, max_cols * 256), dtype=torch.bfloat16
            )
            rr = sp_prefill.SPRows(tokens, world, r).real_rows
            start = r * l
            if rr > 0:
                for s in range(world):
                    out[s, :rr].copy_(rank_locals[s][start : start + rr])
            return out

        monkeypatch.setattr(sp_prefill.SPRows, "exchange_rows", exchange_for_rank)
        local_interleaved = engram_ops.interleaved_sp_rows(
            rank_locals[rank], layout
        )
        expected = full_rows[layout.start : layout.start + real]
        assert local_interleaved.shape == (real, 24 * 256)
        assert local_interleaved.dtype == torch.bfloat16
        assert torch.equal(
            local_interleaved.view(torch.int16), expected.view(torch.int16)
        )


@pytest.mark.parametrize("world", [3, 4])
@pytest.mark.parametrize("tokens", [9, 5])
@pytest.mark.parametrize("with_previous_output", [False, True])
def test_engram_forward_sp_disk_columns_local_sp_product_bit_exact(
    monkeypatch, world: int, tokens: int, with_previous_output: bool
) -> None:
    """With disk hash-column sharding and gathered ``wkv`` SP shards, ``Engram.forward``
    exchanges only the gathered hash columns (one collective per layer, zero after
    ``wkv``), projects only ``sp_rows.real_rows`` rows locally, and matches full-row
    ``Engram.forward`` bit for bit across TP3 and TP4."""
    from vllm.models.deepseek_v4_1.common import engram as engram_ops
    from vllm.models.deepseek_v4_1.tilelang import columns as tl_columns
    from vllm.models.deepseek_v4_1.tilelang import engram as tl_engram
    from vllm.models.deepseek_v4_1.tilelang import mhc as tl_mhc
    from vllm.models.deepseek_v4_1.tilelang.collectives import RankOrderParts

    gen = torch.Generator().manual_seed(
        900 + 29 * world + 7 * tokens + int(with_previous_output)
    )
    streams, dim = 4, 8
    max_cols = (24 + world - 1) // world
    block_cols = 32
    logical_cols = (streams + 1) * dim

    rank_staged = [
        torch.randn((tokens, max_cols * 256), generator=gen, dtype=torch.bfloat16)
        for _ in range(world)
    ]
    full_rows = (
        torch.cat([loc.view(-1, 256) for loc in rank_staged], dim=-1)
        .view(tokens, -1)[:, : 24 * 256]
        .contiguous()
    )

    hidden = torch.randn((tokens, streams, dim), generator=gen, dtype=torch.bfloat16)
    prev_a = torch.randn((tokens, dim), generator=gen, dtype=torch.bfloat16)
    prev_b = torch.randn((tokens, dim), generator=gen, dtype=torch.bfloat16)
    prev_sum = (prev_a.float() + prev_b.float()).to(torch.bfloat16)
    post_mix = torch.randn((tokens, streams), generator=gen, dtype=torch.float32)
    res_mix = torch.randn((tokens, streams, streams), generator=gen, dtype=torch.float32)
    hash_ids = torch.zeros((tokens, 24), dtype=torch.int64)
    token_ids = torch.arange(tokens, dtype=torch.int32)
    if tokens >= 3:
        token_ids[2] = tl_engram.IMAGE_SENTINEL

    weight_full = torch.randn(
        (world * block_cols, 24 * 256), generator=gen, dtype=torch.bfloat16
    )

    def fake_mhc_post(l_out, x_in, p_mix, r_mix, out):
        if l_out.ndim == 3:
            l_out = l_out.float().sum(dim=0).to(torch.bfloat16)
        mixed = torch.einsum("tsr,trd->tsd", r_mix, x_in.float())
        mixed = mixed + p_mix.unsqueeze(-1) * l_out.float().unsqueeze(1)
        out.copy_(mixed.to(torch.bfloat16))

    def fake_gate(state, kv, _nw, mask, out, _eps, _hc):
        s2d = state.view(state.shape[0], -1)
        gated = s2d.float() + kv[:, : s2d.shape[1]].float() * 0.125
        res = gated.to(torch.bfloat16)
        if mask is not None:
            keep = (mask == tl_engram.IMAGE_SENTINEL).unsqueeze(-1)
            res = torch.where(keep, s2d, res)
        out.view_as(s2d).copy_(res)

    def fake_gate_blocked(state, kv_blocks, _nw, mask, out, _eps, _hc):
        real = state.shape[0]
        kv_cat = kv_blocks[:, :real, :].permute(1, 0, 2).reshape(real, -1)
        fake_gate(state, kv_cat[:, : logical_cols], _nw, mask, out, _eps, _hc)

    def fake_post_gate(prev, res_in, p_mix, r_mix, kv, _nw, mask, out, _eps, _hc):
        rec = torch.empty_like(res_in)
        fake_mhc_post(prev, res_in, p_mix, r_mix, rec)
        fake_gate(rec, kv, _nw, mask, out, _eps, _hc)

    def fake_post_gate_blocked(
        prev, res_in, p_mix, r_mix, kv_blocks, _nw, mask, out, _eps, _hc
    ):
        rec = torch.empty_like(res_in)
        fake_mhc_post(prev, res_in, p_mix, r_mix, rec)
        fake_gate_blocked(rec, kv_blocks, _nw, mask, out, _eps, _hc)

    monkeypatch.setattr(tl_mhc, "_mhc_post", fake_mhc_post)
    monkeypatch.setattr(
        tl_columns, "copy_columns", lambda src, dst: dst.copy_(src)
    )
    monkeypatch.setattr(tl_engram, "engram_gate", fake_gate)
    monkeypatch.setattr(tl_engram, "engram_gate_blocked", fake_gate_blocked)
    monkeypatch.setattr(tl_engram, "engram_post_gate", fake_post_gate)
    monkeypatch.setattr(
        tl_engram, "engram_post_gate_blocked", fake_post_gate_blocked
    )

    # Full-row reference output.
    kv_full = (
        (full_rows.float() @ weight_full.float().T)
        .to(torch.bfloat16)[:, :logical_cols]
        .contiguous()
    )
    expected_full = torch.empty_like(hidden)
    if with_previous_output:
        fake_post_gate(
            prev_sum,
            hidden,
            post_mix,
            res_mix,
            kv_full,
            None,
            token_ids,
            expected_full,
            1e-6,
            streams,
        )
    else:
        fake_gate(
            hidden.flatten(1),
            kv_full,
            None,
            token_ids,
            expected_full.flatten(1),
            1e-6,
            streams,
        )

    class _FakeQuantMethod:
        def __init__(self):
            self.cast_rows: list[int] = []

        def apply_shards(self, _layer, x, shards, out):
            self.cast_rows.append(x.shape[0])
            for s, shard in enumerate(shards):
                w_s = shard[0] if isinstance(shard, tuple) else shard
                out[s, : x.shape[0]].copy_(
                    (x.float() @ w_s.float().T).to(torch.bfloat16)
                )
            return out

    class _FakeSPShardedWKV(engram_ops._PaddedColumnParallelLinear):
        def __init__(self, rank: int):
            nn.Module.__init__(self)
            self.tp_size = world
            self.tp_rank = rank
            self.output_size_per_partition = block_cols
            self.logical_output_size = logical_cols
            self.padded_output_size = world * block_cols
            self.weight = nn.Parameter(
                weight_full[rank * block_cols : (rank + 1) * block_cols].clone(),
                requires_grad=False,
            )
            self.tilelang_weight = self.weight.data
            self.tilelang_weight_sf = torch.full(
                (block_cols // 32, 48), rank + 1, dtype=torch.uint32
            )
            self.quant_method = _FakeQuantMethod()
            self.sp_shards = None

    # Verify gather_sp_shards aliases local shard and clones remote shards.
    gather_rank = SimpleNamespace(val=0)
    all_w_stack = torch.stack(
        [weight_full[r * block_cols : (r + 1) * block_cols] for r in range(world)],
        dim=0,
    )
    all_sf_stack = torch.stack(
        [
            torch.full((block_cols // 32, 48), r + 1, dtype=torch.uint32)
            for r in range(world)
        ],
        dim=0,
    )

    def fake_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 0
        if tensor.dtype == torch.uint32:
            return all_sf_stack.clone()
        return all_w_stack.clone()

    import vllm.distributed as vllm_dist

    monkeypatch.setattr(vllm_dist, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        engram_ops, "tensor_model_parallel_all_gather", fake_all_gather
    )

    exchange_calls: list[list[tuple[int, ...]]] = [[] for _ in range(world)]
    prefetch_calls: list[list[int]] = [[] for _ in range(world)]

    for rank in range(world):
        layout = sp_prefill.SPRows(tokens, world, rank)
        wkv = _FakeSPShardedWKV(rank)
        engram = SimpleNamespace(
            embed_tokens=SimpleNamespace(
                table_memory="device",
                columns=tuple(range(rank, 24, world)),
                caps=SimpleNamespace(max_tokens=64),
                open_disk_rows=lambda: None,
            ),
            staged_rows=rank_staged[rank],
            wkv=wkv,
            q_weight=torch.ones((streams, dim), dtype=torch.bfloat16),
            k_weight=torch.ones((streams, dim), dtype=torch.bfloat16),
            norm_weights=torch.empty(streams * dim, dtype=torch.float32),
            prefill_sp_min_rows=1,
            l2_prefetch_after_projection=lambda n, r=rank: prefetch_calls[r].append(n),
            eps=1e-6,
            hc_mult=streams,
        )
        engram_ops.Engram.process_weights_after_loading(engram)
        assert wkv.sp_shards is not None and len(wkv.sp_shards) == world
        # Own shard must alias in place (0 extra bytes).
        assert wkv.sp_shards[rank][0].data_ptr() == wkv.tilelang_weight.data_ptr()
        assert (
            wkv.sp_shards[rank][1].data_ptr() == wkv.tilelang_weight_sf.data_ptr()
        )

        def fake_exchange(
            _self, tensor: torch.Tensor, r: int = rank, l: int = layout.local_rows
        ) -> torch.Tensor:
            exchange_calls[r].append(tuple(tensor.shape))
            out = torch.zeros(
                (world, l, max_cols * 256), dtype=torch.bfloat16
            )
            rr = sp_prefill.SPRows(tokens, world, r).real_rows
            start = r * l
            if rr > 0:
                for s in range(world):
                    out[s, :rr].copy_(rank_staged[s][start : start + rr])
            return out

        monkeypatch.setattr(sp_prefill.SPRows, "exchange_rows", fake_exchange)

        local_h = layout.local_slice(hidden)
        if with_previous_output:
            local_parts = RankOrderParts(
                torch.stack((layout.local_slice(prev_a), layout.local_slice(prev_b)))
            )
            out = engram_ops.Engram.forward(
                engram,
                local_h,
                hash_ids,
                token_ids,
                sp_rows=layout,
                previous_output=local_parts,
                post_mix=layout.local_slice(post_mix),
                res_mix=layout.local_slice(res_mix),
            )
        else:
            out = engram_ops.Engram.forward(
                engram, local_h, hash_ids, token_ids, sp_rows=layout
            )

        real = layout.real_rows
        assert out.dtype == torch.bfloat16
        assert torch.equal(
            out[:real].view(torch.int16),
            expected_full[layout.start : layout.start + real].view(torch.int16),
        )
        # Exactly one collective (staged_rows exchange) and zero after wkv!
        assert exchange_calls[rank] == [(tokens, max_cols * 256)]
        assert prefetch_calls[rank] == [tokens]
        if real > 0:
            assert wkv.quant_method.cast_rows == [real]
        else:
            assert wkv.quant_method.cast_rows == []
        if real < layout.local_rows:
            if with_previous_output:
                expected_pad = torch.empty_like(local_h[real:])
                fake_mhc_post(
                    layout.local_slice(prev_sum)[real:],
                    local_h[real:],
                    layout.local_slice(post_mix)[real:],
                    layout.local_slice(res_mix)[real:],
                    expected_pad,
                )
                assert torch.equal(
                    out[real:].view(torch.int16), expected_pad.view(torch.int16)
                )
            else:
                assert torch.equal(
                    out[real:].view(torch.int16), local_h[real:].view(torch.int16)
                )


@pytest.mark.parametrize("world", [3, 4])
@pytest.mark.parametrize("rows", [1, 6, 65])
def test_tilelang_linear_apply_shards_single_cast_and_bit_exact(
    monkeypatch, world: int, rows: int
) -> None:
    """``TileLangFP8LinearMethod.apply_shards`` casts the local ``rows`` to MXFP8
    only once and invokes the exact same GEMM arithmetic per shard as ``apply``."""
    from vllm.models.deepseek_v4_1.tilelang import linear as tl_linear

    k, n_shard = 64, 32
    gen = torch.Generator().manual_seed(1100 + 19 * world + rows)
    x = torch.randn((rows, k), generator=gen, dtype=torch.bfloat16)
    weights = [
        torch.randn((n_shard, k), generator=gen, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        for _ in range(world)
    ]
    scales = [
        torch.randint(1, 200, (n_shard // 32, 2), generator=gen, dtype=torch.uint32)
        for _ in range(world)
    ]
    shards = tuple((weights[r], scales[r]) for r in range(world))

    cast_calls: list[int] = []

    def fake_scratch(specs, _scratch):
        return tuple(torch.zeros(shape, dtype=dtype) for shape, dtype in specs)

    def fake_per_token_cast(inp, _fmt, _blk, *, round_sf, use_packed_ue8m0, out):
        assert round_sf and use_packed_ue8m0
        cast_calls.append(inp.shape[0])
        xq, sf = out
        xq.copy_(inp.to(torch.float8_e4m3fn))
        sf.fill_(127)

    def fake_gemm(xq, w, sf_u32, w_sf, out):
        m = out.shape[0]
        prod = (xq[:m].float() @ w.float().T) * (
            (sf_u32[:m, :1].float() + w_sf[:, :1].float().T) * 1e-3
        )
        out.copy_(prod.to(torch.bfloat16))

    monkeypatch.setattr(tl_linear, "_scratch", fake_scratch)
    monkeypatch.setattr(tl_linear.pdl, "per_token_cast", fake_per_token_cast)

    method = tl_linear.TileLangFP8LinearMethod(SimpleNamespace(weight_block_size=[32, 32], scale_fmt="ue8m0"))
    decode_table = {b: fake_gemm for b in (16, 32, 64)}

    # Reference: run apply() per shard independently.
    expected = torch.empty((world, rows, n_shard), dtype=torch.bfloat16)
    for r in range(world):
        layer_r = SimpleNamespace(
            weight=weights[r],
            tilelang_weight=weights[r],
            tilelang_weight_sf=scales[r],
            tilelang_groups=1,
            tilelang_mxfp8_input=False,
            tilelang_shards=1,
            tilelang_split_rows=0,
            tilelang_decode_rows=64,
            tilelang_wide={},
            tilelang_prefill=fake_gemm,
            tilelang_decode=decode_table,
        )
        expected[r].copy_(method.apply(layer_r, x))

    cast_calls.clear()
    layer0 = SimpleNamespace(
        weight=weights[0],
        tilelang_weight=weights[0],
        tilelang_weight_sf=scales[0],
        tilelang_groups=1,
        tilelang_mxfp8_input=False,
        tilelang_shards=1,
        tilelang_split_rows=0,
        tilelang_decode_rows=64,
        tilelang_wide={},
        tilelang_prefill=fake_gemm,
        tilelang_decode=decode_table,
    )
    actual = torch.empty((world, rows + 3, n_shard), dtype=torch.bfloat16)
    method.apply_shards(layer0, x, shards, actual)

    assert cast_calls == [rows]
    assert torch.equal(
        actual[:, :rows].view(torch.int16), expected.view(torch.int16)
    )


def test_gather_many_keeps_grouped_pynccl_beside_roce(monkeypatch):
    """With ``b12x_ar_comm`` active and accepting every tensor, ``gather_many``
    still gathers them in one grouped ``pynccl_comm`` collective: the one-shot
    all-gather is not faster for these gathers."""
    layout = sp_prefill.SPRows(10, 3, 2)  # padded=12, local=4, real=2
    vals = torch.arange(4 * 1024, dtype=torch.uint8).reshape(4, 1024)
    scales = torch.arange(4 * 32, dtype=torch.uint8).reshape(4, 32)
    weights = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 6)
    ids = torch.arange(4 * 6, dtype=torch.int32).reshape(4, 6)
    calls = []

    class _Roce:
        disabled = False
        _runtime = SimpleNamespace(topology="ring4")

        def should_all_gather(self, inp, dim=0):
            return True

        def all_gather(self, inp, dim=0):
            raise AssertionError("gather_many took the one-shot all-gather")

    class _PyNccl:
        disabled = False

        def group_start(self):
            calls.append("group_start")

        def all_gather(self, full, local):
            calls.append(("all_gather", tuple(full.shape), tuple(local.shape)))
            full.copy_(local.repeat(3, *([1] * (local.ndim - 1))))

        def group_end(self):
            calls.append("group_end")

    monkeypatch.setattr(sp_prefill, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        sp_prefill,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(
                b12x_ar_comm=_Roce(), pynccl_comm=_PyNccl()
            )
        ),
    )
    out_v, out_s, out_w, out_i = layout.gather_many(
        vals, scales, weights, ids, padded_rows=(64, 64)
    )
    assert calls == [
        "group_start",
        ("all_gather", (12, 1024), (4, 1024)),
        ("all_gather", (12, 32), (4, 32)),
        ("all_gather", (12, 6), (4, 6)),
        ("all_gather", (12, 6), (4, 6)),
        "group_end",
    ]
    assert out_v.shape == (64, 1024) and out_s.shape == (64, 32)
    torch.testing.assert_close(out_v[:12], vals.repeat(3, 1), rtol=0, atol=0)
    torch.testing.assert_close(out_s[:12], scales.repeat(3, 1), rtol=0, atol=0)
    torch.testing.assert_close(out_w, weights.repeat(3, 1)[:10], rtol=0, atol=0)
    torch.testing.assert_close(out_i, ids.repeat(3, 1)[:10], rtol=0, atol=0)


def test_project_exchange_rows_roce_and_pynccl_coexistence(monkeypatch):
    """When ``b12x_ar_comm`` is present alongside ``pynccl_comm``:
    - ``_tp_pynccl_comm()`` remains available for multi-slice exchanges;
    - at ``slices == 1``, ``project_exchange_rows`` projects all rows in one call,
      runs ``then()`` and then the TP group's row exchange (``exchange_rows``, here
      ``b12x_ar_comm.all_to_all_rows``, whose ``should_all_to_all_rows`` is True);
    - at ``slices == 2`` (``T >= ENGRAM_SLICE_TOKENS``), ``project_exchange_rows`` uses
      the sliced ``copy_self=False`` exchange via ``pynccl_comm``."""
    world, total, block_cols = 3, 10, 4
    gen = torch.Generator().manual_seed(42)
    inputs = [
        torch.randn((total, block_cols), generator=gen, dtype=torch.bfloat16)
        for _ in range(world)
    ]

    # Case 1: slices == 1 is one projection call, then(), then the row exchange.
    proj_calls = [[] for _ in range(world)]
    roce_a2a_calls = [0] * world

    for rank in range(world):
        sp = sp_prefill.SPRows(num_tokens=total, world_size=world, rank=rank)
        local = sp.local_rows
        real = sp.real_rows

        class _RoceComm:
            disabled = False

            def should_all_to_all_rows(self, x, l_rows):
                return l_rows == local and x.shape == (total, block_cols)

            def all_to_all_rows(self, x, l_rows):
                roce_a2a_calls[rank] += 1
                out = torch.zeros((world, l_rows, block_cols), dtype=x.dtype)
                for s in range(world):
                    out[s, :real].copy_(inputs[s][rank * l_rows : rank * l_rows + real])
                return out

        roce = _RoceComm()
        steps = []

        def all_to_all_rows(x, n, rc=roce, steps=steps):  # the TP group's exchange
            steps.append("exchange")
            assert rc.should_all_to_all_rows(x, n)
            return rc.all_to_all_rows(x, n)

        monkeypatch.setattr(
            sp_prefill,
            "get_tp_group",
            lambda rc=roce: SimpleNamespace(
                device_communicator=SimpleNamespace(
                    b12x_ar_comm=rc, pynccl_comm=SimpleNamespace(disabled=False)
                )
            ),
        )
        monkeypatch.setattr(
            sp_prefill, "tensor_model_parallel_all_to_all_rows", all_to_all_rows
        )
        assert sp_prefill._tp_pynccl_comm() is not None
        assert sp_prefill._tp_roce_comm() is not None
        got = sp.project_exchange_rows(
            lambda start, stop, dst, r=rank: (
                proj_calls[r].append((start, stop)),
                dst.copy_(inputs[r][start:stop]),
            )[-1],
            (block_cols,),
            torch.bfloat16,
            torch.device("cpu"),
            slices=1,
            then=lambda steps=steps: steps.append("then"),
        )
        assert proj_calls[rank] == [(0, total)]
        assert roce_a2a_calls[rank] == 1
        assert steps == ["then", "exchange"]
        for s in range(world):
            assert torch.equal(
                got[s, :real].view(torch.int16),
                inputs[s][rank * local : rank * local + real].view(torch.int16),
            )

    # Case 2: slices == 2 with active b12x_ar_comm still routes sliced copy_self=False via pynccl_comm.
    fabric = _RowExchangeFabric(world)
    local_comm = threading.local()

    class _RoceNoA2A:
        disabled = False

        def should_all_to_all_rows(self, x, l_rows):
            return False

        def all_to_all_rows(self, x, l_rows):
            raise AssertionError("should not call roce all_to_all_rows when False")

    monkeypatch.setattr(
        sp_prefill,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(
                pynccl_comm=local_comm.comm, b12x_ar_comm=_RoceNoA2A()
            )
        ),
    )
    outputs: list[torch.Tensor | None] = [None] * world
    errors = []

    def run(rank: int) -> None:
        try:
            local_comm.comm = fabric.comm(rank)
            sp = sp_prefill.SPRows(num_tokens=total, world_size=world, rank=rank)
            outputs[rank] = sp.project_exchange_rows(
                lambda start, stop, dst: dst.copy_(inputs[rank][start:stop]),
                (block_cols,),
                torch.bfloat16,
                torch.device("cpu"),
                slices=2,
            )
        except BaseException as exc:
            errors.append((rank, exc))
            fabric.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for me in range(world):
        sp = sp_prefill.SPRows(num_tokens=total, world_size=world, rank=me)
        for s in range(world):
            assert torch.equal(
                outputs[me][s, : sp.real_rows].view(torch.int16),
                inputs[s][sp.start : sp.start + sp.real_rows].view(torch.int16),
            )


def test_exchange_into_passes_send_and_recv_to_should_exchange(monkeypatch):
    sp = sp_prefill.SPRows(num_tokens=8, world_size=4, rank=1)
    send = torch.ones((4, 2, 4), dtype=torch.bfloat16)
    recv = torch.zeros((4, 2, 4), dtype=torch.bfloat16)
    seen = []

    class _RoceEx:
        disabled = False

        def should_exchange(self, inp, out):
            seen.append(("should_exchange", inp is send, out is recv))
            return True

        def exchange_into(self, inp, out):
            seen.append(("exchange_into", inp is send, out is recv))
            out.copy_(inp)

    monkeypatch.setattr(
        sp_prefill,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(
                pynccl_comm=None, b12x_ar_comm=_RoceEx()
            )
        ),
    )
    sp.exchange_into(send, recv)
    assert seen == [
        ("should_exchange", True, True),
        ("exchange_into", True, True),
    ]
    assert torch.equal(recv, send)


def test_gather_many_async_and_gather_mxfp8_async(monkeypatch):
    from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows

    layout = sp_prefill.SPRows(10, 3, 2)  # padded=12, local=4, real=2
    a_cpu = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    b_cpu = torch.arange(12, dtype=torch.int32).reshape(4, 3)

    assert not layout.can_gather_async(torch.device("cpu"))
    monkeypatch.setattr(sp_prefill, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        sp_prefill,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: x.repeat(3, *([1] * (x.ndim - 1))),
    )
    monkeypatch.setattr(
        sp_prefill,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=SimpleNamespace(pynccl_comm=None)),
    )

    ref_a, ref_b = layout.gather_many(a_cpu, b_cpu)
    (async_a, async_b), wait_cpu = layout.gather_many_async(a_cpu, b_cpu)
    wait_cpu()
    wait_cpu()
    assert torch.equal(async_a.view(torch.int16), ref_a.view(torch.int16))
    assert torch.equal(async_b, ref_b)

    x_q_cpu = MXFP8Rows(
        rows=torch.empty((4, 64), dtype=torch.bfloat16),
        values=torch.arange(64 * 64, dtype=torch.uint8).reshape(64, 64),
        scales=torch.arange(64 * 2, dtype=torch.uint8).reshape(64, 2),
    )
    rec_cpu = torch.arange(4 * 8, dtype=torch.uint8).reshape(4, 8)
    ref_q, ref_rec = layout.gather_mxfp8(x_q_cpu, rec_cpu)
    (async_q, async_rec), wait_mxfp8_cpu = layout.gather_mxfp8_async(x_q_cpu, rec_cpu)
    wait_mxfp8_cpu()
    assert torch.equal(async_q.values[:12], ref_q.values[:12])
    assert torch.equal(async_q.scales[:12], ref_q.scales[:12])
    assert torch.equal(async_rec, ref_rec)

    if not torch.cuda.is_available():
        return

    device = torch.device("cuda", torch.cuda.current_device())
    events = []
    recorded_streams = []

    class _FakeSideStream(torch.cuda.Stream):
        def wait_stream(self, stream):
            events.append(("side_wait_stream", stream))
            return super().wait_stream(stream)

    side = _FakeSideStream(device=device)
    main = torch.cuda.current_stream(device)
    orig_main_wait = main.wait_stream
    monkeypatch.setattr(
        main,
        "wait_stream",
        lambda stream: (
            events.append(("main_wait_stream", stream is side)),
            orig_main_wait(stream),
        )[-1],
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda dev=None: main)
    orig_record_stream = torch.Tensor.record_stream

    def spy_record_stream(tensor, stream):
        recorded_streams.append(stream)
        return orig_record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", spy_record_stream)
    monkeypatch.setattr(sp_prefill, "_side_stream", lambda dev: side)

    class _Comm:
        disabled = False

        def group_start(self):
            events.append("group_start")

        def all_gather(self, full, local, stream=None):
            events.append(("all_gather", tuple(full.shape), tuple(local.shape), stream is side))
            with torch.cuda.stream(stream or main):
                full.copy_(local.repeat(3, *([1] * (local.ndim - 1))))

        def group_end(self):
            events.append("group_end")

    comm = _Comm()
    monkeypatch.setattr(
        sp_prefill,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=SimpleNamespace(pynccl_comm=comm)),
    )
    assert layout.can_gather_async(device) is True
    assert not layout.can_gather_async(torch.device("cpu"))

    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)
    ref_ag, ref_bg = layout.gather_many(a_gpu, b_gpu)
    events.clear()
    recorded_streams.clear()

    (out_ag, out_bg), wait_many = layout.gather_many_async(a_gpu, b_gpu)
    assert events == [
        ("side_wait_stream", main),
        "group_start",
        ("all_gather", (12, 2), (4, 2), True),
        ("all_gather", (12, 3), (4, 3), True),
        "group_end",
    ]
    assert recorded_streams == [side, side, side, side]
    wait_many()
    wait_many()
    assert events[-1] == ("main_wait_stream", True)
    assert sum(1 for e in events if e == ("main_wait_stream", True)) == 1
    assert torch.equal(out_ag.view(torch.int16), ref_ag.view(torch.int16))
    assert torch.equal(out_bg, ref_bg)

    x_q_gpu = MXFP8Rows(
        rows=x_q_cpu.rows.to(device),
        values=x_q_cpu.values.to(device),
        scales=x_q_cpu.scales.to(device),
    )
    rec_gpu = rec_cpu.to(device)
    ref_qg, ref_recg = layout.gather_mxfp8(x_q_gpu, rec_gpu)
    events.clear()
    recorded_streams.clear()

    (out_qg, out_recg), wait_mxfp8 = layout.gather_mxfp8_async(x_q_gpu, rec_gpu)
    assert events == [
        ("side_wait_stream", main),
        "group_start",
        ("all_gather", (12, 64), (4, 64), True),
        ("all_gather", (12, 2), (4, 2), True),
        ("all_gather", (12, 8), (4, 8), True),
        "group_end",
    ]
    assert recorded_streams == [side] * 6
    wait_mxfp8()
    assert events[-1] == ("main_wait_stream", True)
    assert torch.equal(out_qg.values[:12], ref_qg.values[:12])
    assert torch.equal(out_qg.scales[:12], ref_qg.scales[:12])
    assert torch.equal(out_recg, ref_recg)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not layout.can_gather_async(device)


def test_sp_replicated_async_overlaps_compressor_indexer_and_split_wqb(monkeypatch):
    from vllm.models.deepseek_v4_1 import attention, kernels
    from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
    from vllm.models.deepseek_v4_1.tilelang import cache_writer, front
    from vllm.models.deepseek_v4_1.tilelang.attention import DeepseekV41TileLangAttention
    from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows, mxfp8_rows

    monkeypatch.setattr(attention, "_INDEX_SPLIT_CHECK", False)
    monkeypatch.setattr(attention, "is_breakable_cudagraph_enabled", lambda: False)

    def make_module():
        module = DeepseekV41TileLangAttention.__new__(DeepseekV41TileLangAttention)
        nn.Module.__init__(module)
        module.prefix = "model.layers.5.attn"
        module.layer_id = 5
        module.candidate_source_layer = 3
        module.capacity = 300
        module.swa_width = 128
        module.compress_ratio = 2
        module._main_page = 32
        module._index_page = 64
        module._main_width = 16
        module._index_width = 16
        module.is_ced_decoder = False
        module.is_index_source = True
        module.topk_indices_buffer = None
        module.q_lora_rank = 32
        module.n_local_heads = 4
        module.kv_lora_rank = 512
        module.qk_rope_head_dim = 64
        module._l2pf_wo = None
        module._l2pf_comp = None
        module._ready = True
        module.config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            speculative_config=None,
            compilation_config=SimpleNamespace(max_cudagraph_capture_size=24),
        )
        module.swa_cache_layer = SimpleNamespace(
            prefix="swa",
            kv_cache=torch.zeros((8, 64 * 528), dtype=torch.uint8),
            block_size=64,
        )
        module.kv_cache = torch.zeros((4, module._main_page * 288), dtype=torch.uint8)
        module.q_norm = SimpleNamespace(
            weight=torch.ones(32, dtype=torch.bfloat16), variance_epsilon=1e-6
        )
        module.kv_norm = SimpleNamespace(
            weight=torch.ones(512, dtype=torch.bfloat16), variance_epsilon=1e-6
        )
        module.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.ones((256, 64), dtype=torch.float32)
        )
        module._owner = lambda: SimpleNamespace(prefix=module.prefix)
        return module

    orig_write_compressed = attention.DeepseekV4Attention._write_compressed_index_keys

    def run_case(*, async_enabled: bool):
        module = make_module()
        events = []

        num_tokens = 200
        local_rows = 50
        start = 50
        real = 50
        tail = 100

        full_hidden = (
            torch.arange(num_tokens * 32, dtype=torch.bfloat16).reshape(num_tokens, 32)
            * 0.01
        )
        local_x = full_hidden[start : start + local_rows].clone()
        positions = torch.arange(num_tokens, dtype=torch.int64)
        slots = torch.arange(1000, 1000 + num_tokens, dtype=torch.int64)

        full_q_vals = torch.empty((mxfp8_rows(num_tokens), 32), dtype=torch.float8_e4m3fn)
        full_q_vals[:num_tokens].copy_(
            (torch.arange(num_tokens, dtype=torch.int64)[:, None] % 7 + 1)
            .expand(num_tokens, 32)
            .to(torch.float8_e4m3fn)
        )
        full_q_scales = torch.full((mxfp8_rows(num_tokens), 1), 127, dtype=torch.uint8)
        full_records = (
            torch.arange(num_tokens, dtype=torch.uint8)[:, None]
            .expand(num_tokens, 528)
            .contiguous()
        )
        full_comp_vals = full_hidden[:, :4].to(torch.float32) + 1.0
        full_comp_gates = full_hidden[:, 4:8].to(torch.float32) + 2.0
        full_weights = torch.arange(num_tokens * 4, dtype=torch.bfloat16).reshape(
            num_tokens, 4
        ) * 0.05

        class _FakeSPRows:
            def __init__(self):
                self.num_tokens = num_tokens
                self.world_size = 4
                self.rank = 1
                self.local_rows = local_rows
                self.start = start
                self.real_rows = real

            def can_gather_async(self, device=None):
                return async_enabled

            def gather_mxfp8(self, x_q, *extra):
                events.append(("gather_mxfp8", tuple(x_q.rows.shape), tuple(extra[0].shape)))
                rows_q = mxfp8_rows(num_tokens)
                gq = MXFP8Rows(
                    rows=torch.empty((num_tokens, 32), dtype=torch.bfloat16),
                    values=full_q_vals[:rows_q].clone(),
                    scales=full_q_scales[:rows_q].clone(),
                )
                return (gq, full_records.clone())

            def gather_mxfp8_async(self, x_q, *extra):
                events.append(
                    ("gather_mxfp8_async", tuple(x_q.rows.shape), tuple(extra[0].shape))
                )
                rows_q = mxfp8_rows(num_tokens)
                gq = MXFP8Rows(
                    rows=torch.empty((num_tokens, 32), dtype=torch.bfloat16),
                    values=full_q_vals[:rows_q].clone(),
                    scales=full_q_scales[:rows_q].clone(),
                )

                def wait_front():
                    events.append("wait_front")

                return (gq, full_records.clone()), wait_front

            def gather(self, x):
                events.append(("gather", tuple(x.shape)))
                if x.shape == full_comp_vals[start : start + real].shape:
                    return full_comp_vals.clone()
                return full_comp_gates.clone()

            def gather_many_async(self, *tensors):
                events.append(("gather_many_async", len(tensors)))

                def wait_comp():
                    events.append("wait_comp")

                return (full_comp_vals.clone(), full_comp_gates.clone()), wait_comp

        comp = DeepseekCompressor.__new__(DeepseekCompressor)
        nn.Module.__init__(comp)
        comp.compress_ratio = 2
        comp._sp_kernel = object()
        comp.state_cache = SimpleNamespace(prefix="comp.state_cache")
        comp.can_project_sp_async = lambda sr: True

        def project_sp_local(hidden, sr):
            events.append(("compressor.project_sp_local", tuple(hidden.shape)))
            assert torch.equal(hidden, local_x)
            return (
                full_comp_vals[start : start + local_rows].clone(),
                full_comp_gates[start : start + local_rows].clone(),
            )

        def finish_sp_forward(sr, main, state, *, values, gates=None, **kw):
            events.append(("compressor.finish_sp_forward", sr.num_tokens))
            latent = (values + gates).to(torch.bfloat16)
            comp_slots = torch.arange(sr.num_tokens, dtype=torch.int64)
            return latent, comp_slots

        def comp_forward(hidden, main, state, *, sp_rows=None, **kw):
            vals = sp_rows.gather(full_comp_vals[start : start + local_rows])
            gates = sp_rows.gather(full_comp_gates[start : start + local_rows])
            return finish_sp_forward(sp_rows, main, state, values=vals, gates=gates, **kw)

        comp.project_sp_local = project_sp_local
        comp.finish_sp_forward = finish_sp_forward
        comp.forward = comp_forward
        module.compressor = comp

        def rep_wqa_wkv(h):
            events.append(("fused_wqa_wkv", tuple(h.shape)))
            base = h[:, :1].to(torch.float32) + torch.arange(
                32 + 512, dtype=torch.float32
            )
            return base.to(torch.bfloat16), None

        rep_wqa_wkv.tp_size = 4
        rep_wqa_wkv.replicated_sp_prefill = True
        module.fused_wqa_wkv = rep_wqa_wkv

        def fake_sp_front(
            qa_kv, q_w, kv_w, eps, pos, cos_sin, q_values, q_scales, records
        ):
            events.append(("sp_attention_front", tuple(qa_kv.shape)))
            q_values[:real].copy_(full_q_vals[start : start + real])
            q_scales[:real].copy_(full_q_scales[start : start + real])
            records[:real].copy_(full_records[start : start + real])

        def fake_write_swa_records(records, cache, slot_map, *, page_size):
            events.append(("write_swa_records", tuple(records.shape)))
            assert torch.equal(records, full_records)

        orig_slice_mxfp8 = kernels.slice_mxfp8
        monkeypatch.setattr(front, "sp_attention_front", fake_sp_front)
        monkeypatch.setattr(cache_writer, "write_swa_records", fake_write_swa_records)
        monkeypatch.setattr(kernels, "slice_mxfp8", orig_slice_mxfp8)

        def fake_wq_b(qr):
            provided = kernels.try_provided_mxfp8(qr)
            assert provided is not None
            r = qr.shape[0]
            mode = "full" if r == num_tokens else ("local" if r == real and len(events) < 8 else "peer")
            # Determine exact role via values vs local/head/tail slices for assertions
            vals_bf16 = provided.values[:r].to(torch.bfloat16)
            events.append(("wq_b", r))
            out = (
                vals_bf16.repeat(1, (module.n_local_heads * 512) // 32)
                + torch.arange(module.n_local_heads * 512, dtype=torch.bfloat16) * 0.001
            )
            return out

        fake_wq_b.tilelang_prefill = True
        fake_wq_b.tilelang_decode_rows = 16
        fake_wq_b.tilelang_shards = 1
        fake_wq_b.tilelang_split_rows = 0
        module.wq_b = fake_wq_b

        module._rope = lambda x, pos, role: x + pos[:, None, None].to(x.dtype) * 0.01

        def fake_idx_wq_b(qr):
            events.append(("indexer.wq_b", qr.shape[0]))
            provided = kernels.try_provided_mxfp8(qr)
            assert provided is not None
            r = qr.shape[0]
            return provided.values[:r].to(torch.bfloat16).repeat(1, (4 * 128) // 32)

        fake_idx_wq_b.tilelang_prefill = True
        fake_idx_wq_b.tilelang_decode_rows = 16
        module.indexer = SimpleNamespace(
            heads=4,
            wq_b=fake_idx_wq_b,
            weights_proj=lambda h: (
                events.append(("indexer.weights_proj", tuple(h.shape))),
                full_weights[start : start + h.shape[0]].clone(),
            )[1],
            k_cache=SimpleNamespace(prefix="idx.k_cache"),
            wk=lambda lat: lat,
        )

        def fake_quant_iq(mode, iq, data, scale, positions=None):
            data.copy_(iq.view(iq.shape[0], 4, 128)[:, :, :64].to(torch.uint8))
            scale.copy_(iq.view(iq.shape[0], 4, 128)[:, :, :4].to(torch.uint8))

        module._quantize_index_query = fake_quant_iq

        orig_index_query = DeepseekV41TileLangAttention._index_query

        def tracked_index_query(
            qr, pos, hidden, weights, first, last, swa, q_cast=None
        ):
            tag = "local" if (first == 0 and last == real and qr.shape[0] == real) else "full"
            events.append(("_index_query", tag, first, last))
            return orig_index_query(
                module, qr, pos, hidden, weights, first, last, swa, q_cast
            )

        module._index_query = tracked_index_query

        def tracked_write_compressed(self_mod, pos, r, meta, lat, comp_slots, fuse):
            events.append(("_write_compressed_index_keys", r, fuse))
            return orig_write_compressed(
                self_mod, pos, r, meta, lat, comp_slots, fuse
            )

        monkeypatch.setattr(
            attention.DeepseekV4Attention,
            "_write_compressed_index_keys",
            tracked_write_compressed,
        )
        module._write_index_keys = lambda key, slot_map, positions=None: events.append(
            ("write_index_keys", tuple(key.shape))
        )

        def fake_forward_mqa(q, kv, pos, output, *, index_query=None):
            events.append(("forward_mqa", tuple(q.shape)))
            iq_data, iq_scale, weights, *_ = index_query
            w_sum = weights.to(torch.float32).sum(dim=-1, keepdim=True)
            iq_sum = iq_data[:real].to(torch.float32).mean(dim=(1, 2), keepdim=True)
            scale_sum = iq_scale[:real].to(torch.float32).mean(dim=(1, 2), keepdim=True)
            output.copy_(q)
            output[start:start + real, :, :1] += (
                w_sum[:, None, :] + iq_sum + scale_sum
            ).to(q.dtype)

        module.forward_mqa = fake_forward_mqa
        module._o_proj = lambda output, sp_rows=None: (
            events.append(("_o_proj", tuple(output.shape))),
            output.reshape(num_tokens, -1).sum(dim=-1, keepdim=True).repeat(1, 32),
        )[1]

        metadata = {
            "swa": SimpleNamespace(is_decode=False, slot_mapping=slots),
            module.prefix: "main_meta",
            "comp.state_cache": "state_meta",
            "idx.k_cache": SimpleNamespace(slot_mapping=slots),
        }
        monkeypatch.setattr(
            attention,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=metadata),
        )
        monkeypatch.setattr(sp_prefill, "current", lambda: _FakeSPRows())
        return module._forward(positions, local_x), events

    async_out, async_events = run_case(async_enabled=True)
    sync_out, sync_events = run_case(async_enabled=False)

    assert torch.equal(async_out.view(torch.int16), sync_out.view(torch.int16))
    assert async_events == [
        ("fused_wqa_wkv", (50, 32)),
        ("indexer.weights_proj", (50, 32)),
        ("sp_attention_front", (50, 544)),
        ("gather_mxfp8_async", (50, 32), (50, 528)),
        ("compressor.project_sp_local", (50, 32)),
        ("_index_query", "local", 0, 50),
        ("indexer.wq_b", 50),
        ("wq_b", 50),
        "wait_front",
        ("gather_many_async", 2),
        ("write_swa_records", (200, 528)),
        ("wq_b", 50),
        ("wq_b", 100),
        "wait_comp",
        ("compressor.finish_sp_forward", 200),
        ("_write_compressed_index_keys", 200, True),
        ("write_index_keys", (200, 4)),
        ("forward_mqa", (200, 4, 512)),
        ("_o_proj", (200, 4, 512)),
    ]
    assert ("gather_mxfp8", (50, 32), (50, 528)) in sync_events
