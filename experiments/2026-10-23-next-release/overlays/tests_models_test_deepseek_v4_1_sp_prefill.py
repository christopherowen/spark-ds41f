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


def test_engram_gates_only_the_real_local_rows(monkeypatch):
    pytest.importorskip("b12x")
    from vllm.models.deepseek_v4_1.common import engram as engram_ops

    tokens, streams, width = 10, 2, 4

    def mix(state, kv, norm_weights, *, eps, plan, out, token_mask=None):
        assert kv.shape[0] == state.shape[0] == out.shape[0]
        gate = kv[:, : state.shape[1]]
        if token_mask is not None:
            assert token_mask.shape == (state.shape[0],)
            gate = gate * token_mask[:, None]
        return out.copy_(state + gate)

    monkeypatch.setattr(engram_ops.hyperconnection, "run_engram_mix", mix)
    monkeypatch.setattr(engram_ops, "tensor_model_parallel_all_reduce", lambda x: x)
    staged = torch.randn(tokens, 6)
    engram = SimpleNamespace(
        embed_tokens=SimpleNamespace(table_memory="device", columns=None),
        overlap_epochs=None,
        staged_rows=staged,
        wkv=lambda rows: rows.repeat(1, 2),
        norm_weights=None,
        eps=1e-6,
        mix_plans={False: None, True: None},
    )
    hidden = torch.randn(tokens, streams, width)
    hash_ids = torch.zeros(tokens, 24, dtype=torch.int64)
    mask = torch.arange(tokens) % 3 != 1
    full = engram_ops.Engram.forward(engram, hidden, hash_ids, mask)
    for rank in range(3):
        layout = sp_prefill.SPRows(tokens, 3, rank)
        local = layout.local_slice(hidden)
        output = engram_ops.Engram.forward(
            engram, local, hash_ids, mask, sp_rows=layout
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
        )

    def post_pre(self, x, residual, post, comb, fn, scale, base, norm, pre):
        return self.pre(self.post(x, residual, post, comb), fn, scale, base, norm, pre)


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

    def forward(self, positions, hidden, scaling, *, global_kv_ready=None):
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
@pytest.mark.parametrize("skip_mtp_buffer", (False, True))
def test_model_forward_matches_full_rows(
    monkeypatch, num_tokens, use_embeds, skip_mtp_buffer
):
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
        buffer = torch.full((16, 8), -7.0, dtype=torch.float64)
        target = SimpleNamespace(
            embed_input_ids=lambda input_ids: table[input_ids],
            embed_tokens=_Embedding(table, world),
            use_mega_moe=False,
            use_sequence_parallel=False,
            prefill_sp_min_rows=min_rows,
            prefill_sp_unreduced=False,
            prefill_sp_skips_mtp_buffer=skip_mtp_buffer,
            disk_engram=False,
            engram_hash=None,
            start_layer=0,
            end_layer=3,
            ced_decoder_start=None,
            layers=nn.ModuleList(_Layer(i, world, local_rows) for i in range(3)),
            aux_hidden_state_layers={2, 3},
            _mtp_hidden_buffer=buffer,
            # L2 prefetch is on by default on SM121; its plans are built.
            _l2pf_ready=True,
            # The sink detector's column buffer: none, the detector is off here.
            _sink_columns=None,
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
        return output, aux, buffer

    expected, expected_aux, expected_buffer = run(0, [])
    local_rows = [[] for _ in range(3)]
    results = world.run(lambda rank: run(4, local_rows[rank]))
    layout = sp_prefill.SPRows(num_tokens, 3, 0)
    for rank, (output, aux, buffer) in enumerate(results):
        # Attention received only this rank's rows in every layer.
        assert local_rows[rank] == [layout.local_rows] * 3
        torch.testing.assert_close(output, expected, rtol=1e-9, atol=1e-9)
        assert len(aux) == len(expected_aux) == 2
        for actual, value in zip(aux, expected_aux):
            torch.testing.assert_close(actual, value, rtol=1e-9, atol=1e-9)
        if skip_mtp_buffer:
            assert (buffer == -7).all()
        else:
            torch.testing.assert_close(buffer, expected_buffer, rtol=1e-9, atol=1e-9)
