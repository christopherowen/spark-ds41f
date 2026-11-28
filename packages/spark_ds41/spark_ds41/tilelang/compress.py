# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang CSA compression for DeepSeek-V4.1: pair pooling, BF16 rounding, RMSNorm.

DeepSeek's reference (``inference/model.py``, the compressor) emits one 512-wide
latent per compressed position. At ratio 1 every token emits ``norm(value)``; at
ratio 2 only odd positions emit, pooling the pair (the even predecessor and the token)
with a per-channel softmax of their gates::

    pooled = v0 * e0 / (e0 + e1) + v1 * e1 / (e0 + e1),  e_i = exp(g_i - max(g0, g1))
    x      = bf16(pooled)                     (rounded BEFORE the RMS, as the reference)
    out    = bf16(x * rsqrt(mean(x^2) + 1e-20) * weight)

A ratio-2 token's predecessor is the previous token of its request in this step, or,
for a request's first token, the even position the state ring holds from an earlier
step (``gather_partial_state``, which must finish before ``compress`` overwrites ring
slots). ``compress`` also saves each request's latest ``page`` positions of the step
(value and gate) to their ring slots, so a slot holds its latest position.

This replaced B12X's ``mla_compress`` (one 128-thread block per token of the whole
batch capacity every call, then a pending-ring commit vLLM never reads) and vLLM's two
Triton ring kernels (the load, and the save over the whole capacity): two launches
sized by the step. One warp owns a token, 16
values a lane, and reduces its sum of squares with warp shuffles. ``exp`` and
``rsqrt`` are the FP32 library functions, as the reference computes them (B12X used
fast approximations).
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import tilelang
import tilelang.language as T
import torch

from . import pdl
from .cache_writer import _CVT_HEADER, _E4M3_MAX, RECORD_BYTES

DIM = 512
STATE = 2 * DIM  # a ring slot: the value, then the gate
EPS = 1e-20
_VEC = 4  # FP32 per 16-byte access
_CHUNKS = DIM // (32 * _VEC)  # 16-byte accesses per lane
_PASS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
_INDEXED_RECORD = RECORD_BYTES["indexed"]
_INDEXED_PAYLOAD = DIM // 2


@tilelang.jit(pass_configs=_PASS)
def gather_partial_state_kernel(max_requests: int, page: int):
    """Each live request's ring carry: the even predecessor of an odd first position."""
    blocks = T.dynamic("blocks")
    width = T.dynamic("width")
    pool_stride = T.dynamic("pool_stride")  # the allocator pads its pages
    table_rows = T.dynamic("table_rows")
    table_cols = T.dynamic("table_cols")
    table_stride = T.dynamic("table_stride")

    @T.prim_func
    def compress_gather_partial_state(
        pool: T.StridedTensor((blocks, width), (pool_stride, 1), T.float32),
        table: T.StridedTensor((table_rows, table_cols), (table_stride, 1), T.int32),
        starts: T.Tensor((max_requests + 1,), T.int32),
        positions: T.Tensor((max_requests,), T.int64),
        counts: T.Tensor((2,), T.int32),
        pending_values: T.Tensor((max_requests, DIM), T.float32),
        pending_gates: T.Tensor((max_requests, DIM), T.float32),
        tags: T.Tensor((max_requests,), T.int64),
        local_ids: T.Tensor((max_requests,), T.int64),
    ):
        with T.Kernel(max_requests, threads=128) as r:
            tx = T.get_thread_binding()
            buf = T.alloc_local((_VEC,), T.float32)
            nt = counts[0]
            first = starts[r]
            end = starts[r + 1]
            pos = positions[r]
            live = (
                (r < counts[1])
                & (first >= 0)
                & (first < end)
                & (end <= nt)
                & (pos >= 0)
            )
            needed = live & (pos > 0) & (pos % 2 == 1)
            page_id = T.if_then_else(needed, T.Cast(T.int64, table[r, 0]), T.int64(0))
            valid = needed & (page_id > 0)
            base = ((pos - 1) % page) * STATE
            pdl.chain()
            for half in T.serial(2):
                for i in T.vectorized(_VEC):
                    buf[i] = T.if_then_else(
                        valid, pool[page_id, base + half * DIM + tx * _VEC + i], 0.0
                    )
                for i in T.vectorized(_VEC):
                    if half == 0:
                        pending_values[r, tx * _VEC + i] = buf[i]
                    else:
                        pending_gates[r, tx * _VEC + i] = buf[i]
            if tx == 0:
                tags[r] = T.if_then_else(valid, pos - 1, T.int64(-1))
                local_ids[r] = T.if_then_else(live, T.Cast(T.int64, r), T.int64(-1))

    return compress_gather_partial_state


@tilelang.jit(pass_configs=_PASS)
def compress_kernel(
    ratio: int,
    max_requests: int,
    page: int,
    warps: int = 4,
    write_kv: bool = False,
    kv_page_size: int = 64,
    rope_dim: int = 64,
):
    """Rows ``[0, rows)`` of the step: latents, emitted flags and slots (zero, false
    and -1 past the live tokens), at ratio 2 the live tokens' ring saves, and with
    ``write_kv`` the rotated 288-byte ``indexed`` KV-cache records in the same launch."""
    assert ratio in (1, 2)
    rows = T.dynamic("rows")
    gate_rows = T.dynamic("gate_rows")  # rows at ratio 2; any buffer at ratio 1
    blocks = T.dynamic("blocks")
    width = T.dynamic("width")
    pool_stride = T.dynamic("pool_stride")  # the allocator pads its pages
    positions_max = T.dynamic("positions_max")
    kv_pages = T.dynamic("kv_pages")
    kv_page_bytes = T.dynamic("kv_page_bytes")
    kv_page_stride = T.dynamic("kv_page_stride")
    kv_page_words = T.dynamic("kv_page_words")
    kv_word_stride = T.dynamic("kv_word_stride")
    value_dtype = T.float32 if ratio == 2 else T.bfloat16
    rope_start = DIM - rope_dim
    rope_half = rope_dim // 2

    @T.macro
    def body(
        values,
        gates,
        weight,
        starts,
        positions,
        state_ids,
        slots,
        state_slots,
        counts,
        pending_values,
        pending_gates,
        tags,
        pool,
        out,
        emitted,
        emitted_slots,
        token_positions,
        cos_sin,
        kv_cache,
        kv_words,
    ):
        with T.Kernel(T.ceildiv(rows, warps), threads=warps * 32) as bx:
            if write_kv:
                T.import_source(_CVT_HEADER)
            tx = T.get_thread_binding()
            token = bx * warps + tx // 32
            lane = tx % 32
            x = T.alloc_local((_CHUNKS * _VEC,), T.float32)
            v1 = T.alloc_local((_VEC,), value_dtype)
            g1 = T.alloc_local((_VEC,), T.float32)
            v0 = T.alloc_local((_VEC,), T.float32)
            g0 = T.alloc_local((_VEC,), T.float32)
            w = T.alloc_local((_CHUNKS * _VEC,), T.float32)
            y = T.alloc_local((_VEC,), T.bfloat16)
            rv = T.alloc_local((_VEC,), T.float32)
            # request, first, position, valid, pending carry, request end, nt
            info = T.alloc_local((7,), T.int64)
            slot_info = T.alloc_local((3,), T.int64)
            ss = T.alloc_local((1,), T.float32)
            amax = T.alloc_var(T.float32)
            for c in T.unroll(_CHUNKS):
                col = (c * 32 + lane) * _VEC
                for i in T.vectorized(_VEC):
                    w[c * _VEC + i] = weight[col + i]
            if token < rows:
                slot_info[0] = slots[token]
                if ratio == 2:
                    slot_info[1] = state_slots[token]
                if write_kv:
                    slot_info[2] = token_positions[token]
                nt = counts[0]
                nr = counts[1]
                info[0] = T.int64(-1)
                info[1] = T.int64(0)
                info[2] = T.int64(-1)
                info[3] = T.int64(0)
                info[4] = T.int64(0)
                info[5] = T.int64(-1)
                info[6] = T.Cast(T.int64, nt)
                if (nt >= 0) & (nr > 0) & (nr <= max_requests) & (token < nt):
                    # The request whose rows contain the token: the first whose end
                    # passes it (empty requests never match).
                    for q in T.serial(max_requests):
                        if (info[0] < 0) & (q < nr) & (starts[q + 1] > token):
                            info[0] = T.Cast(T.int64, q)
                    if info[0] >= 0:
                        q = info[0]
                        first = T.Cast(T.int64, starts[q])
                        end = T.Cast(T.int64, starts[q + 1])
                        start_pos = positions[q]
                        info[5] = end
                        if (
                            (first >= 0)
                            & (first <= token)
                            & (end <= nt)
                            & (token < end)
                            & (start_pos >= 0)
                        ):
                            info[1] = first
                            info[2] = start_pos + (token - first)
                            info[3] = T.int64(1)
            pdl.chain()
            if token < rows:
                nt = T.Cast(T.int32, info[6])
                if info[3] == 1:
                    q = info[0]
                    sid = state_ids[q]
                    if (sid >= 0) & (sid < max_requests):
                        if ratio == 2:
                            if info[2] % 2 == 1:
                                if token == info[1]:
                                    info[4] = T.int64(1)
                                    if tags[sid] != info[2] - 1:
                                        info[3] = T.int64(0)
                            else:
                                info[3] = T.int64(0)
                    else:
                        info[3] = T.int64(0)
                valid = info[3] == 1
                ss[0] = T.float32(0)
                for c in T.serial(_CHUNKS):
                    col = (c * 32 + lane) * _VEC
                    for i in T.vectorized(_VEC):
                        v1[i] = values[token, col + i]
                    if ratio == 2:
                        for i in T.vectorized(_VEC):
                            g1[i] = gates[token, col + i]
                        if valid:
                            if info[4] == 1:
                                sid = state_ids[info[0]]
                                for i in T.vectorized(_VEC):
                                    v0[i] = pending_values[sid, col + i]
                                for i in T.vectorized(_VEC):
                                    g0[i] = pending_gates[sid, col + i]
                            else:
                                for i in T.vectorized(_VEC):
                                    v0[i] = T.Cast(
                                        T.float32, values[token - 1, col + i]
                                    )
                                for i in T.vectorized(_VEC):
                                    g0[i] = gates[token - 1, col + i]
                    for i in T.unroll(_VEC):
                        if valid:
                            if ratio == 2:
                                m = T.max(g0[i], g1[i])
                                e0 = T.exp(g0[i] - m)
                                e1 = T.exp(g1[i] - m)
                                d = e0 + e1
                                pooled = v0[i] * (e0 / d) + T.Cast(T.float32, v1[i]) * (
                                    e1 / d
                                )
                                x[c * _VEC + i] = T.Cast(
                                    T.float32, T.Cast(T.bfloat16, pooled)
                                )
                            else:
                                x[c * _VEC + i] = T.Cast(T.float32, v1[i])
                        else:
                            x[c * _VEC + i] = T.float32(0)
                        ss[0] += x[c * _VEC + i] * x[c * _VEC + i]
                    if ratio == 2:
                        # The live token's raw value and gate into its ring slot. The
                        # metadata gives only a request's last ``page`` rows of the step
                        # a ring slot (earlier rows would share slots); the kernel keeps
                        # that bound itself as well.
                        slot = slot_info[1]
                        if (token < nt) & (slot >= page) & (token >= info[5] - page):
                            base = (slot % page) * STATE + col
                            for i in T.vectorized(_VEC):
                                pool[slot // page, base + i] = v1[i]
                            for i in T.vectorized(_VEC):
                                pool[slot // page, base + DIM + i] = g1[i]
                total = T.warp_reduce_sum(ss[0])
                inverse = T.rsqrt(total / DIM + EPS)
                kv_slot = T.if_then_else(valid, slot_info[0], T.int64(-1))
                kv_page = kv_slot // kv_page_size
                for c in T.unroll(_CHUNKS):
                    col = (c * 32 + lane) * _VEC
                    for i in T.unroll(_VEC):
                        y[i] = T.Cast(
                            T.bfloat16, x[c * _VEC + i] * inverse * w[c * _VEC + i]
                        )
                    for i in T.vectorized(_VEC):
                        out[token, col + i] = y[i]
                    if write_kv:
                        for i in T.unroll(_VEC):
                            rv[i] = T.Cast(T.float32, y[i])
                        if rope_dim > 0 and c == _CHUNKS - 1:
                            if col >= rope_start:
                                pos = (
                                    slot_info[2] // ratio * ratio
                                    if ratio > 1
                                    else slot_info[2]
                                )
                                pair_base = (col - rope_start) // 2
                                for p in T.unroll(_VEC // 2):
                                    first = rv[2 * p]
                                    second = rv[2 * p + 1]
                                    cosine = cos_sin[pos, pair_base + p]
                                    sine = cos_sin[pos, rope_half + pair_base + p]
                                    rv[2 * p] = T.Cast(
                                        T.float32,
                                        T.Cast(
                                            T.bfloat16,
                                            T.ieee_fmaf(first, cosine, -(second * sine)),
                                        ),
                                    )
                                    rv[2 * p + 1] = T.Cast(
                                        T.float32,
                                        T.Cast(
                                            T.bfloat16,
                                            T.ieee_fmaf(first, sine, second * cosine),
                                        ),
                                    )
                        amax = 0.0
                        for i in T.unroll(_VEC):
                            amax = T.max(amax, T.abs(rv[i]))
                        amax = T.max(amax, T.shfl_xor(amax, 1))
                        amax = T.max(amax, T.shfl_xor(amax, 2))
                        if (kv_slot >= 0) & (kv_page < kv_pages):
                            kv_base = (kv_slot % kv_page_size) * _INDEXED_RECORD
                            scale = T.max(amax, 6.0 * 2.0**-9) / 6.0
                            scale_code = T.cast(
                                T.min(scale, _E4M3_MAX), T.float8_e4m3fn
                            )
                            decoded = T.cast(scale_code, T.float32)
                            word = T.call_extern(
                                "uint16",
                                "dsv41_e2m1x4",
                                rv[0] / decoded,
                                rv[1] / decoded,
                                rv[2] / decoded,
                                rv[3] / decoded,
                            )
                            kv_words[kv_page, (kv_base + col // 2) // 2] = word
                            if lane % 4 == 0:
                                group = c * 8 + lane // 4
                                kv_cache[
                                    kv_page, kv_base + _INDEXED_PAYLOAD + group
                                ] = T.reinterpret(scale_code, T.uint8)
                if lane == 0:
                    emitted[token] = valid
                    emitted_slots[token] = kv_slot

    if write_kv:

        @T.prim_func
        def compress(
            values: T.Tensor((rows, DIM), value_dtype),
            gates: T.Tensor((gate_rows, DIM), T.float32),
            weight: T.Tensor((DIM,), T.float32),
            starts: T.Tensor((max_requests + 1,), T.int32),
            positions: T.Tensor((max_requests,), T.int64),
            state_ids: T.Tensor((max_requests,), T.int64),
            slots: T.Tensor((rows,), T.int64),
            state_slots: T.Tensor((rows,), T.int64),
            counts: T.Tensor((2,), T.int32),
            pending_values: T.Tensor((max_requests, DIM), T.float32),
            pending_gates: T.Tensor((max_requests, DIM), T.float32),
            tags: T.Tensor((max_requests,), T.int64),
            pool: T.StridedTensor((blocks, width), (pool_stride, 1), T.float32),
            out: T.Tensor((rows, DIM), T.bfloat16),
            emitted: T.Tensor((rows,), T.bool),
            emitted_slots: T.Tensor((rows,), T.int64),
            token_positions: T.Tensor((rows,), T.int64),
            cos_sin: T.Tensor((positions_max, rope_dim), T.float32),
            kv_cache: T.StridedTensor(
                (kv_pages, kv_page_bytes), (kv_page_stride, 1), T.uint8
            ),
            kv_words: T.StridedTensor(
                (kv_pages, kv_page_words), (kv_word_stride, 1), T.uint16
            ),
        ):
            body(
                values,
                gates,
                weight,
                starts,
                positions,
                state_ids,
                slots,
                state_slots,
                counts,
                pending_values,
                pending_gates,
                tags,
                pool,
                out,
                emitted,
                emitted_slots,
                token_positions,
                cos_sin,
                kv_cache,
                kv_words,
            )

        return compress

    @T.prim_func
    def compress(
        values: T.Tensor((rows, DIM), value_dtype),
        gates: T.Tensor((gate_rows, DIM), T.float32),
        weight: T.Tensor((DIM,), T.float32),
        starts: T.Tensor((max_requests + 1,), T.int32),
        positions: T.Tensor((max_requests,), T.int64),
        state_ids: T.Tensor((max_requests,), T.int64),
        slots: T.Tensor((rows,), T.int64),
        state_slots: T.Tensor((rows,), T.int64),
        counts: T.Tensor((2,), T.int32),
        pending_values: T.Tensor((max_requests, DIM), T.float32),
        pending_gates: T.Tensor((max_requests, DIM), T.float32),
        tags: T.Tensor((max_requests,), T.int64),
        pool: T.StridedTensor((blocks, width), (pool_stride, 1), T.float32),
        out: T.Tensor((rows, DIM), T.bfloat16),
        emitted: T.Tensor((rows,), T.bool),
        emitted_slots: T.Tensor((rows,), T.int64),
    ):
        body(
            values,
            gates,
            weight,
            starts,
            positions,
            state_ids,
            slots,
            state_slots,
            counts,
            pending_values,
            pending_gates,
            tags,
            pool,
            out,
            emitted,
            emitted_slots,
            None,
            None,
            None,
            None,
        )

    return compress


def gather_partial_state(
    pool, table, starts, positions, counts, pending_values, pending_gates, tags,
    local_ids, *, page,
):  # fmt: skip
    """Each request's carry from the state ring."""
    max_requests = positions.shape[0]
    gather_partial_state_kernel(max_requests, page)(
        pool, table, starts, positions, counts, pending_values, pending_gates, tags,
        local_ids,
    )  # fmt: skip


def compress(
    rows, ratio, values, gates, weight, starts, positions, state_ids, slots, counts,
    out, emitted, emitted_slots, *, pending_values=None, pending_gates=None,
    tags=None, pool=None, state_slots=None, page=1,
    kv_cache=None, token_positions=None, cos_sin=None, kv_page_size=64,
):  # fmt: skip
    """Rows ``[0, rows)`` of one compression call (``out``, ``emitted`` and
    ``emitted_slots`` keep their contents past them). When ``kv_cache``,
    ``token_positions`` and ``cos_sin`` are provided, also rotates the trailing
    ``cos_sin.shape[1]`` columns in registers and writes the 288-byte ``indexed``
    KV-cache records at ``emitted_slots`` in the same launch."""
    max_requests = positions.shape[0]
    if ratio == 1:
        # Ratio 1 reads no gates, ring or carry: a placeholder of the shapes.
        dummy = placeholder(max_requests, out.device)
        gates = pending_values = pending_gates = pool = dummy
        tags, state_slots = state_ids, slots
    else:
        gates = gates[:rows]
    if kv_cache is not None and token_positions is not None and cos_sin is not None:
        pages = kv_cache.view(torch.uint8)
        if pages.dim() > 2:
            pages = pages.flatten(1)
        compress_kernel(
            ratio,
            max_requests,
            page,
            write_kv=True,
            kv_page_size=kv_page_size,
            rope_dim=cos_sin.shape[1],
        )(
            values[:rows],
            gates,
            weight,
            starts,
            positions,
            state_ids,
            slots[:rows],
            state_slots[:rows],
            counts,
            pending_values,
            pending_gates,
            tags,
            pool,
            out[:rows],
            emitted[:rows],
            emitted_slots[:rows],
            token_positions[:rows],
            cos_sin,
            pages,
            pages.view(torch.uint16),
        )
        return out[:rows], emitted_slots[:rows]
    compress_kernel(ratio, max_requests, page)(
        values[:rows], gates, weight, starts, positions, state_ids, slots[:rows],
        state_slots[:rows], counts, pending_values, pending_gates, tags, pool,
        out[:rows], emitted[:rows], emitted_slots[:rows],
    )  # fmt: skip
    return out[:rows], emitted_slots[:rows]


_PLACEHOLDERS: dict = {}


def placeholder(max_requests: int, device: torch.device) -> torch.Tensor:
    """One FP32 ``[max_requests, 512]`` buffer per device for ratio 1's unused
    operands, allocated on the first (eager) call, never during graph capture."""
    key = (max_requests, device)
    if key not in _PLACEHOLDERS:
        _PLACEHOLDERS[key] = torch.zeros((max_requests, DIM), device=device)
    return _PLACEHOLDERS[key]


__all__ = [
    "compress",
    "compress_kernel",
    "gather_partial_state",
    "gather_partial_state_kernel",
    "placeholder",
]
