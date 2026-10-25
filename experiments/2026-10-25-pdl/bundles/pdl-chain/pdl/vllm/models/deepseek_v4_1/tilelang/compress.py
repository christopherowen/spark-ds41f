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

This replaces B12X's ``mla_compress`` (one 128-thread block per token of the whole
batch capacity every call, then a pending-ring commit vLLM never reads) and vLLM's two
Triton ring kernels (``_load_partial_state``, ``_save_partial_states``, the latter also
over the whole capacity): two launches sized by the step. One warp owns a token, 16
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

DIM = 512
STATE = 2 * DIM  # a ring slot: the value, then the gate
EPS = 1e-20
_VEC = 4  # FP32 per 16-byte access
_CHUNKS = DIM // (32 * _VEC)  # 16-byte accesses per lane
_PASS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


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
            pdl.chain()
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
def compress_kernel(ratio: int, max_requests: int, page: int, warps: int = 4):
    """Rows ``[0, rows)`` of the step: latents, emitted flags and slots (zero, false
    and -1 past the live tokens), and at ratio 2 the live tokens' ring saves."""
    assert ratio in (1, 2)
    rows = T.dynamic("rows")
    gate_rows = T.dynamic("gate_rows")  # rows at ratio 2; any buffer at ratio 1
    blocks = T.dynamic("blocks")
    width = T.dynamic("width")
    pool_stride = T.dynamic("pool_stride")  # the allocator pads its pages
    value_dtype = T.float32 if ratio == 2 else T.bfloat16

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
        with T.Kernel(T.ceildiv(rows, warps), threads=warps * 32) as bx:
            pdl.chain()
            tx = T.get_thread_binding()
            token = bx * warps + tx // 32
            lane = tx % 32
            x = T.alloc_local((_CHUNKS * _VEC,), T.float32)
            v1 = T.alloc_local((_VEC,), value_dtype)
            g1 = T.alloc_local((_VEC,), T.float32)
            v0 = T.alloc_local((_VEC,), T.float32)
            g0 = T.alloc_local((_VEC,), T.float32)
            w = T.alloc_local((_VEC,), T.float32)
            y = T.alloc_local((_VEC,), T.bfloat16)
            # request, first, position, valid, pending carry, request end
            info = T.alloc_local((6,), T.int64)
            ss = T.alloc_local((1,), T.float32)
            if token < rows:
                nt = counts[0]
                nr = counts[1]
                info[0] = T.int64(-1)
                info[1] = T.int64(0)
                info[2] = T.int64(-1)
                info[3] = T.int64(0)
                info[4] = T.int64(0)
                info[5] = T.int64(-1)
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
                        sid = state_ids[q]
                        start_pos = positions[q]
                        info[5] = end
                        if (
                            (first >= 0)
                            & (first <= token)
                            & (end <= nt)
                            & (token < end)
                            & (sid >= 0)
                            & (sid < max_requests)
                            & (start_pos >= 0)
                        ):
                            info[1] = first
                            info[2] = start_pos + (token - first)
                            info[3] = T.int64(1)
                            if ratio == 2:
                                if info[2] % 2 == 1:
                                    if token == first:
                                        info[4] = T.int64(1)
                                        if tags[sid] != info[2] - 1:
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
                        slot = state_slots[token]
                        if (token < nt) & (slot >= page) & (token >= info[5] - page):
                            base = (slot % page) * STATE + col
                            for i in T.vectorized(_VEC):
                                pool[slot // page, base + i] = v1[i]
                            for i in T.vectorized(_VEC):
                                pool[slot // page, base + DIM + i] = g1[i]
                total = T.warp_reduce_sum(ss[0])
                inverse = T.rsqrt(total / DIM + EPS)
                for c in T.serial(_CHUNKS):
                    col = (c * 32 + lane) * _VEC
                    for i in T.vectorized(_VEC):
                        w[i] = weight[col + i]
                    for i in T.unroll(_VEC):
                        y[i] = T.Cast(T.bfloat16, x[c * _VEC + i] * inverse * w[i])
                    for i in T.vectorized(_VEC):
                        out[token, col + i] = y[i]
                if lane == 0:
                    emitted[token] = valid
                    emitted_slots[token] = T.if_then_else(
                        valid, slots[token], T.int64(-1)
                    )

    return compress


def gather_partial_state(
    pool, table, starts, positions, counts, pending_values, pending_gates, tags,
    local_ids, *, page,
):  # fmt: skip
    """vLLM's ``_load_partial_state``: each request's carry from the state ring."""
    max_requests = positions.shape[0]
    gather_partial_state_kernel(max_requests, page)(
        pool, table, starts, positions, counts, pending_values, pending_gates, tags,
        local_ids,
    )  # fmt: skip


def compress(
    rows, ratio, values, gates, weight, starts, positions, state_ids, slots, counts,
    out, emitted, emitted_slots, *, pending_values=None, pending_gates=None,
    tags=None, pool=None, state_slots=None, page=1,
):  # fmt: skip
    """Rows ``[0, rows)`` of one compression call (``out``, ``emitted`` and
    ``emitted_slots`` keep their contents past them)."""
    max_requests = positions.shape[0]
    if ratio == 1:
        # Ratio 1 reads no gates, ring or carry: a placeholder of the shapes.
        dummy = placeholder(max_requests, out.device)
        gates = pending_values = pending_gates = pool = dummy
        tags, state_slots = state_ids, slots
    else:
        gates = gates[:rows]
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
