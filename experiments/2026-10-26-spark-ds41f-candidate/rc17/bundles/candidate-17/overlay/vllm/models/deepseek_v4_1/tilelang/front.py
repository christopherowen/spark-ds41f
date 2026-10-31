# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's attention front after the fused Q-A/KV projection, in one launch.

At decode sizes the projection's two latents went through five small kernels:
TileKernels' RMSNorm of each (``norm_forward``), ``per_token_cast`` of the
normalized q latent for Q-B, and the kv latent's RoPE and SWA cache record
(``rope_kernel``, ``cache_writer_kernel``). ``attention_front`` is one kernel that
does all of it, each row's arithmetic theirs to the bit:

- q CTAs: TileKernels' ``norm_forward_and_per_token_cast`` (a TileLang port of its
  CUDA kernel: the same row split, fragment layouts, reducers and cast helpers)
  writes the normalized q latent in BF16 and its MXFP8 cast, which equals
  ``per_token_cast`` of that BF16 row;
- kv CTAs: the same norm of the kv latent, then, in registers, ``rope_kernel``'s
  rotation of its last 64 columns and ``cache_writer_kernel``'s SWA record of the
  rotated row. The rotated row is written too (``kv``), as the in-place RoPE left
  it.

A thread of the kv norm holds 8-column vectors ``(v * 16 + lane) * 8``: the 64
rotated columns are the last vector of lanes 8-15, exactly ``rope_kernel``'s lanes
0-7, and a 32-column record group is four adjacent lanes, whose amax two lane
shuffles give (a maximum does not depend on its order).
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import math

import tilelang
import torch
from tilelang.cuda import language as T

from . import pdl, prepare
from .cache_writer import _CVT_HEADER, _E4M3_MAX, RECORD_BYTES

Q_LATENT = 1280
KV_LATENT = 512
ROPE_DIM = 64
_THREADS = 128
_VEC = 8  # BF16 values per 16-byte vector


def _row_split(hidden: int) -> dict:
    """TileKernels' split of a BF16 row of ``hidden`` columns over 128 threads
    (``get_norm_forward_and_per_token_cast_kernel_cuda``) and its fragment layouts."""
    vec_size = math.gcd(16 // 2, hidden)
    assert vec_size == _VEC
    num_vecs = hidden // vec_size
    row_threads = min(
        _THREADS,
        max(
            min(8, tilelang.next_power_of_2(num_vecs)),
            tilelang.next_power_of_2(-(-hidden // 32)),
        ),
    )
    block_m = _THREADS // row_threads
    warp_threads = min(32, row_threads)
    num_row_warps = row_threads // warp_threads
    return dict(
        hidden=hidden,
        num_thread_vec=-(-num_vecs // row_threads),
        row_threads=row_threads,
        block_m=block_m,
        warp_threads=warp_threads,
        num_row_warps=num_row_warps,
        row_layout=T.Fragment(
            (block_m, row_threads), forward_thread_fn=lambda i, j: i * row_threads + j
        ),
        warp_layout=T.Fragment(
            (block_m, num_row_warps),
            forward_thread_fn=lambda i, w, rep: (
                i * row_threads + w * warp_threads + rep
            ),
            replicate=warp_threads,
        ),
        warp_reduce_layout=T.Fragment(
            (block_m, num_row_warps, warp_threads),
            forward_thread_fn=lambda i, w, j: i * row_threads + w * warp_threads + j,
        ),
        row_reduce_layout=T.Fragment(
            (block_m, row_threads, num_row_warps),
            forward_thread_fn=lambda i, j, w: i * row_threads + j,
        ),
    )


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def attention_front_kernel(eps: float, page_size: int, slot_dtype: str = "int64"):
    """q CTAs (two rows each) then kv CTAs (eight rows each); see the module
    docstring. ``qa_kv`` is the fused projection's output, q columns first."""
    from tile_kernels.quant.common import (
        get_cast_output_config,
        get_sf_and_inv,
        get_sf_index,
        get_sf_inv_dtype,
        get_sf_shape,
    )

    out_config = get_cast_output_config("e4m3", (1, 32), False, True, True)
    sf_inv_dtype = get_sf_inv_dtype(T.bfloat16, out_config)
    group_lanes = 32 // _VEC
    out_scale = 1.0
    q, k = _row_split(Q_LATENT), _row_split(KV_LATENT)
    assert (
        k["num_row_warps"] == 1 and k["row_threads"] == 16 and k["num_thread_vec"] == 4
    )
    half = ROPE_DIM // 2
    rope_vec = KV_LATENT // _VEC - ROPE_DIM // _VEC  # the first rotated vector: 56
    last = k["num_thread_vec"] - 1  # the vector index (of four) holding the rotation
    record = RECORD_BYTES["swa"]
    width = Q_LATENT + KV_LATENT
    M, Mq = T.dynamic("M, Mq")
    sf_shape = get_sf_shape((M, Q_LATENT), out_config)
    positions_max = T.dynamic("positions_max")
    pages, page_bytes, page_stride = T.dynamic("pages, page_bytes, page_stride")
    page_words, word_stride = T.dynamic("page_words, word_stride")

    def make_norm(split):
        """TileKernels' RMSNorm of ``split``'s rows from ``column`` of ``qa_kv``: the
        sum of squares reduced over each row's threads (``sum_fragment``), in its
        order."""
        hidden, row_threads = split["hidden"], split["row_threads"]
        block_m, warp_threads = split["block_m"], split["warp_threads"]
        num_row_warps = split["num_row_warps"]

        @T.macro
        def norm(
            qa_kv,
            column,
            pid_token,
            tid,
            sum_partial,
            warp_reducer,
            warp_fragment,
            sum_shared,
            row_partial,
            sum_reducer,
            sum_fragment,
            x_local,
            sum_local,
        ):
            row_id = tid // row_threads
            row_offset = tid % row_threads
            token_id = pid_token * block_m + row_id
            T.clear(sum_local)
            for vi in T.unroll(split["num_thread_vec"]):
                token_offset = (vi * row_threads + row_offset) * _VEC
                if token_id < M and token_offset < hidden:
                    for j in T.vectorized(_VEC):
                        x_local[vi, j] = qa_kv[token_id, column + token_offset + j]
                else:
                    T.clear(x_local[vi, :])
                for j in T.vectorized(_VEC):
                    sum_local[j % 2] += T.float32(x_local[vi, j]) * T.float32(
                        x_local[vi, j]
                    )
            # Reduce RSTD over the threads of the row
            sum_partial[row_id, row_offset] = sum_local[0] + sum_local[1]
            T.reducer_init(warp_reducer)
            for i, w, j in T.Parallel(
                block_m,
                num_row_warps,
                warp_threads,
                loop_layout=split["warp_reduce_layout"],
            ):
                T.reducer_update(
                    warp_reducer[i, w], sum_partial[i, w * warp_threads + j]
                )
            T.finalize_reducer(warp_reducer, warp_fragment)
            if row_offset % warp_threads == 0:
                sum_shared[row_id, row_offset // warp_threads] = warp_fragment[
                    row_id, row_offset // warp_threads
                ]
            T.sync_threads()
            for i, j, w in T.Parallel(
                block_m,
                row_threads,
                num_row_warps,
                loop_layout=split["row_reduce_layout"],
            ):
                row_partial[i, j, w] = sum_shared[i, w]
            T.reducer_init(sum_reducer)
            for i, j, w in T.Parallel(
                block_m,
                row_threads,
                num_row_warps,
                loop_layout=split["row_reduce_layout"],
            ):
                T.reducer_update(sum_reducer[i, j], row_partial[i, j, w])
            T.finalize_reducer(sum_reducer, sum_fragment)

        return norm

    q_norm, kv_norm = make_norm(q), make_norm(k)
    qb, qt, qw = q["block_m"], q["row_threads"], q["num_row_warps"]
    kb, kt = k["block_m"], k["row_threads"]

    @T.prim_func
    def attention_front(
        qa_kv: T.Tensor((M, width), T.bfloat16),
        q_weight: T.Tensor((Q_LATENT,), T.bfloat16),
        kv_weight: T.Tensor((KV_LATENT,), T.bfloat16),
        positions: T.Tensor((M,), T.int64),
        cos_sin: T.Tensor((positions_max, ROPE_DIM), T.float32),
        cache: T.StridedTensor((pages, page_bytes), (page_stride, 1), T.uint8),
        words: T.StridedTensor((pages, page_words), (word_stride, 1), T.uint32),
        slots: T.Tensor((M,), slot_dtype),
        qn: T.Tensor((M, Q_LATENT), T.bfloat16),
        q_values: T.Tensor((Mq, Q_LATENT), out_config.dtype),
        q_scales: T.Tensor((Mq, sf_shape[1]), out_config.sf_dtype),
        kv: T.Tensor((M, KV_LATENT), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(M, qb) + T.ceildiv(M, kb), threads=_THREADS) as bx:
            T.import_source(_CVT_HEADER)
            pdl.chain()
            tid = T.get_thread_binding()
            q_sum_partial = T.alloc_fragment((qb, qt), T.float32)
            q_warp_reducer = T.alloc_reducer((qb, qw), T.float32, op="sum")
            q_warp_fragment = T.alloc_fragment((qb, qw), T.float32)
            q_sum_shared = T.alloc_shared((qb, qw), T.float32)
            q_row_partial = T.alloc_fragment((qb, qt, qw), T.float32)
            q_sum_reducer = T.alloc_reducer((qb, qt), T.float32, op="sum")
            q_sum_fragment = T.alloc_fragment((qb, qt), T.float32)
            k_sum_partial = T.alloc_fragment((kb, kt), T.float32)
            k_warp_reducer = T.alloc_reducer((kb, 1), T.float32, op="sum")
            k_warp_fragment = T.alloc_fragment((kb, 1), T.float32)
            k_sum_shared = T.alloc_shared((kb, 1), T.float32)
            k_row_partial = T.alloc_fragment((kb, kt, 1), T.float32)
            k_sum_reducer = T.alloc_reducer((kb, kt), T.float32, op="sum")
            k_sum_fragment = T.alloc_fragment((kb, kt), T.float32)
            T.annotate_layout(
                {
                    q_sum_partial: q["row_layout"],
                    q_warp_fragment: q["warp_layout"],
                    q_row_partial: q["row_reduce_layout"],
                    q_sum_fragment: q["row_layout"],
                    k_sum_partial: k["row_layout"],
                    k_warp_fragment: k["warp_layout"],
                    k_row_partial: k["row_reduce_layout"],
                    k_sum_fragment: k["row_layout"],
                }
            )
            q_x = T.alloc_local((q["num_thread_vec"], _VEC), T.bfloat16)
            q_y = T.alloc_local((q["num_thread_vec"], _VEC), T.bfloat16)
            k_x = T.alloc_local((k["num_thread_vec"], _VEC), T.bfloat16)
            k_y = T.alloc_local((k["num_thread_vec"], _VEC), T.bfloat16)
            y_fp32_local = T.alloc_local((_VEC,), T.float32)
            sum_local = T.alloc_local((2,), T.float32)
            weight_local = T.alloc_local((_VEC,), T.bfloat16)
            sf_local = T.alloc_local((q["num_thread_vec"],), out_config.sf_dtype)
            amax_local = T.alloc_local((q["num_thread_vec"],), T.float32)
            cs = T.alloc_local((_VEC,), T.float32)
            values = T.alloc_local((_VEC,), T.float32)

            if bx < T.ceildiv(M, qb):
                # The q latent: TileKernels' norm with its cast, and a BF16 copy.
                q_norm(
                    qa_kv,
                    0,
                    bx,
                    tid,
                    q_sum_partial,
                    q_warp_reducer,
                    q_warp_fragment,
                    q_sum_shared,
                    q_row_partial,
                    q_sum_reducer,
                    q_sum_fragment,
                    q_x,
                    sum_local,
                )
                row_offset = tid % qt
                token_id = bx * qb + tid // qt
                rstd_value = T.alloc_var(
                    T.float32,
                    init=T.rsqrt(
                        q_sum_fragment[tid // qt, row_offset] / Q_LATENT + eps
                    ),
                )
                for vi in T.unroll(q["num_thread_vec"]):
                    token_offset = (vi * qt + row_offset) * _VEC
                    is_valid = token_id < M and token_offset < Q_LATENT
                    if token_offset < Q_LATENT:
                        for j in T.vectorized(_VEC):
                            weight_local[j] = q_weight[token_offset + j]
                    for j in T.vectorized(_VEC):
                        y_fp32_local[j] = (
                            T.float32(q_x[vi, j])
                            * rstd_value
                            * (T.float32(weight_local[j]) * out_scale)
                        )
                    for j in T.vectorized(_VEC):
                        q_y[vi, j] = y_fp32_local[j]
                    if is_valid:
                        for j in T.vectorized(_VEC):
                            qn[token_id, token_offset + j] = q_y[vi, j]
                    # Reduce SF over the threads of the group, as in `per_token_cast`
                    amax = T.alloc_var(T.float32, init=0.0)
                    for j in T.unroll(_VEC):
                        amax = T.max(amax, T.abs(y_fp32_local[j]))
                    amax = T.float32(T.cast(amax, T.bfloat16))
                    for i in T.unroll(int(math.log2(group_lanes))):
                        amax = T.max(amax, T.shfl_xor(amax, 1 << i))
                    amax_local[vi] = amax
                for vi in T.unroll(q["num_thread_vec"]):
                    token_offset = (vi * qt + row_offset) * _VEC
                    sf, sf_inv = get_sf_and_inv(
                        amax_local[vi], out_config, sf_inv_dtype
                    )
                    sf_local[vi] = sf
                    if token_id < M and token_offset < Q_LATENT:
                        for j in T.vectorized(_VEC):
                            q_values[token_id, token_offset + j] = q_y[vi, j] * sf_inv
                if token_id < M and row_offset % group_lanes == 0:
                    for vi in T.unroll(q["num_thread_vec"]):
                        token_offset = (vi * qt + row_offset) * _VEC
                        if token_offset < Q_LATENT:
                            sf_row, sf_column = get_sf_index(
                                token_id, token_offset // 32, out_config
                            )
                            q_scales[sf_row, sf_column] = sf_local[vi]
            else:
                # The kv latent: TileKernels' norm, then the RoPE and the SWA record
                # from registers.
                pid = bx - T.ceildiv(M, qb)
                kv_norm(
                    qa_kv,
                    Q_LATENT,
                    pid,
                    tid,
                    k_sum_partial,
                    k_warp_reducer,
                    k_warp_fragment,
                    k_sum_shared,
                    k_row_partial,
                    k_sum_reducer,
                    k_sum_fragment,
                    k_x,
                    sum_local,
                )
                lane = tid % kt
                token_id = pid * kb + tid // kt
                rstd_value = T.alloc_var(
                    T.float32,
                    init=T.rsqrt(k_sum_fragment[tid // kt, lane] / KV_LATENT + eps),
                )
                for vi in T.unroll(k["num_thread_vec"]):
                    token_offset = (vi * kt + lane) * _VEC
                    for j in T.vectorized(_VEC):
                        weight_local[j] = kv_weight[token_offset + j]
                    for j in T.vectorized(_VEC):
                        y_fp32_local[j] = (
                            T.float32(k_x[vi, j])
                            * rstd_value
                            * (T.float32(weight_local[j]) * out_scale)
                        )
                    for j in T.vectorized(_VEC):
                        k_y[vi, j] = y_fp32_local[j]
                # rope_kernel's rotation of the last 64 columns (GPT-J pairs): the
                # last vector of lanes 8-15, rope_kernel's lanes 0-7.
                if (last * kt + lane >= rope_vec) & (token_id < M):
                    rope_lane = last * kt + lane - rope_vec
                    position = positions[token_id]
                    for i in T.vectorized(_VEC // 2):
                        cs[i] = cos_sin[position, rope_lane * (_VEC // 2) + i]
                    for i in T.vectorized(_VEC // 2):
                        cs[_VEC // 2 + i] = cos_sin[
                            position, half + rope_lane * (_VEC // 2) + i
                        ]
                    for j in T.unroll(_VEC // 2):
                        first = T.cast(k_y[last, 2 * j], T.float32)
                        second = T.cast(k_y[last, 2 * j + 1], T.float32)
                        cosine = cs[j]
                        sine = cs[_VEC // 2 + j]
                        k_y[last, 2 * j] = T.cast(
                            T.ieee_fmaf(first, cosine, -(second * sine)), T.bfloat16
                        )
                        k_y[last, 2 * j + 1] = T.cast(
                            T.ieee_fmaf(first, sine, second * cosine), T.bfloat16
                        )
                # cache_writer_kernel's SWA record of the rotated row.
                for vi in T.unroll(k["num_thread_vec"]):
                    col = (vi * kt + lane) * _VEC
                    if token_id < M:
                        for j in T.vectorized(_VEC):
                            kv[token_id, col + j] = k_y[vi, j]
                    amax = T.alloc_var(T.float32, init=0.0)
                    for j in T.unroll(_VEC):
                        values[j] = T.cast(k_y[vi, j], T.float32)
                        amax = T.max(amax, T.abs(values[j]))
                    amax = T.max(amax, T.shfl_xor(amax, 1))
                    amax = T.max(amax, T.shfl_xor(amax, 2))
                    if token_id < M:
                        slot = T.cast(slots[token_id], T.int64)
                        page = slot // page_size
                        if (slot >= 0) & (page < pages):
                            base = (slot % page_size) * record
                            bits = T.reinterpret(
                                T.fmul(T.max(amax, 1e-4), T.float32(1.0 / _E4M3_MAX)),
                                T.uint32,
                            )
                            bumped = T.if_then_else(
                                (bits & 0x7FFFFF) != 0,
                                (bits + 0x800000) & 0x7F800000,
                                bits,
                            )
                            exponent = T.cast(T.shift_right(bumped, 23) & 0xFF, T.int32)
                            # The exact reciprocal of the power-of-two scale.
                            inverse = T.reinterpret(
                                T.cast(T.shift_left(254 - exponent, 23), T.uint32),
                                T.float32,
                            )
                            for h in T.unroll(2):
                                words[page, (base + col + 4 * h) // 4] = T.call_extern(
                                    "uint32",
                                    "dsv41_e4m3x4",
                                    T.fmul(values[4 * h], inverse),
                                    T.fmul(values[4 * h + 1], inverse),
                                    T.fmul(values[4 * h + 2], inverse),
                                    T.fmul(values[4 * h + 3], inverse),
                                )
                            if lane % group_lanes == 0:
                                cache[page, base + KV_LATENT + col // 32] = T.cast(
                                    exponent, T.uint8
                                )

    return attention_front


def front_call(eps: float, page_size: int, slot_dtype: torch.dtype):
    """The kernel ``attention_front`` launches, as a ``prepare.call``."""
    return prepare.call(
        attention_front_kernel,
        eps,
        page_size,
        str(slot_dtype).removeprefix("torch."),
    )


def attention_front(
    qa_kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int,
    qn: torch.Tensor,
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    kv: torch.Tensor,
) -> None:
    """The front of ``qa_kv``'s rows ([T, 1792] BF16): ``qn`` and its MXFP8 cast
    (``q_values`` and ``q_scales`` bytes, whole decode tiles), the rotated ``kv`` and
    its SWA records in ``cache`` at ``slot_mapping``."""
    if qa_kv.shape[0] == 0:
        return
    pages = cache.view(torch.uint8)
    if pages.dim() > 2:  # [pages, records, record bytes]
        pages = pages.flatten(1)
    function, args, kwargs = front_call(eps, page_size, slot_mapping.dtype)
    function(*args, **kwargs)(
        qa_kv,
        q_weight,
        kv_weight,
        positions,
        cos_sin_cache,
        pages,
        pages.view(torch.uint32),
        slot_mapping,
        qn,
        q_values,
        q_scales.view(torch.uint8),
        kv,
    )


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def context_kv_front_kernel(eps: float, page_size: int, slot_dtype: str = "int64"):
    """The KV CTA body of ``attention_front_kernel`` on standalone ``[M, 512]`` BF16
    ``kv_raw`` rows (DSpark's context KV projections): TileKernels' ``norm_forward``,
    trailing 64-column ``rope_kernel`` rotation in registers, and ``cache_writer``'s
    528-byte SWA record write in one launch, without writing the rotated BF16 row."""
    group_lanes = 32 // _VEC
    out_scale = 1.0
    k = _row_split(KV_LATENT)
    assert (
        k["num_row_warps"] == 1 and k["row_threads"] == 16 and k["num_thread_vec"] == 4
    )
    half = ROPE_DIM // 2
    rope_vec = KV_LATENT // _VEC - ROPE_DIM // _VEC
    last = k["num_thread_vec"] - 1
    record = RECORD_BYTES["swa"]
    kb, kt = k["block_m"], k["row_threads"]
    M = T.dynamic("M")
    positions_max = T.dynamic("positions_max")
    pages, page_bytes, page_stride = T.dynamic("pages, page_bytes, page_stride")
    page_words, word_stride = T.dynamic("page_words, word_stride")

    @T.prim_func
    def attention_front(
        kv_raw: T.Tensor((M, KV_LATENT), T.bfloat16),
        kv_weight: T.Tensor((KV_LATENT,), T.bfloat16),
        positions: T.Tensor((M,), T.int64),
        cos_sin: T.Tensor((positions_max, ROPE_DIM), T.float32),
        cache: T.StridedTensor((pages, page_bytes), (page_stride, 1), T.uint8),
        words: T.StridedTensor((pages, page_words), (word_stride, 1), T.uint32),
        slots: T.Tensor((M,), slot_dtype),
    ):
        with T.Kernel(T.ceildiv(M, kb), threads=_THREADS) as pid:
            T.import_source(_CVT_HEADER)
            pdl.chain()
            tid = T.get_thread_binding()
            k_sum_partial = T.alloc_fragment((kb, kt), T.float32)
            k_warp_reducer = T.alloc_reducer((kb, 1), T.float32, op="sum")
            k_warp_fragment = T.alloc_fragment((kb, 1), T.float32)
            k_sum_shared = T.alloc_shared((kb, 1), T.float32)
            k_row_partial = T.alloc_fragment((kb, kt, 1), T.float32)
            k_sum_reducer = T.alloc_reducer((kb, kt), T.float32, op="sum")
            k_sum_fragment = T.alloc_fragment((kb, kt), T.float32)
            T.annotate_layout(
                {
                    k_sum_partial: k["row_layout"],
                    k_warp_fragment: k["warp_layout"],
                    k_row_partial: k["row_reduce_layout"],
                    k_sum_fragment: k["row_layout"],
                }
            )
            k_x = T.alloc_local((k["num_thread_vec"], _VEC), T.bfloat16)
            k_y = T.alloc_local((k["num_thread_vec"], _VEC), T.bfloat16)
            y_fp32_local = T.alloc_local((_VEC,), T.float32)
            sum_local = T.alloc_local((2,), T.float32)
            weight_local = T.alloc_local((_VEC,), T.bfloat16)
            cs = T.alloc_local((_VEC,), T.float32)
            values = T.alloc_local((_VEC,), T.float32)

            row_id = tid // kt
            lane = tid % kt
            token_id = pid * kb + row_id
            T.clear(sum_local)
            for vi in T.unroll(k["num_thread_vec"]):
                token_offset = (vi * kt + lane) * _VEC
                if token_id < M:
                    for j in T.vectorized(_VEC):
                        k_x[vi, j] = kv_raw[token_id, token_offset + j]
                else:
                    T.clear(k_x[vi, :])
                for j in T.vectorized(_VEC):
                    sum_local[j % 2] += T.float32(k_x[vi, j]) * T.float32(k_x[vi, j])
            k_sum_partial[row_id, lane] = sum_local[0] + sum_local[1]
            T.reducer_init(k_warp_reducer)
            for i, w, j in T.Parallel(
                kb, 1, kt, loop_layout=k["warp_reduce_layout"]
            ):
                T.reducer_update(k_warp_reducer[i, w], k_sum_partial[i, w * kt + j])
            T.finalize_reducer(k_warp_reducer, k_warp_fragment)
            if lane == 0:
                k_sum_shared[row_id, 0] = k_warp_fragment[row_id, 0]
            T.sync_threads()
            for i, j, w in T.Parallel(kb, kt, 1, loop_layout=k["row_reduce_layout"]):
                k_row_partial[i, j, w] = k_sum_shared[i, w]
            T.reducer_init(k_sum_reducer)
            for i, j, w in T.Parallel(kb, kt, 1, loop_layout=k["row_reduce_layout"]):
                T.reducer_update(k_sum_reducer[i, j], k_row_partial[i, j, w])
            T.finalize_reducer(k_sum_reducer, k_sum_fragment)

            rstd_value = T.alloc_var(
                T.float32,
                init=T.rsqrt(k_sum_fragment[row_id, lane] / KV_LATENT + eps),
            )
            for vi in T.unroll(k["num_thread_vec"]):
                token_offset = (vi * kt + lane) * _VEC
                for j in T.vectorized(_VEC):
                    weight_local[j] = kv_weight[token_offset + j]
                for j in T.vectorized(_VEC):
                    y_fp32_local[j] = (
                        T.float32(k_x[vi, j])
                        * rstd_value
                        * (T.float32(weight_local[j]) * out_scale)
                    )
                for j in T.vectorized(_VEC):
                    k_y[vi, j] = y_fp32_local[j]
            if (last * kt + lane >= rope_vec) & (token_id < M):
                rope_lane = last * kt + lane - rope_vec
                position = positions[token_id]
                for i in T.vectorized(_VEC // 2):
                    cs[i] = cos_sin[position, rope_lane * (_VEC // 2) + i]
                for i in T.vectorized(_VEC // 2):
                    cs[_VEC // 2 + i] = cos_sin[
                        position, half + rope_lane * (_VEC // 2) + i
                    ]
                for j in T.unroll(_VEC // 2):
                    first = T.cast(k_y[last, 2 * j], T.float32)
                    second = T.cast(k_y[last, 2 * j + 1], T.float32)
                    cosine = cs[j]
                    sine = cs[_VEC // 2 + j]
                    k_y[last, 2 * j] = T.cast(
                        T.ieee_fmaf(first, cosine, -(second * sine)), T.bfloat16
                    )
                    k_y[last, 2 * j + 1] = T.cast(
                        T.ieee_fmaf(first, sine, second * cosine), T.bfloat16
                    )
            for vi in T.unroll(k["num_thread_vec"]):
                col = (vi * kt + lane) * _VEC
                amax = T.alloc_var(T.float32, init=0.0)
                for j in T.unroll(_VEC):
                    values[j] = T.cast(k_y[vi, j], T.float32)
                    amax = T.max(amax, T.abs(values[j]))
                amax = T.max(amax, T.shfl_xor(amax, 1))
                amax = T.max(amax, T.shfl_xor(amax, 2))
                if token_id < M:
                    slot = T.cast(slots[token_id], T.int64)
                    page = slot // page_size
                    if (slot >= 0) & (page < pages):
                        base = (slot % page_size) * record
                        bits = T.reinterpret(
                            T.fmul(T.max(amax, 1e-4), T.float32(1.0 / _E4M3_MAX)),
                            T.uint32,
                        )
                        bumped = T.if_then_else(
                            (bits & 0x7FFFFF) != 0,
                            (bits + 0x800000) & 0x7F800000,
                            bits,
                        )
                        exponent = T.cast(T.shift_right(bumped, 23) & 0xFF, T.int32)
                        inverse = T.reinterpret(
                            T.cast(T.shift_left(254 - exponent, 23), T.uint32),
                            T.float32,
                        )
                        for h in T.unroll(2):
                            words[page, (base + col + 4 * h) // 4] = T.call_extern(
                                "uint32",
                                "dsv41_e4m3x4",
                                T.fmul(values[4 * h], inverse),
                                T.fmul(values[4 * h + 1], inverse),
                                T.fmul(values[4 * h + 2], inverse),
                                T.fmul(values[4 * h + 3], inverse),
                            )
                        if lane % group_lanes == 0:
                            cache[page, base + KV_LATENT + col // 32] = T.cast(
                                exponent, T.uint8
                            )

    return attention_front


def context_kv_front_call(eps: float, page_size: int, slot_dtype: torch.dtype):
    """The kernel ``context_kv_front`` launches, as a ``prepare.call``."""
    return prepare.call(
        context_kv_front_kernel,
        eps,
        page_size,
        str(slot_dtype).removeprefix("torch."),
    )


def context_kv_front(
    kv_raw: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int,
) -> None:
    """Fused ``kv_norm`` + ``rope`` + SWA ``cache_writer`` for ``kv_raw`` ([T, 512])."""
    if kv_raw.shape[0] == 0:
        return
    pages = cache.view(torch.uint8)
    if pages.dim() > 2:
        pages = pages.flatten(1)
    function, args, kwargs = context_kv_front_call(eps, page_size, slot_mapping.dtype)
    function(*args, **kwargs)(
        kv_raw,
        kv_weight,
        positions,
        cos_sin_cache,
        pages,
        pages.view(torch.uint32),
        slot_mapping,
    )


__all__ = [
    "attention_front",
    "attention_front_kernel",
    "context_kv_front",
    "context_kv_front_call",
    "context_kv_front_kernel",
    "front_call",
]
