# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang kernels of DeepSeek-V4.1's vision tower: the ViT and the aligner.

One image is evaluated per call; every kernel takes the image's live rows as a
dynamic dimension, so a capacity-sized workspace serves any grid and one compiled
kernel serves every image size. The arithmetic of each elementwise and row-wise
operation is B12X's (``b12x.norm.vision`` and ``b12x.norm.hyperconnection``), with
each rounding written out (``T.fma``, ``T.fmul``, ``T.ieee_*``) so the compiler
cannot contract or reorder it:

- ``pad_columns``: rows of ``width`` BF16 values copied into ``padded``-wide rows
  whose tail is zero (the patch embedding's 588 columns become 640, whole K tiles,
  16-byte rows). Exact.
- ``linear``: ``A @ W^T`` on BF16 tensor cores for images of more than
  ``GEMV_ROWS`` rows. Each 64-wide K block is summed alone, from zero, by four
  16-wide MMA steps, and the blocks' FP32 sums are added in K order on the CUDA
  cores: the tensor core's accumulation never chains more than four steps. (One
  chain of every step rounded 0.7% of the outputs of a K = 9216 row away from the
  correctly rounded value, and twice as many as cuBLAS's at K = 2816.) A row's bits
  do not depend on the tile shape. With
  ``bias`` the BF16 bias is added in FP32 before the one rounding to BF16, as
  B12X's projections do; with ``gelu`` the rounded biased value goes through GELU
  (below) in the same epilogue.
- ``swiglu_linear``: the MLP's gate and up projections of one column block in one
  CTA (two blocked accumulations sharing the activation tile), each rounded to
  BF16 as B12X's GEMM stores them, then B12X's vision SwiGLU (limit infinite, so no
  clamp): ``silu = g * (1 / (1 + __expf(-g)))`` (CuTe's ``exp(..., fastmath=True)``,
  an IEEE reciprocal), rounded to BF16, times ``u``, rounded to BF16. The
  ``[rows, 2I]`` gate/up tensor is never stored.
- ``gemv``: images of at most ``GEMV_ROWS`` rows (and merged rows), B12X's SIMT
  projection (``SmallNGemvKernel``, the plan B12X runs below nine rows) to the
  rounding: one 128-thread CTA an output column and eight rows; thread ``t`` adds
  ``w * x`` by FMA from zero over the 8-wide K vectors ``t, t + 128, ...``, elements
  in order (single elements ``t, t + 128, ...`` when K is not whole vectors, the
  588-wide patches); a butterfly over each warp, the four warp sums in one warp's
  butterfly; the bias added in FP32; one rounding. Its epilogues are the MMA
  kernels' (bias, bias and GELU, SwiGLU of a column's gate and up rows).
- GELU of a rounded BF16 value ``x`` is DeepSeek's reference, ``F.gelu`` as torch
  evaluates it on BF16: ``(0.5 * x) * (1 + erff(x * M_SQRT1_2))`` in FP32, rounded
  to BF16. B12X evaluates the same formula with CuTe's ``erf``, whose last bits are
  not CUDA's ``erff``: 21 of 327680 aligner outputs differ from B12X's (DeepSeek's
  reference semantics over production's bits).
- ``rope_qkv``: B12X's 2-D split-half rotation of q and k, in place in the fused
  projection's output (v stays where it is; attention reads all three there). Pair
  ``j`` of a head (columns ``j`` and ``j + head_dim / 2``) rotates by
  ``position * inv_freq[j % (head_dim / 4)]``, the position being the patch's row
  for ``j < head_dim / 4`` and its column otherwise, with the accurate ``cosf`` and
  ``sinf`` of that FP32 product (a row's 32 angles are computed once, in shared
  memory). Each output is one FMA over the other term's rounded product, as nvcc
  and libNVVM contract B12X's ``q0 * cos + sign * q1 * sin`` (the left product is
  fused): ``first' = fma(first, cos, -(second * sin))``,
  ``second' = fma(second, cos, first * sin)``, each rounded to BF16.
- ``attention``: bidirectional softmax attention over the image's rows, one CTA a
  query tile and head: FP32 scores, an online softmax in base 2 (scale folded),
  BF16 probabilities into the value MMA, FP32 accumulation, one division at the
  end. Key columns past the live rows are masked after the score MMA, and loads
  past the live rows read zeros, so stale capacity rows (NaN included) never
  contribute. Not B12X's bits (its accumulation order is its own): gated against
  DeepSeek's FP64 softmax attention.
- ``add_norm``: B12X's residual add (``bf16(float(x) + float(branch))``) and its
  grouped RMSNorm (eps 1e-6, no zero-centering) of the rounded sum, in one launch.
  The reduction is B12X's to the rounding: 256 threads a row; thread ``t`` adds
  the squares of columns ``t, t + 256, ...`` in order with FMAs from zero; a
  butterfly over the warp's lanes (xor 1, 2, 4, 8, 16); the eight warp sums
  through shared memory into one warp's lanes 0-7 (zeros elsewhere) and the same
  butterfly; ``rsqrt(sum / width + eps)`` (``rsqrtf``, the MUFU approximation, as
  CuTe's ``rsqrt``); each output ``bf16((x * inverse) * (0 + weight))``.
  TileKernels' RMSNorm (``tilelang/norm.py``) sums in another order (128 threads,
  even/odd accumulators over 16-byte vectors), so it is not used here. Without
  ``add`` the kernel normalizes its input (the first block's first norm).
- ``spatial_merge``: the aligner's channel-major ``unfold`` of each
  ``ratio x ratio`` window (output column ``c * ratio^2 + ki * ratio + kj``), rows
  and columns past the image zero, staged through shared memory so both the reads
  and the writes are 16-byte vectors. Exact.

``*_call`` give each kernel as the ``prepare.call`` its owner declares and binds;
the tower (``nvidia/tilelang_vision.py``) launches nothing else.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
import tilelang
import tilelang.language as T

from vllm.models.deepseek_v4_1.tilelang import prepare

EPS = 1e-6  # B12X's (and DeepSeek's) vision RMSNorm epsilon
_LOG2E = 1.4426950408889634
_VEC = 8  # BF16 values in one 16-byte vector
_PIECE = 4  # BF16 values in one 8-byte piece (a patch row is 8-byte aligned)
_NORM_THREADS = 256  # B12X's grouped RMSNorm block
_K_ALIGN = 64  # the GEMMs' K tile: also each blocked FP32 partial sum's width
_GEMV_THREADS = 128  # B12X's SIMT projection block
# Images of at most this many rows (merged rows, for the aligner) project through
# the GEMV, B12X's SIMT projection: B12X plans it for at most eight rows.
GEMV_ROWS = 8
_EPILOGUES = ("none", "bias", "gelu", "swiglu")

# Elementwise and row-wise kernels index every access themselves.
_ELEMENTWISE = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}

# Tiles by kernel. A GEMM's tile changes its speed only (each output adds the same
# 64-wide K blocks in the same order, whatever the tile; K tiles stay 64); the
# attention's tiles change the grouping of its online softmax, so its bits (within
# the reference tolerance). A tile sweep of the port against B12X chose them.
LINEAR_TILE = dict(block_M=128, block_N=128, num_stages=2, threads=256)
SWIGLU_TILE = dict(block_M=128, block_N=64, num_stages=2, threads=256)
ATTENTION_TILE = dict(block_M=128, block_N=64, num_stages=2, threads=128)

_SQRT_HALF = 0.7071067811865476  # torch's M_SQRT1_2 for GELU, rounded to FP32


def padded_width(width: int) -> int:
    """``width`` rounded up to whole K tiles (the patch embedding's padded K)."""
    return -(-width // _K_ALIGN) * _K_ALIGN


def _zero_bf16():
    return T.cast(T.float32(0), T.bfloat16)


@tilelang.jit(pass_configs=_ELEMENTWISE)
def pad_columns_kernel(width: int, padded: int, threads: int = 128):
    """``dst[r] = [src[r] | 0 ...]`` for ``src`` rows of ``width`` BF16 (8-byte
    aligned: ``width`` is a multiple of four) into ``padded``-wide rows."""
    if width % _PIECE or padded % _VEC or padded < width:
        raise ValueError(f"cannot pad {width} columns to {padded}")
    pieces = padded // _PIECE
    rows = T.dynamic("rows")

    @T.prim_func
    def vision_pad_columns(
        src: T.Tensor((rows, width), T.bfloat16),
        dst: T.Tensor((rows, padded), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(pieces, threads), rows, threads=threads) as (bx, row):
            piece = bx * threads + T.get_thread_binding()
            values = T.alloc_local((_PIECE,), T.bfloat16)
            if piece < pieces:
                if piece * _PIECE < width:
                    for v in T.vectorized(_PIECE):
                        values[v] = src[row, piece * _PIECE + v]
                else:
                    for v in T.vectorized(_PIECE):
                        values[v] = _zero_bf16()
                for v in T.vectorized(_PIECE):
                    dst[row, piece * _PIECE + v] = values[v]

    return vision_pad_columns


def _gelu(value):
    """DeepSeek's reference GELU of a BF16 value, ``F.gelu`` as torch evaluates it
    on BF16: ``(0.5 * x) * (1 + erff(x * M_SQRT1_2))`` in FP32, rounded to BF16."""
    x = T.cast(value, T.float32)
    inner = T.erf(T.fmul(x, T.float32(_SQRT_HALF)))
    return T.cast(
        T.fmul(T.fmul(T.float32(0.5), x), T.ieee_add(T.float32(1), inner)),
        T.bfloat16,
    )


def _swiglu(gate, up):
    """B12X's vision SwiGLU of FP32 gate and up sums: each rounded to BF16, then
    ``bf16(bf16(g * (1 / (1 + __expf(-g)))) * u)``."""
    g = T.cast(T.cast(gate, T.bfloat16), T.float32)
    u = T.cast(T.cast(up, T.bfloat16), T.float32)
    sigmoid = T.ieee_fdiv(
        T.float32(1),
        T.ieee_add(T.float32(1), T.call_extern(T.float32, "__expf", -g)),
    )
    silu = T.cast(T.cast(T.fmul(g, sigmoid), T.bfloat16), T.float32)
    return T.cast(T.fmul(silu, u), T.bfloat16)


def _biased(total, bias):
    """The FP32 sum plus the BF16 bias, in FP32, rounded once to BF16."""
    return T.cast(T.ieee_add(total, T.cast(bias, T.float32)), T.bfloat16)


@tilelang.jit
def linear_kernel(
    N: int,
    K: int,
    bias: bool,
    gelu: bool = False,
    block_M: int = 128,
    block_N: int = 128,
    num_stages: int = 2,
    threads: int = 256,
):
    """``C[M, N] = bf16(A @ W^T (+ bias))`` for BF16 ``A[M, K]``, ``W[N, K]`` and
    ``bias[N]``: each 64-wide K block summed from zero on the tensor cores, the
    blocks added in FP32 in K order; with ``gelu``, ``bf16(gelu(C))``."""
    block_K = _K_ALIGN
    if N % block_N or K % block_K:
        raise ValueError(f"[{N}, {K}] is not whole {block_N} x {block_K} tiles")
    if gelu and not bias:
        raise ValueError("the GELU epilogue follows a biased projection")
    k_blocks = K // block_K
    num_stages = min(num_stages, k_blocks)  # a narrow K has fewer tiles than stages
    M = T.dynamic("M")

    @T.macro
    def body(A, W, Bias, C):
        with T.Kernel(N // block_N, T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            W_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
            acc = T.alloc_fragment((block_M, block_N), T.float32)
            part = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(acc)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(W[bx * block_N, ko * block_K], W_shared)
                T.gemm(A_shared, W_shared, part, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_M, block_N):
                    acc[i, j] = acc[i, j] + part[i, j]
            if bias:
                out = T.alloc_fragment((block_M, block_N), T.bfloat16)
                for i, j in T.Parallel(block_M, block_N):
                    biased = _biased(acc[i, j], Bias[bx * block_N + j])
                    if gelu:
                        out[i, j] = _gelu(biased)
                    else:
                        out[i, j] = biased
                T.copy(out, C[by * block_M, bx * block_N])
            else:
                T.copy(acc, C[by * block_M, bx * block_N])

    if bias:

        @T.prim_func
        def vision_linear(
            A: T.Tensor((M, K), T.bfloat16),
            W: T.Tensor((N, K), T.bfloat16),
            Bias: T.Tensor((N,), T.bfloat16),
            C: T.Tensor((M, N), T.bfloat16),
        ):
            body(A, W, Bias, C)

    else:

        @T.prim_func
        def vision_linear(
            A: T.Tensor((M, K), T.bfloat16),
            W: T.Tensor((N, K), T.bfloat16),
            C: T.Tensor((M, N), T.bfloat16),
        ):
            body(A, W, None, C)

    return vision_linear


@tilelang.jit
def swiglu_linear_kernel(
    inter: int,
    K: int,
    block_M: int = 128,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 256,
):
    """``Y[M, I] = swiglu(bf16(A @ W[:I]^T), bf16(A @ W[I:]^T))`` for BF16 ``A[M, K]``
    and the gate-then-up weight ``W[2I, K]``, with B12X's vision SwiGLU; both
    projections accumulate as ``linear_kernel``'s (64-wide blocks, added in FP32)."""
    block_K = _K_ALIGN
    if inter % block_N or K % block_K:
        raise ValueError(f"[{inter}, {K}] is not whole {block_N} x {block_K} tiles")
    k_blocks = K // block_K
    num_stages = min(num_stages, k_blocks)
    M = T.dynamic("M")

    @T.prim_func
    def vision_swiglu_linear(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((2 * inter, K), T.bfloat16),
        Y: T.Tensor((M, inter), T.bfloat16),
    ):
        with T.Kernel(inter // block_N, T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            G_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
            U_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
            gate = T.alloc_fragment((block_M, block_N), T.float32)
            up = T.alloc_fragment((block_M, block_N), T.float32)
            gate_part = T.alloc_fragment((block_M, block_N), T.float32)
            up_part = T.alloc_fragment((block_M, block_N), T.float32)
            out = T.alloc_fragment((block_M, block_N), T.bfloat16)
            T.clear(gate)
            T.clear(up)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(W[bx * block_N, ko * block_K], G_shared)
                T.copy(W[inter + bx * block_N, ko * block_K], U_shared)
                T.gemm(
                    A_shared, G_shared, gate_part, transpose_B=True, clear_accum=True
                )
                T.gemm(A_shared, U_shared, up_part, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_M, block_N):
                    gate[i, j] = gate[i, j] + gate_part[i, j]
                    up[i, j] = up[i, j] + up_part[i, j]
            for i, j in T.Parallel(block_M, block_N):
                out[i, j] = _swiglu(gate[i, j], up[i, j])
            T.copy(out, Y[by * block_M, bx * block_N])

    return vision_swiglu_linear


@tilelang.jit(pass_configs=_ELEMENTWISE)
def gemv_kernel(N: int, K: int, epilogue: str, rows_per_tile: int = GEMV_ROWS):
    """``C[M, :] = epilogue(A @ W^T)`` for BF16 ``A[M, K]`` and ``W[N, K]``, a few
    rows, with B12X's SIMT projection arithmetic (module docstring). ``epilogue``:
    ``none``; ``bias`` (``Bias[N]``); ``gelu`` (the bias, then GELU); ``swiglu``
    (``W`` stacks the gate's and the up's ``N / 2`` rows; ``C`` is ``N / 2`` wide)."""
    if epilogue not in _EPILOGUES:
        raise ValueError(f"unknown projection epilogue {epilogue!r}")
    swiglu = epilogue == "swiglu"
    if swiglu and N % 2:
        raise ValueError("a SwiGLU projection stacks two equal halves")
    parts = 2 if swiglu else 1
    columns = N // parts
    threads, warps = _GEMV_THREADS, _GEMV_THREADS // 32
    # B12X reads 8-wide vectors when K is whole vectors, else single elements.
    vector = K % _VEC == 0
    units = K // _VEC if vector else K
    sums = parts * rows_per_tile
    M = T.dynamic("M")

    @T.macro
    def body(A, W, Bias, C):
        with T.Kernel(columns, T.ceildiv(M, rows_per_tile), threads=threads) as (
            col,
            tile,
        ):
            tx = T.get_thread_binding()
            lane = tx % 32
            first = tile * rows_per_tile
            acc = T.alloc_local((sums,), T.float32)
            weights = T.alloc_local((parts * _VEC,), T.bfloat16)
            values = T.alloc_local((_VEC,), T.bfloat16)
            partial = T.alloc_shared((sums, warps), T.float32)
            total = T.alloc_var(T.float32)
            for i in T.unroll(sums):
                acc[i] = T.float32(0)
            for step in T.serial(-(-units // threads)):
                index = step * threads + tx
                if index < units:
                    if vector:
                        for p in T.unroll(parts):
                            for e in T.vectorized(_VEC):
                                weights[p * _VEC + e] = W[
                                    p * columns + col, index * _VEC + e
                                ]
                        for r in T.unroll(rows_per_tile):
                            if first + r < M:
                                for e in T.vectorized(_VEC):
                                    values[e] = A[first + r, index * _VEC + e]
                                for p in T.unroll(parts):
                                    for e in T.unroll(_VEC):
                                        acc[p * rows_per_tile + r] = T.fma(
                                            T.cast(weights[p * _VEC + e], T.float32),
                                            T.cast(values[e], T.float32),
                                            acc[p * rows_per_tile + r],
                                        )
                    else:
                        for p in T.unroll(parts):
                            weights[p] = W[p * columns + col, index]
                        for r in T.unroll(rows_per_tile):
                            if first + r < M:
                                for p in T.unroll(parts):
                                    acc[p * rows_per_tile + r] = T.fma(
                                        T.cast(A[first + r, index], T.float32),
                                        T.cast(weights[p], T.float32),
                                        acc[p * rows_per_tile + r],
                                    )
            # B12X's block reduction: each warp's butterfly, then the warp sums in
            # one warp's lanes (zeros past them) and the same butterfly.
            for i in T.unroll(sums):
                total = acc[i]
                for level in T.unroll(5):
                    total = total + T.shfl_xor(total, T.shift_left(1, level))
                if lane == 0:
                    partial[i, tx // 32] = total
            T.sync_threads()
            if tx < 32:
                for i in T.unroll(sums):
                    total = T.if_then_else(
                        lane < warps, partial[i, T.min(lane, warps - 1)], T.float32(0)
                    )
                    for level in T.unroll(5):
                        total = total + T.shfl_xor(total, T.shift_left(1, level))
                    acc[i] = total
                if tx == 0:
                    for r in T.unroll(rows_per_tile):
                        if first + r < M:
                            if swiglu:
                                C[first + r, col] = _swiglu(
                                    acc[r], acc[rows_per_tile + r]
                                )
                            elif epilogue == "none":
                                C[first + r, col] = T.cast(acc[r], T.bfloat16)
                            elif epilogue == "gelu":
                                C[first + r, col] = _gelu(_biased(acc[r], Bias[col]))
                            else:
                                C[first + r, col] = _biased(acc[r], Bias[col])

    if epilogue in ("bias", "gelu"):

        @T.prim_func
        def vision_gemv(
            A: T.Tensor((M, K), T.bfloat16),
            W: T.Tensor((N, K), T.bfloat16),
            Bias: T.Tensor((N,), T.bfloat16),
            C: T.Tensor((M, columns), T.bfloat16),
        ):
            body(A, W, Bias, C)

    else:

        @T.prim_func
        def vision_gemv(
            A: T.Tensor((M, K), T.bfloat16),
            W: T.Tensor((N, K), T.bfloat16),
            C: T.Tensor((M, columns), T.bfloat16),
        ):
            body(A, W, None, C)

    return vision_gemv


@tilelang.jit(pass_configs=_ELEMENTWISE)
def rope_qkv_kernel(heads: int, head_dim: int, threads: int = 128):
    """Rotate q and k of ``qkv`` ([M, 3 * heads * head_dim], q | k | v) in place for
    an image ``image_width`` patches wide; row ``r`` is patch
    ``(r // image_width, r % image_width)``. ``inv_freq`` is FP32
    ``[head_dim / 4]``."""
    half, quarter = head_dim // 2, head_dim // 4
    if half % _VEC or half > threads:
        raise ValueError(f"head_dim {head_dim} is not whole 16-column halves")
    width = heads * head_dim
    chunks = half // _VEC
    units = 2 * heads * chunks
    M = T.dynamic("M")

    @T.prim_func
    def vision_rope_qkv(
        qkv: T.Tensor((M, 3 * width), T.bfloat16),
        inv_freq: T.Tensor((quarter,), T.float32),
        image_width: T.int32,
    ):
        with T.Kernel(M, threads=threads) as row:
            tx = T.get_thread_binding()
            table = T.alloc_shared((2, half), T.float32)
            first = T.alloc_local((_VEC,), T.bfloat16)
            second = T.alloc_local((_VEC,), T.bfloat16)
            first_out = T.alloc_local((_VEC,), T.bfloat16)
            second_out = T.alloc_local((_VEC,), T.bfloat16)
            if tx < half:
                position = T.if_then_else(
                    tx < quarter, row // image_width, row % image_width
                )
                angle = T.fmul(T.cast(position, T.float32), inv_freq[tx % quarter])
                table[0, tx] = T.cos(angle)
                table[1, tx] = T.sin(angle)
            T.sync_threads()
            for step in T.unroll(-(-units // threads)):
                unit = step * threads + tx
                if unit < units:
                    part = unit // (heads * chunks)
                    head = unit // chunks % heads
                    chunk = unit % chunks
                    col = part * width + head * head_dim + chunk * _VEC
                    for v in T.vectorized(_VEC):
                        first[v] = qkv[row, col + v]
                    for v in T.vectorized(_VEC):
                        second[v] = qkv[row, col + half + v]
                    for v in T.unroll(_VEC):
                        x1 = T.cast(first[v], T.float32)
                        x2 = T.cast(second[v], T.float32)
                        cosine = table[0, chunk * _VEC + v]
                        sine = table[1, chunk * _VEC + v]
                        first_out[v] = T.cast(
                            T.fma(x1, cosine, -T.fmul(x2, sine)), T.bfloat16
                        )
                        second_out[v] = T.cast(
                            T.fma(x2, cosine, T.fmul(x1, sine)), T.bfloat16
                        )
                    for v in T.vectorized(_VEC):
                        qkv[row, col + v] = first_out[v]
                    for v in T.vectorized(_VEC):
                        qkv[row, col + half + v] = second_out[v]

    return vision_rope_qkv


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def attention_kernel(
    heads: int,
    head_dim: int,
    block_M: int = 128,
    block_N: int = 64,
    num_stages: int = 2,
    threads: int = 128,
):
    """``O[:, h] = softmax(Q_h K_h^T / sqrt(head_dim)) V_h`` over all ``M`` rows, for
    each head ``h``, with Q, K and V the three ``heads * head_dim`` column blocks of
    ``QKV`` ([M, 3 * heads * head_dim]); ``Out`` is [M, heads * head_dim]."""
    if head_dim % 16 or block_M % 16 or block_N % 16:
        raise ValueError("attention tiles are whole 16-row MMA atoms")
    width = heads * head_dim
    scale = head_dim**-0.5 * _LOG2E
    M = T.dynamic("M")
    accum = T.float32

    @T.prim_func
    def vision_attention(
        QKV: T.Tensor((M, 3 * width), T.bfloat16),
        Out: T.Tensor((M, width), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(M, block_M), heads, threads=threads) as (bx, h):
            Q_shared = T.alloc_shared((block_M, head_dim), T.bfloat16)
            K_shared = T.alloc_shared((block_N, head_dim), T.bfloat16)
            V_shared = T.alloc_shared((block_N, head_dim), T.bfloat16)
            O_shared = T.alloc_shared((block_M, head_dim), T.bfloat16)
            scores = T.alloc_fragment((block_M, block_N), accum)
            probs = T.alloc_fragment((block_M, block_N), T.bfloat16)
            acc = T.alloc_fragment((block_M, head_dim), accum)
            row_max = T.alloc_fragment((block_M,), accum)
            prev_max = T.alloc_fragment((block_M,), accum)
            rescale = T.alloc_fragment((block_M,), accum)
            row_sum = T.alloc_fragment((block_M,), accum)
            denom = T.alloc_fragment((block_M,), accum)

            T.copy(QKV[bx * block_M, h * head_dim], Q_shared)
            T.fill(acc, 0)
            T.fill(denom, 0)
            T.fill(row_max, -T.infinity(accum))
            for kb in T.Pipelined(T.ceildiv(M, block_N), num_stages=num_stages):
                T.copy(QKV[kb * block_N, width + h * head_dim], K_shared)
                T.gemm(
                    Q_shared,
                    K_shared,
                    scores,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                # Columns past the live rows are masked whatever their scores hold.
                for i, j in T.Parallel(block_M, block_N):
                    scores[i, j] = T.if_then_else(
                        kb * block_N + j < M, scores[i, j] * scale, -T.infinity(accum)
                    )
                T.copy(row_max, prev_max)
                T.reduce_max(scores, row_max, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    row_max[i] = T.max(row_max[i], prev_max[i])
                    rescale[i] = T.exp2(prev_max[i] - row_max[i])
                for i, j in T.Parallel(block_M, block_N):
                    scores[i, j] = T.exp2(scores[i, j] - row_max[i])
                T.reduce_sum(scores, row_sum, dim=1)
                for i in T.Parallel(block_M):
                    denom[i] = denom[i] * rescale[i] + row_sum[i]
                T.copy(scores, probs)
                for i, j in T.Parallel(block_M, head_dim):
                    acc[i, j] *= rescale[i]
                T.copy(QKV[kb * block_N, 2 * width + h * head_dim], V_shared)
                T.gemm(probs, V_shared, acc, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_M, head_dim):
                acc[i, j] /= denom[i]
            T.copy(acc, O_shared)
            T.copy(O_shared, Out[bx * block_M, h * head_dim])

    return vision_attention


@tilelang.jit(pass_configs=_ELEMENTWISE)
def add_norm_kernel(width: int, add: bool, eps: float = EPS):
    """With ``add``: ``Sum = bf16(X + Branch)`` and ``Out = rmsnorm(Sum) * Weight``;
    without: ``Out = rmsnorm(X) * Weight``. B12X's reduction order (module
    docstring); one 256-thread block a row."""
    threads = _NORM_THREADS
    warps = threads // 32
    items = -(-width // threads)
    M = T.dynamic("M")

    @T.macro
    def body(X, Branch, Weight, Sum, Out):
        with T.Kernel(M, threads=threads) as row:
            tx = T.get_thread_binding()
            lane = tx % 32
            values = T.alloc_local((items,), T.float32)
            partial = T.alloc_shared((warps,), T.float32)
            inverse = T.alloc_shared((1,), T.float32)
            total = T.alloc_var(T.float32)
            block = T.alloc_var(T.float32)
            total = T.float32(0)
            for item in T.unroll(items):
                col = item * threads + tx
                values[item] = T.float32(0)
                if col < width:
                    if add:
                        summed = T.cast(
                            T.ieee_add(
                                T.cast(X[row, col], T.float32),
                                T.cast(Branch[row, col], T.float32),
                            ),
                            T.bfloat16,
                        )
                        Sum[row, col] = summed
                        values[item] = T.cast(summed, T.float32)
                    else:
                        values[item] = T.cast(X[row, col], T.float32)
                total = T.fma(values[item], values[item], total)
            for level in T.unroll(5):
                total = total + T.shfl_xor(total, T.shift_left(1, level))
            if lane == 0:
                partial[tx // 32] = total
            T.sync_threads()
            block = T.if_then_else(
                lane < warps, partial[T.min(lane, warps - 1)], T.float32(0)
            )
            for level in T.unroll(5):
                block = block + T.shfl_xor(block, T.shift_left(1, level))
            if tx == 0:
                inverse[0] = T.rsqrt(
                    T.ieee_add(T.ieee_fdiv(block, T.float32(width)), T.float32(eps))
                )
            T.sync_threads()
            for item in T.unroll(items):
                col = item * threads + tx
                if col < width:
                    learned = T.ieee_add(T.float32(0), T.cast(Weight[col], T.float32))
                    Out[row, col] = T.cast(
                        T.fmul(T.fmul(values[item], inverse[0]), learned), T.bfloat16
                    )

    if add:

        @T.prim_func
        def vision_add_norm(
            X: T.Tensor((M, width), T.bfloat16),
            Branch: T.Tensor((M, width), T.bfloat16),
            Weight: T.Tensor((width,), T.bfloat16),
            Sum: T.Tensor((M, width), T.bfloat16),
            Out: T.Tensor((M, width), T.bfloat16),
        ):
            body(X, Branch, Weight, Sum, Out)

    else:

        @T.prim_func
        def vision_add_norm(
            X: T.Tensor((M, width), T.bfloat16),
            Weight: T.Tensor((width,), T.bfloat16),
            Out: T.Tensor((M, width), T.bfloat16),
        ):
            body(X, None, Weight, None, Out)

    return vision_add_norm


@tilelang.jit(pass_configs=_ELEMENTWISE)
def spatial_merge_kernel(channels: int, ratio: int, threads: int = 256):
    """``y[m, c * ratio^2 + ki * ratio + kj] = x[(i * ratio + ki) * width +
    (j * ratio + kj), c]`` for merged row ``m = i * ceil(width / ratio) + j``, zero
    past the ``height x width`` image (``x`` is [height * width, channels])."""
    if channels % _VEC:
        raise ValueError(f"{channels} channels are not whole 16-byte vectors")
    window = ratio * ratio
    vectors = channels // _VEC
    loads = window * vectors
    out_width = channels * window
    stores = out_width // _VEC
    # A window row padded by one vector spreads the gather over the banks.
    stride = channels + _VEC
    rows = T.dynamic("rows")
    merged = T.dynamic("merged")

    @T.prim_func
    def vision_spatial_merge(
        x: T.Tensor((rows, channels), T.bfloat16),
        y: T.Tensor((merged, out_width), T.bfloat16),
        height: T.int32,
        width: T.int32,
    ):
        with T.Kernel(merged, threads=threads) as row:
            tx = T.get_thread_binding()
            tile = T.alloc_shared((window, stride), T.bfloat16)
            values = T.alloc_local((_VEC,), T.bfloat16)
            gathered = T.alloc_local((_VEC,), T.bfloat16)
            merged_width = (width + ratio - 1) // ratio
            top = row // merged_width * ratio
            left = row % merged_width * ratio
            for step in T.unroll(-(-loads // threads)):
                e = step * threads + tx
                if e < loads:
                    k = e // vectors
                    c = e % vectors * _VEC
                    ih = top + k // ratio
                    iw = left + k % ratio
                    if (ih < height) & (iw < width):
                        for v in T.vectorized(_VEC):
                            values[v] = x[ih * width + iw, c + v]
                    else:
                        for v in T.vectorized(_VEC):
                            values[v] = _zero_bf16()
                    for v in T.vectorized(_VEC):
                        tile[k, c + v] = values[v]
            T.sync_threads()
            for step in T.unroll(-(-stores // threads)):
                e = step * threads + tx
                if e < stores:
                    for v in T.unroll(_VEC):
                        gathered[v] = tile[
                            (e * _VEC + v) % window, (e * _VEC + v) // window
                        ]
                    for v in T.vectorized(_VEC):
                        y[row, e * _VEC + v] = gathered[v]

    return vision_spatial_merge


def _tile(N: int, tile: dict, overrides: dict) -> dict:
    """``tile`` with ``overrides``, its block_N the widest of block_N, 64, 32 and 16
    that divides ``N`` (a narrow test projection)."""
    config = {**tile, **overrides}
    preferred = config["block_N"]
    for block_N in (preferred, 64, 32, 16):
        if block_N <= preferred and N % block_N == 0:
            config["block_N"] = block_N
            return config
    raise ValueError(f"{N} output columns are not a multiple of 16")


def pad_call(width: int) -> prepare.TileLangCall:
    return prepare.call(pad_columns_kernel, width, padded_width(width))


def linear_call(
    N: int, K: int, *, bias: bool, gelu: bool = False, **overrides
) -> prepare.TileLangCall:
    return prepare.call(
        linear_kernel, N, K, bias, gelu, **_tile(N, LINEAR_TILE, overrides)
    )


def swiglu_call(inter: int, K: int, **overrides) -> prepare.TileLangCall:
    return prepare.call(
        swiglu_linear_kernel, inter, K, **_tile(inter, SWIGLU_TILE, overrides)
    )


def gemv_call(N: int, K: int, epilogue: str) -> prepare.TileLangCall:
    """The GEMV of an ``[N, K]`` weight (``N`` stacks gate and up for ``swiglu``)."""
    return prepare.call(gemv_kernel, N, K, epilogue)


def rope_call(heads: int, head_dim: int) -> prepare.TileLangCall:
    return prepare.call(rope_qkv_kernel, heads, head_dim)


def attention_call(heads: int, head_dim: int, **overrides) -> prepare.TileLangCall:
    return prepare.call(
        attention_kernel, heads, head_dim, **{**ATTENTION_TILE, **overrides}
    )


def norm_call(width: int, *, add: bool) -> prepare.TileLangCall:
    return prepare.call(add_norm_kernel, width, add, EPS)


def merge_call(channels: int, ratio: int) -> prepare.TileLangCall:
    return prepare.call(spatial_merge_kernel, channels, ratio)


def resolve(call: prepare.TileLangCall):
    """The kernel of a declared call (after the kernel cache freezes, an undeclared
    call raises ``KernelNotPreparedError``)."""
    function, args, kwargs = call
    return function(*args, **kwargs)


__all__ = [
    "ATTENTION_TILE",
    "EPS",
    "GEMV_ROWS",
    "LINEAR_TILE",
    "SWIGLU_TILE",
    "add_norm_kernel",
    "attention_call",
    "attention_kernel",
    "gemv_call",
    "gemv_kernel",
    "linear_call",
    "linear_kernel",
    "merge_call",
    "norm_call",
    "pad_call",
    "pad_columns_kernel",
    "padded_width",
    "resolve",
    "rope_call",
    "rope_qkv_kernel",
    "spatial_merge_kernel",
    "swiglu_call",
    "swiglu_linear_kernel",
]
