#!/usr/bin/env python3
"""Build overlay mhc-seq: B12X mHC lagged partials with one sequential FP32 sum per output.

usage: make_overlay_mhcseq.py MHC_MT2_DIR   (writes ~/spark3-overlay/mhc-seq/b12x/norm/mhc/)

MHC_MT2_DIR is overlay mhc-mt2's norm/mhc directory (the r5o image's B12X plus patch 0008).

The lagged native partial kernels give each of a 128-wide hidden tile's 512 products to one
thread per hidden element: a four-term product sum per thread, a warp shuffle tree, then the
four warps summed in order, for each of 25 partials and every token. The reductions, not the
arithmetic, set its speed: at prefill sizes it is 1.4x the TF32 projection.

MHCPostPreSequentialPartialKernel keeps the same inputs and outputs (residual_out, y and
partials[token, tile, j], so finalize is unchanged) but sums each partial in one fixed order:
for each residual stream, one thread adds the tile's 128 products in hidden order, and the four
stream sums are then added in stream order. A CTA covers one hidden tile for tokens_per_cta
tokens (a power of two). Warps 0-2 hold one (mix, stream) pair per lane and stream that pair's
weights from global memory with vector loads; each hidden element's tokens come from shared
memory as one vector, so each weight serves every token. Warp 3 sums the squares. A row's sums
depend only on its own data, never on how many rows share a CTA or a launch, so every capacity
can use it.
MhcConfig gains sequential_partials (default False; config schema version 6).
"""
import os
import sys

SRC = sys.argv[1]
OUT = os.environ.get("MHCSEQ_OUT", os.path.expanduser("~/spark3-overlay/mhc-seq/b12x/norm/mhc"))
os.makedirs(OUT, exist_ok=True)

k = open(os.path.join(SRC, "_kernels.py")).read()

# 1. Storage and kernel, after the multi-token kernel class.
anchor = "class MHCPostPreDecodeSplitNPartialKernel:"
assert k.count(anchor) == 1
kernel_code = '''from b12x._lib.intrinsics import (
    ld_global_nc_v4_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    st_shared_v4_u32,
    u32_as_f32,
)

# Per token: prev_post (4), prev_comb (16, [from][to]) and pre_mix (4).
_SEQ_PARAMS = 24
# Mixing-weight vectors (4 hidden elements each) a mix thread keeps in flight: with few tokens
# each vector is consumed quickly, so the whole row (32 vectors) is requested at once.
def _sequential_fn_prefetch(tokens_per_cta: int) -> int:
    return 32 if tokens_per_cta <= 2 else 16 if tokens_per_cta == 4 else 8


@dsl_user_op
def _ld_shared_f32(smem_addr: Int32, *, loc=None, ip=None) -> Float32:
    """Load one FP32 value from a u32 shared-memory address."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _sequential_tokens(tokens_per_cta: int) -> int:
    """tokens_per_cta rounded up to a power of two, so a thread's tokens fill whole vectors."""
    tokens = 1
    while tokens < int(tokens_per_cta):
        tokens *= 2
    return tokens


@lru_cache(maxsize=16)
def _post_pre_sequential_storage_cls(tokens_per_cta: int, lagged_mix: bool):
    class PostPreSequentialStorage:
        pass

    annotations = {
        # Post-mixed residual in FP32, [stream][hidden][token]. A thread's tokens are
        # contiguous (vector loads); each stream's block carries 16 bytes of padding so the
        # four streams a warp reads at once start in different banks.
        "r_tile": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Float32, _MHC_MULT * (_SOURCE_TILE_H * tokens_per_cta + 4)
            ],
            16,
        ],
        # Per-stream sums, [token][item][stream], combined in stream order.
        "stream_sums": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, tokens_per_cta * (_PARTIALS + 1) * _MHC_MULT], 16
        ],
        # Each token's mixing scalars.
        "params": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, tokens_per_cta * _SEQ_PARAMS], 16
        ],
    }
    if lagged_mix:
        annotations["y_tile"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, _SOURCE_TILE_H * tokens_per_cta], 16
        ]
    PostPreSequentialStorage.__annotations__ = annotations
    return cute.struct(PostPreSequentialStorage)


class MHCPostPreSequentialPartialKernel(MHCPostPrePartialKernel):
    \"\"\"Lagged partials summed in one fixed order per row, whatever shares the launch.

    Inputs and outputs are MHCPostPrePartialKernel's (residual_out, y, partials[token, tile,
    j] and the lagged y-squared sums), so the finalize kernel is unchanged. For each item of a
    row (the residual's sum of squares, 24 mixes, the lagged y sum of squares) and each
    residual stream, one thread sums the tile's 128 products in hidden order; the four stream
    sums are then added in stream order.

    A CTA covers one hidden tile for tokens_per_cta tokens (rounded up to a power of two).
    Warps 0-2 hold one (mix, stream) pair per lane: the thread streams its own 128 mixing
    weights from global memory with vector loads, a few vectors ahead, and reads the CTA's
    tokens for each hidden element as one shared-memory vector, so each weight serves every
    token. Warp 3 sums the squares (lanes 0-3, one stream each; lane 4 the lagged y). A row's
    sums depend only on its own data, never on how many rows share a CTA or a launch.
    \"\"\"

    items_per_token = _PARTIALS + 1  # 25 partials and the lagged y-squared sum

    def __init__(self, *, tokens_per_cta: int, **kwargs):
        super().__init__(**kwargs)
        tokens_per_cta = int(tokens_per_cta)
        if not 1 <= tokens_per_cta <= 16:
            raise ValueError(f"tokens_per_cta must be in [1, 16], got {tokens_per_cta}")
        if self.post_only or self.compute_gram:
            raise ValueError("the sequential partial kernel computes lagged partial sums only")
        if self.source_tile_h != self.num_threads:
            raise ValueError("the sequential partial kernel needs one thread per hidden element")
        if (_MIXES + 1) * _MHC_MULT > self.num_threads or self.num_threads % 32:
            raise ValueError("mix threads and a squares warp must fit the CTA")
        self.tokens_per_cta = _sequential_tokens(tokens_per_cta)
        self.combine_per_thread = (
            self.tokens_per_cta * self.items_per_token + self.num_threads - 1
        ) // self.num_threads

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
        pre_mix: cute.Tensor,
        y: cute.Tensor,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr((not self.pre_only) and x.element_type != cutlass.BFloat16):
            raise TypeError("x must be BFloat16")
        if const_expr(residual.element_type != cutlass.BFloat16):
            raise TypeError("residual must be BFloat16")
        if const_expr((not self.pre_only) and prev_post.element_type != cutlass.Float32):
            raise TypeError("prev_post must be Float32")
        if const_expr((not self.pre_only) and prev_comb.element_type != cutlass.Float32):
            raise TypeError("prev_comb must be Float32")
        if const_expr(fn.element_type != cutlass.Float32):
            raise TypeError("fn must be Float32")
        if const_expr(partials.element_type != cutlass.Float32):
            raise TypeError("partials must be Float32")
        if const_expr(out.element_type != cutlass.BFloat16):
            raise TypeError("out must be BFloat16")
        if const_expr(self.lagged_mix and pre_mix.element_type != cutlass.Float32):
            raise TypeError("pre_mix must be Float32")
        if const_expr(self.lagged_mix and y.element_type != cutlass.BFloat16):
            raise TypeError("y must be BFloat16")
        self.kernel(
            x, residual, prev_post, prev_comb, fn, partials, out, pre_mix, y, num_tokens,
        ).launch(
            grid=(
                self.source_tiles,
                1,
                (num_tokens + Int32(self.tokens_per_cta - 1)) // Int32(self.tokens_per_cta),
            ),
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=_MHC_PDL,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        fn: cute.Tensor,
        partials: cute.Tensor,
        out: cute.Tensor,
        pre_mix: cute.Tensor,
        y: cute.Tensor,
        num_tokens: Int32,
    ):
        hidden_tile, _, token_block = cute.arch.block_idx()
        tidx = cute.arch.thread_idx()[0]
        T = self.tokens_per_cta
        H = self.source_tile_h
        RS = H * T + 4  # r_tile stream stride, in floats
        PF = _sequential_fn_prefetch(T)
        MIX_THREADS = _MIXES * _MHC_MULT
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(_post_pre_sequential_storage_cls(T, self.lagged_mix))
        r_tile = storage.r_tile.get_tensor(
            cute.make_layout((_MHC_MULT, H, T), stride=(RS, T, 1))
        )
        r_addr = shared_ptr_to_u32(storage.r_tile.data_ptr())
        stream_sums = storage.stream_sums.get_tensor(
            cute.make_layout(
                (T, self.items_per_token, _MHC_MULT),
                stride=(self.items_per_token * _MHC_MULT, _MHC_MULT, 1),
            )
        )
        params = storage.params.get_tensor(
            cute.make_layout((T, _SEQ_PARAMS), stride=(_SEQ_PARAMS, 1))
        )
        y_addr = r_addr
        if const_expr(self.lagged_mix):
            y_tile = storage.y_tile.get_tensor(cute.make_layout((H, T), stride=(T, 1)))
            y_addr = shared_ptr_to_u32(storage.y_tile.data_ptr())
        broadcast = const_expr(self.pre_only and cute.rank(residual) == 2)

        # A mix thread's weights are constant. With one or two rows per CTA, latency rules: the
        # first vectors are requested before the wait on the previous kernel. With more rows
        # they are requested after the post-mix, so they do not hold registers through the
        # residual loads (occupancy).
        mix = tidx // Int32(_MHC_MULT)
        s = tidx - mix * Int32(_MHC_MULT)
        fn_addr = get_ptr_as_int64(
            fn,
            cute.crd2idx(
                (mix, Int64(s) * Int64(self.hidden_size) + Int64(hidden_tile) * Int64(H)),
                fn.layout,
            ),
        )
        wbuf = cute.make_rmem_tensor(
            cute.make_layout((4 * PF,), stride=(1,)), Float32
        )
        if const_expr(T <= 2):
            if tidx < Int32(MIX_THREADS):
                for v in cutlass.range_constexpr(PF):
                    a0, a1, a2, a3 = ld_global_nc_v4_u32(fn_addr + Int64(16 * v))
                    wbuf[4 * v + 0] = u32_as_f32(a0)
                    wbuf[4 * v + 1] = u32_as_f32(a1)
                    wbuf[4 * v + 2] = u32_as_f32(a2)
                    wbuf[4 * v + 3] = u32_as_f32(a3)
        if const_expr(_MHC_PDL):
            cute.arch.griddepcontrol_wait()

        h = Int64(hidden_tile) * Int64(H) + Int64(tidx)
        token0 = Int64(token_block) * Int64(T)
        last = Int64(num_tokens) - Int64(1)
        valid = Int64(num_tokens) - token0
        if valid > Int64(T):
            valid = Int64(T)

        # Every token's residual (and x) first, so all of the CTA's loads are in flight at once.
        # Rows past the end read the last row; their sums are never written.
        q = cute.make_rmem_tensor(cute.make_layout((T, _MHC_MULT), stride=(_MHC_MULT, 1)), Float32)
        xs = cute.make_rmem_tensor(cute.make_layout((T,), stride=(1,)), Float32)
        for t in cutlass.range_constexpr(T):
            token = token0 + Int64(t)
            if token > last:
                token = last
            if const_expr(broadcast):
                q[t, 0] = Float32(residual[token, h])
            else:
                for st in cutlass.range_constexpr(_MHC_MULT):
                    q[t, st] = Float32(residual[token, Int32(st), h])
            if const_expr(not self.pre_only):
                xs[t] = Float32(x[token, h])
        # The tokens' mixing scalars, shared through shared memory.
        for k in cutlass.range_constexpr(
            (T * _SEQ_PARAMS + self.num_threads - 1) // self.num_threads
        ):
            idx = Int32(k * self.num_threads) + tidx
            if idx < Int32(T * _SEQ_PARAMS):
                t = idx // Int32(_SEQ_PARAMS)
                j = idx - t * Int32(_SEQ_PARAMS)
                token = token0 + Int64(t)
                if token > last:
                    token = last
                value = Float32(0.0)
                if j < Int32(4):
                    if const_expr(not self.pre_only):
                        value = Float32(prev_post[token, j])
                elif j < Int32(20):
                    if const_expr(not self.pre_only):
                        value = Float32(
                            prev_comb[token, (j - Int32(4)) // Int32(4), (j - Int32(4)) % Int32(4)]
                        )
                else:
                    if const_expr(self.lagged_mix):
                        value = Float32(pre_mix[token, j - Int32(20)])
                params[t, j] = value
        cute.arch.sync_threads()

        # Post-mix (or plain residual for pre), exactly as the warp-reduced kernels compute and
        # store it.
        r = cute.make_rmem_tensor(cute.make_layout((_MHC_MULT, T), stride=(T, 1)), Float32)
        yv = cute.make_rmem_tensor(cute.make_layout((T,), stride=(1,)), Float32)
        for t in cutlass.range_constexpr(T):
            token = token0 + Int64(t)
            if const_expr(self.pre_only):
                for st in cutlass.range_constexpr(_MHC_MULT):
                    if const_expr(broadcast):
                        r[st, t] = q[t, 0]
                    else:
                        r[st, t] = q[t, st]
                if token < Int64(num_tokens):
                    for st in cutlass.range_constexpr(_MHC_MULT):
                        out[token, Int32(st), h] = r[st, t].to(cutlass.BFloat16)
            else:
                for st in cutlass.range_constexpr(_MHC_MULT):
                    o = (
                        params[t, st] * xs[t]
                        + params[t, 4 + st] * q[t, 0]
                        + params[t, 8 + st] * q[t, 1]
                        + params[t, 12 + st] * q[t, 2]
                        + params[t, 16 + st] * q[t, 3]
                    ).to(cutlass.BFloat16)
                    if token < Int64(num_tokens):
                        out[token, Int32(st), h] = o
                    r[st, t] = Float32(o)
            if const_expr(self.lagged_mix):
                y_bf16 = _contract_four(
                    params[t, 20], params[t, 21], params[t, 22], params[t, 23],
                    r[0, t], r[1, t], r[2, t], r[3, t],
                ).to(cutlass.BFloat16)
                if token < Int64(num_tokens):
                    y[token, h] = y_bf16
                yv[t] = Float32(y_bf16)
        if const_expr(T > 2):
            if tidx < Int32(MIX_THREADS):
                for v in cutlass.range_constexpr(PF):
                    a0, a1, a2, a3 = ld_global_nc_v4_u32(fn_addr + Int64(16 * v))
                    wbuf[4 * v + 0] = u32_as_f32(a0)
                    wbuf[4 * v + 1] = u32_as_f32(a1)
                    wbuf[4 * v + 2] = u32_as_f32(a2)
                    wbuf[4 * v + 3] = u32_as_f32(a3)
        for st in cutlass.range_constexpr(_MHC_MULT):
            if const_expr(T % 4 == 0):
                for c in cutlass.range_constexpr(T // 4):
                    st_shared_v4_u32(
                        r_addr + (Int32(st * RS + 4 * c) + tidx * Int32(T)) * Int32(4),
                        f32_to_raw_bits(r[st, 4 * c + 0]),
                        f32_to_raw_bits(r[st, 4 * c + 1]),
                        f32_to_raw_bits(r[st, 4 * c + 2]),
                        f32_to_raw_bits(r[st, 4 * c + 3]),
                    )
            else:
                for t in cutlass.range_constexpr(T):
                    r_tile[st, tidx, t] = r[st, t]
        if const_expr(self.lagged_mix):
            if const_expr(T % 4 == 0):
                for c in cutlass.range_constexpr(T // 4):
                    st_shared_v4_u32(
                        y_addr + (Int32(4 * c) + tidx * Int32(T)) * Int32(4),
                        f32_to_raw_bits(yv[4 * c + 0]),
                        f32_to_raw_bits(yv[4 * c + 1]),
                        f32_to_raw_bits(yv[4 * c + 2]),
                        f32_to_raw_bits(yv[4 * c + 3]),
                    )
            else:
                for t in cutlass.range_constexpr(T):
                    y_tile[tidx, t] = yv[t]
        cute.arch.sync_threads()

        acc = cute.make_rmem_tensor(cute.make_layout((T,), stride=(1,)), Float32)
        for t in cutlass.range_constexpr(T):
            acc[t] = Float32(0.0)
        v = cute.make_rmem_tensor(cute.make_layout((T,), stride=(1,)), Float32)
        if tidx < Int32(MIX_THREADS):
            # Warps 0-2: one (mix, stream) pair per lane, weights a few vectors ahead.
            row_addr = r_addr + s * Int32(RS * 4)
            for e4 in cutlass.range_constexpr(H // 4):
                slot = e4 % PF
                w0 = wbuf[4 * slot + 0]
                w1 = wbuf[4 * slot + 1]
                w2 = wbuf[4 * slot + 2]
                w3 = wbuf[4 * slot + 3]
                if const_expr(e4 + PF < H // 4):
                    a0, a1, a2, a3 = ld_global_nc_v4_u32(fn_addr + Int64(16 * (e4 + PF)))
                    wbuf[4 * slot + 0] = u32_as_f32(a0)
                    wbuf[4 * slot + 1] = u32_as_f32(a1)
                    wbuf[4 * slot + 2] = u32_as_f32(a2)
                    wbuf[4 * slot + 3] = u32_as_f32(a3)
                for i in cutlass.range_constexpr(4):
                    e = 4 * e4 + i
                    if const_expr(T % 4 == 0):
                        for c in cutlass.range_constexpr(T // 4):
                            b0, b1, b2, b3 = ld_shared_v4_u32(
                                row_addr + Int32((e * T + 4 * c) * 4)
                            )
                            v[4 * c + 0] = u32_as_f32(b0)
                            v[4 * c + 1] = u32_as_f32(b1)
                            v[4 * c + 2] = u32_as_f32(b2)
                            v[4 * c + 3] = u32_as_f32(b3)
                    elif const_expr(T == 2):
                        b0, b1 = ld_shared_v2_u32(row_addr + Int32(e * T * 4))
                        v[0] = u32_as_f32(b0)
                        v[1] = u32_as_f32(b1)
                    else:
                        v[0] = _ld_shared_f32(row_addr + Int32(e * 4))
                    w = w0
                    if const_expr(i == 1):
                        w = w1
                    if const_expr(i == 2):
                        w = w2
                    if const_expr(i == 3):
                        w = w3
                    for t in cutlass.range_constexpr(T):
                        acc[t] = acc[t] + w * v[t]
            for t in cutlass.range_constexpr(T):
                stream_sums[t, mix + Int32(1), s] = acc[t]
        else:
            # Warp 3: sums of squares, lanes 0-3 the residual streams, lane 4 the lagged y.
            lane = tidx - Int32(MIX_THREADS)
            row_addr = r_addr
            if lane < Int32(_MHC_MULT):
                row_addr = r_addr + lane * Int32(RS * 4)
            if const_expr(self.lagged_mix):
                if lane == Int32(_MHC_MULT):
                    row_addr = y_addr
            for e in cutlass.range_constexpr(H):
                if const_expr(T % 4 == 0):
                    for c in cutlass.range_constexpr(T // 4):
                        b0, b1, b2, b3 = ld_shared_v4_u32(row_addr + Int32((e * T + 4 * c) * 4))
                        v[4 * c + 0] = u32_as_f32(b0)
                        v[4 * c + 1] = u32_as_f32(b1)
                        v[4 * c + 2] = u32_as_f32(b2)
                        v[4 * c + 3] = u32_as_f32(b3)
                elif const_expr(T == 2):
                    b0, b1 = ld_shared_v2_u32(row_addr + Int32(e * T * 4))
                    v[0] = u32_as_f32(b0)
                    v[1] = u32_as_f32(b1)
                else:
                    v[0] = _ld_shared_f32(row_addr + Int32(e * 4))
                for t in cutlass.range_constexpr(T):
                    acc[t] = acc[t] + v[t] * v[t]
            if lane < Int32(_MHC_MULT):
                for t in cutlass.range_constexpr(T):
                    stream_sums[t, Int32(0), lane] = acc[t]
            if const_expr(self.lagged_mix):
                if lane == Int32(_MHC_MULT):
                    for t in cutlass.range_constexpr(T):
                        stream_sums[t, Int32(_PARTIALS), Int32(0)] = acc[t]
        cute.arch.sync_threads()

        # Stream sums added in stream order, one (token, item) per thread and slot.
        for slot in cutlass.range_constexpr(self.combine_per_thread):
            unit = Int32(slot * self.num_threads) + tidx
            t = unit // Int32(self.items_per_token)
            j = unit - t * Int32(self.items_per_token)
            if Int64(t) < valid:
                token = token0 + Int64(t)
                if j < Int32(_PARTIALS):
                    total = stream_sums[t, j, Int32(0)]
                    single = False
                    if const_expr(broadcast):
                        single = j > Int32(0)
                    if not single:
                        total = total + stream_sums[t, j, Int32(1)]
                        total = total + stream_sums[t, j, Int32(2)]
                        total = total + stream_sums[t, j, Int32(3)]
                    partials[token, hidden_tile, j] = total
                elif const_expr(self.lagged_mix):
                    partials[token, Int32(self.gram_row0) + hidden_tile, Int32(0)] = stream_sums[
                        t, j, Int32(0)
                    ]

        if const_expr(_MHC_PDL):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()


'''
k = k.replace(anchor, kernel_code + anchor)

# 2. Factory.
old = """    tokens_per_cta: int = 1,
) -> MHCPostPrePartialKernel:
    kwargs = dict("""
assert k.count(old) == 1, k.count(old)
k = k.replace(old, """    tokens_per_cta: int = 1,
    sequential: bool = False,
) -> MHCPostPrePartialKernel:
    kwargs = dict(""")
old = """    if int(tokens_per_cta) > 1:
        return MHCPostPrePartialMultiTokenKernel(tokens_per_cta=int(tokens_per_cta), **kwargs)
    return MHCPostPrePartialKernel(**kwargs)"""
assert k.count(old) == 1
k = k.replace(old, """    if sequential:
        return MHCPostPreSequentialPartialKernel(tokens_per_cta=int(tokens_per_cta), **kwargs)
    if int(tokens_per_cta) > 1:
        return MHCPostPrePartialMultiTokenKernel(tokens_per_cta=int(tokens_per_cta), **kwargs)
    return MHCPostPrePartialKernel(**kwargs)""")

# 3. post_pre launch: the sequential flag from the native launch record.
old = "    tokens_per_cta = int(getattr(native, \"tokens_per_cta\", 1))\n"
assert k.count(old) == 1
k = k.replace(old, old + "    sequential = bool(getattr(native, \"sequential\", False))\n")
for spec in ('"integration.residual.mhc_post_pre_partial_hidden4096_hctile128"\n            f"_all{partials_per_cta}"\n            + (f"_t{tokens_per_cta}" if tokens_per_cta > 1 else "")',
             '"integration.residual.mhc_post_pre_partial_"\n            f"{hidden_specialization}_hctile128_all{partials_per_cta}"\n            + (f"_t{tokens_per_cta}" if tokens_per_cta > 1 else "")'):
    assert spec in k, spec
    k = k.replace(spec, spec + '\n            + ("_seq" if sequential else "")')
old = """            partials_per_cta,
            tokens_per_cta,
        )"""
assert k.count(old) == 2, k.count(old)
k = k.replace(old, """            partials_per_cta,
            tokens_per_cta,
            sequential,
        )""")

# 4. pre launch: an explicit sequential argument.
old = "    tokens_per_cta: int = 1,\n    _prepared=None,\n) -> None:\n    tokens_per_cta = int(tokens_per_cta)\n"
assert k.count(old) == 1
k = k.replace(old, "    tokens_per_cta: int = 1,\n    sequential: bool = False,\n    _prepared=None,\n) -> None:\n    tokens_per_cta = int(tokens_per_cta)\n    sequential = bool(sequential)\n")
for spec in ('"integration.residual.mhc_pre_partial_hidden4096_hctile128"\n            f"_all{partials_per_cta}"\n            + (f"_t{tokens_per_cta}" if tokens_per_cta > 1 else "")',
             '"integration.residual.mhc_pre_partial_"\n            f"{hidden_specialization}_hctile128_all{partials_per_cta}"\n            + (f"_t{tokens_per_cta}" if tokens_per_cta > 1 else "")'):
    assert spec in k, spec
    k = k.replace(spec, spec + '\n            + ("_seq" if sequential else "")')
old = """            partials_per_cta=partials_per_cta,
            tokens_per_cta=tokens_per_cta,
        ),"""
assert k.count(old) == 1, k.count(old)
k = k.replace(old, """            partials_per_cta=partials_per_cta,
            tokens_per_cta=tokens_per_cta,
            sequential=sequential,
        ),""")
# compile keys gain the sequential flag only when it is set.
before = k.count('*((("tokens_per_cta", tokens_per_cta),) if tokens_per_cta > 1 else ()),\n')
assert before >= 2, before
k = k.replace('*((("tokens_per_cta", tokens_per_cta),) if tokens_per_cta > 1 else ()),\n',
              '*((("tokens_per_cta", tokens_per_cta),) if tokens_per_cta > 1 else ()),\n'
              '            *((("sequential", True),) if sequential else ()),\n')
# 5. Finalize: the lagged-prepared path can give a row's whole hidden size to one narrower CTA.
# The default (1024 threads, one CTA per 1024-wide tile) repeats the row's scalar finalize --
# 25 partial-sum reductions and the Sinkhorn iterations, on one thread -- in each of the row's
# five CTAs, and one 1024-thread CTA fills an SM, so the SM idles through that serial part.
# lagged_threads=256 runs it once per row, with six CTAs per SM; each element's arithmetic is
# unchanged.
old = """        lagged_mix: bool = False,
        lagged_prepared: bool = False,
    ):
        self.lagged_prepared = bool(lagged_prepared)
"""
assert k.count(old) == 1
k = k.replace(old, """        lagged_mix: bool = False,
        lagged_prepared: bool = False,
        lagged_threads: int = _GRAM_BLOCK_H,
    ):
        self.lagged_prepared = bool(lagged_prepared)
        self.lagged_threads = int(lagged_threads)
        if self.lagged_threads != _GRAM_BLOCK_H and not self.lagged_prepared:
            raise ValueError("lagged_threads applies to the prepared lagged finalize only")
""")
old = """        self.num_threads = (
            _GRAM_BLOCK_H
            if self.lagged_prepared
"""
assert k.count(old) == 1
k = k.replace(old, """        self.num_threads = (
            self.lagged_threads
            if self.lagged_prepared
""")
old = """        self.tiles_per_cta = (
            1
            if self.lagged_prepared and self.lagged_norm
"""
assert k.count(old) == 1
k = k.replace(old, """        self.tiles_per_cta = (
            (1 if self.lagged_threads == _GRAM_BLOCK_H else self.hidden_tiles)
            if self.lagged_prepared and self.lagged_norm
""")
old = """                if const_expr(self.lagged_norm):
                    h = Int64(tile_group) * Int64(self.num_threads) + Int64(tidx)
                    value = Float32(y[token, h])
                    value = value * Float32(s_post[0]) * Float32(norm_weight[h])
                    y[token, h] = value.to(cutlass.BFloat16)
"""
assert k.count(old) == 1
k = k.replace(old, """                if const_expr(self.lagged_norm):
                    # Every load before any store: y is read and written in place, so
                    # interleaved stores would order each load behind the previous store.
                    y_values = cute.make_rmem_tensor(
                        cute.make_layout((self.tiles_per_cta,), stride=(1,)), Float32
                    )
                    weights = cute.make_rmem_tensor(
                        cute.make_layout((self.tiles_per_cta,), stride=(1,)), Float32
                    )
                    for tile in cutlass.range_constexpr(self.tiles_per_cta):
                        h = (
                            Int64(tile_group) * Int64(self.tiles_per_cta) + Int64(tile)
                        ) * Int64(self.num_threads) + Int64(tidx)
                        y_values[tile] = Float32(y[token, h])
                        weights[tile] = Float32(norm_weight[h])
                    for tile in cutlass.range_constexpr(self.tiles_per_cta):
                        h = (
                            Int64(tile_group) * Int64(self.tiles_per_cta) + Int64(tile)
                        ) * Int64(self.num_threads) + Int64(tidx)
                        value = y_values[tile] * Float32(s_post[0]) * weights[tile]
                        y[token, h] = value.to(cutlass.BFloat16)
""")
old = """    lagged_mix: bool = False,
    lagged_prepared: bool = False,
) -> MHCFinalizeGramKernel:
    return MHCFinalizeGramKernel("""
assert k.count(old) == 1
k = k.replace(old, """    lagged_mix: bool = False,
    lagged_prepared: bool = False,
    lagged_threads: int = _GRAM_BLOCK_H,
) -> MHCFinalizeGramKernel:
    return MHCFinalizeGramKernel(""")
old = """        lagged_mix=lagged_mix,
        lagged_prepared=lagged_prepared,
    )
"""
assert k.count(old) == 1
k = k.replace(old, """        lagged_mix=lagged_mix,
        lagged_prepared=lagged_prepared,
        lagged_threads=lagged_threads,
    )
""")
old = """    lagged_prepared: bool = False,
    single_cta_threads: int = 0,
    single_cta_groups: int = 1,
    _prepared=None,
) -> None:
"""
assert k.count(old) == 1
k = k.replace(old, """    lagged_prepared: bool = False,
    single_cta_threads: int = 0,
    single_cta_groups: int = 1,
    lagged_threads: int = _GRAM_BLOCK_H,
    _prepared=None,
) -> None:
    lagged_threads = int(lagged_threads)
""")
old = """        single_cta_groups, active_source_splits, lagged_mix, bool(lagged_prepared),
    )
"""
assert k.count(old) == 1
k = k.replace(old, """        single_cta_groups, active_source_splits, lagged_mix, bool(lagged_prepared),
        lagged_threads,
    )
""")
old = """        ("lagged_prepared", bool(lagged_prepared)),
        ("compact_partials", compact_partials),
"""
assert k.count(old) == 1
k = k.replace(old, """        ("lagged_prepared", bool(lagged_prepared)),
        *((("lagged_threads", lagged_threads),) if lagged_threads != _GRAM_BLOCK_H else ()),
        ("compact_partials", compact_partials),
""")

compile(k, "_kernels.py", "exec")
open(os.path.join(OUT, "_kernels.py"), "w").write(k)

# 6. Config and preparation.
t = open(os.path.join(SRC, "_tuning.py")).read()
old = "    tokens_per_cta: int = 1\n"
assert t.count(old) == 1
t = t.replace(old, old + "    sequential_partials: bool = False\n")
old = """        if frozenset(payload) | {"tokens_per_cta"} != expected:"""
assert t.count(old) == 1
t = t.replace(old, """        if frozenset(payload) | {"tokens_per_cta", "sequential_partials"} != expected:""")
old = """            tokens_per_cta=int(payload.get("tokens_per_cta", 1)),
        )"""
assert t.count(old) == 1
t = t.replace(old, """            tokens_per_cta=int(payload.get("tokens_per_cta", 1)),
            sequential_partials=bool(payload.get("sequential_partials", False)),
        )""")
old = """            "tokens_per_cta": self.tokens_per_cta,
        }"""
assert t.count(old) == 1
t = t.replace(old, """            "tokens_per_cta": self.tokens_per_cta,
            "sequential_partials": self.sequential_partials,
        }""")
old = "    config_schema_version=5,\n"
assert t.count(old) == 1
t = t.replace(old, "    config_schema_version=6,\n")
# The integer check over the geometry fields skips the new boolean, as it skips lagged_prepare.
old = 'if field not in ("backend", "lagged_prepare")):'
assert t.count(old) == 1
t = t.replace(old, 'if field not in ("backend", "lagged_prepare", "sequential_partials")):')
old = """    if config.tokens_per_cta != 1 and not config.lagged_prepare:
        raise ValueError("several tokens per CTA require the fused lagged producer")"""
assert t.count(old) == 1
t = t.replace(old, old + """
    if type(config.sequential_partials) is not bool:
        raise TypeError("sequential_partials must be a boolean")
    if config.sequential_partials and not config.lagged_prepare:
        raise ValueError("sequential partials require the fused lagged producer")""")
compile(t, "_tuning.py", "exec")
open(os.path.join(OUT, "_tuning.py"), "w").write(t)

p = open(os.path.join(SRC, "_preparation.py")).read()
old = "    tokens_per_cta: int = 1\n"
assert p.count(old) == 1
p = p.replace(old, old + "    sequential: bool = False\n")
old = """        config.tokens_per_cta if config.lagged_prepare and route in ("pre", "post_pre") else 1,
    )"""
assert p.count(old) == 1
p = p.replace(old, """        config.tokens_per_cta if config.lagged_prepare and route in ("pre", "post_pre") else 1,
        bool(config.sequential_partials) and config.lagged_prepare and route in ("pre", "post_pre"),
    )""")
old = """                    tokens_per_cta=native.tokens_per_cta,
                )"""
assert p.count(old) == 1, p.count(old)
p = p.replace(old, """                    tokens_per_cta=native.tokens_per_cta,
                    sequential=native.sequential,
                )""")
old = """                            tokens_per_cta=native.tokens_per_cta)"""
assert p.count(old) == 1, p.count(old)
p = p.replace(old, """                            tokens_per_cta=native.tokens_per_cta,
                            sequential=native.sequential)""")
# The sequential path's finalize runs once per row (step 5).
old = """        lagged_prepared=config.lagged_prepare,
    )


def _post_pre_primary"""
assert p.count(old) == 1
p = p.replace(old, """        lagged_prepared=config.lagged_prepare,
        **({"lagged_threads": 256}
           if native.sequential
           and query.max_tokens > int(os.environ.get("B12X_MHC_SEQ_WIDE_FINALIZE_MAX", "8"))
           else {}),
    )


def _post_pre_primary""")
compile(p, "_preparation.py", "exec")
open(os.path.join(OUT, "_preparation.py"), "w").write(p)
print("wrote", OUT)
