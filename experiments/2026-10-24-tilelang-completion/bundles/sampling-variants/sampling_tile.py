"""Temperature and select in Triton's tile form.

Triton loads a whole block before it computes or stores anything, and masks memory
operations instead of branching around them. The production ports interleave: each
of temperature's runs is loaded, divided and stored before the next run's load, and
the logits are read and written through one pointer, so the next load waits for the
previous store; select reads its ranks in a loop of dynamic length. These variants
load every run (or rank) first, with the count fixed when the kernel is built.
"""

import tilelang
import tilelang.language as T

from vllm.v1.worker.gpu.sample import tilelang_sampling as tls


@T.macro
def _load_run_at(src, raw, base, row, col0, vocab, run):
    """``tls._load_run`` into ``raw[base:base + run]``."""
    if run == 1:
        if col0 < vocab:
            raw[base] = src[row, col0]
    else:
        if col0 + run <= vocab:
            for k in T.vectorized(run):
                raw[base + k] = src[row, col0 + k]
        else:
            for k in T.unroll(run):
                if col0 + k < vocab:
                    raw[base + k] = src[row, col0 + k]


@T.macro
def _store_run_at(dst, raw, base, row, col0, vocab, run):
    """``tls._store_run`` from ``raw[base:base + run]``."""
    if run == 1:
        if col0 < vocab:
            dst[row, col0] = raw[base]
    else:
        if col0 + run <= vocab:
            for k in T.vectorized(run):
                dst[row, col0 + k] = raw[base + k]
        else:
            for k in T.unroll(run):
                if col0 + k < vocab:
                    dst[row, col0 + k] = raw[base + k]


@tilelang.jit(pass_configs=tls._PASS)
def temperature_tile_kernel(idx_dtype: str, run: int, threads: int):
    """``tls.temperature_kernel`` with the block's runs all loaded, then all divided,
    then all stored."""
    tokens, vocab, idx_len, temp_len = T.dynamic("tokens, vocab, idx_len, temp_len")
    logits_stride = T.dynamic("logits_stride", "int64")
    runs = tls._TEMPERATURE_BLOCK // (threads * run)

    @T.prim_func
    def temperature(
        logits: T.StridedTensor((tokens, vocab), (logits_stride, 1), T.float32),
        idx_mapping: T.Tensor((idx_len,), idx_dtype),
        temperature: T.Tensor((temp_len,), T.float32),
    ):
        with T.Kernel(
            tokens, T.ceildiv(vocab, tls._TEMPERATURE_BLOCK), threads=threads
        ) as (t, blk):
            T.import_source(tls._HEADER)
            tx = T.get_thread_binding()
            temp = T.alloc_local((1,), T.float32)
            raw = T.alloc_local((runs * run,), T.float32)
            if run > 1:
                T.assume(logits_stride % 16 == 0)
            temp[0] = temperature[T.cast(idx_mapping[t], T.int64)]
            if (temp[0] != T.float32(0)) & (temp[0] != T.float32(1)):
                for r in T.unroll(runs):
                    col0 = blk * tls._TEMPERATURE_BLOCK + r * (threads * run) + run * tx
                    _load_run_at(logits, raw, r * run, t, col0, vocab, run)
                for k in T.unroll(runs * run):
                    raw[k] = T.call_extern("float32", "vllm_div_full", raw[k], temp[0])
                for r in T.unroll(runs):
                    col0 = blk * tls._TEMPERATURE_BLOCK + r * (threads * run) + run * tx
                    _store_run_at(logits, raw, r * run, t, col0, vocab, run)

    return temperature


@tilelang.jit(pass_configs=tls._PASS)
def select_tile_kernel(output_dtype: str, ranks: int):
    """``tls.select_first_argmax_kernel`` (a warp a row) for a fixed number of
    ranks: lane ``k < ranks`` loads rank ``k``'s pair, the others hold Triton's
    masked values (-inf, 2**30), and the lanes reduce over the next power of two."""
    assert 1 <= ranks <= 32
    rows, width, out_len = T.dynamic("rows, width, out_len")
    pairs_stride = T.dynamic("pairs_stride", "int64")
    span = 1 << max(ranks - 1, 0).bit_length()

    @T.prim_func
    def select_first_argmax(
        pairs: T.StridedTensor((rows, width), (pairs_stride, 1), T.float32),
        out: T.Tensor((out_len,), output_dtype),
        out_stride: T.int64,
    ):
        with T.Kernel(rows, threads=32) as row:
            T.import_source(tls._HEADER)
            tx = T.get_thread_binding()
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            lane = T.min(tx, ranks - 1)
            value = pairs[row, 2 * lane]
            index = pairs[row, 2 * lane + 1]
            best[0] = T.if_then_else(tx < ranks, value, -T.infinity(T.float32))
            best_index[0] = T.if_then_else(
                tx < ranks, T.cast(index, T.int32), T.int32(1 << 30)
            )
            if span >= 32:
                tls._first_exchange(best, best_index, other, other_index, 16)
            if span >= 16:
                tls._first_exchange(best, best_index, other, other_index, 8)
            if span >= 8:
                tls._first_exchange(best, best_index, other, other_index, 4)
            if span >= 4:
                tls._first_exchange(best, best_index, other, other_index, 2)
            if span >= 2:
                tls._first_exchange(best, best_index, other, other_index, 1)
            if tx == 0:
                out[T.cast(row, T.int64) * out_stride] = T.cast(
                    best_index[0], output_dtype
                )

    return select_first_argmax


@tilelang.jit(pass_configs=tls._PASS)
def partial_tile_kernel(base_dtype: str, bias_dtype: str | None, vector: bool, threads: int):
    """``tls.partial_argmax_kernel`` with a thread's runs all loaded before any is
    combined (production loads, combines, then loads the next run)."""
    rows, vocab, values_len, indices_len = T.dynamic(
        "rows, vocab, values_len, indices_len"
    )
    base_stride, bias_stride = T.dynamic("base_stride, bias_stride", "int64")
    biased = bias_dtype is not None
    if not biased:
        bias_dtype = base_dtype
    warps = threads // 32
    vec = tls._VEC
    runs = tls.GREEDY_BLOCK // (threads * vec)
    passes = tls.GREEDY_BLOCK // threads

    @T.prim_func
    def partial_argmax(
        base: T.StridedTensor((rows, vocab), (base_stride, 1), base_dtype),
        bias: T.StridedTensor((rows, vocab), (bias_stride, 1), bias_dtype),
        values: T.Tensor((values_len,), T.float32),
        indices: T.Tensor((indices_len,), T.int32),
    ):
        with T.Kernel(rows, T.ceildiv(vocab, tls.GREEDY_BLOCK), threads=threads) as (
            row,
            part,
        ):
            T.import_source(tls._HEADER)
            tx = T.get_thread_binding()
            a = T.alloc_local((passes,), base_dtype)
            b = T.alloc_local((passes,), bias_dtype)
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            partial = T.alloc_shared((warps,), T.float32)
            partial_index = T.alloc_shared((warps,), T.int32)
            best[0] = -T.infinity(T.float32)
            best_index[0] = tls._INT32_MAX
            if vector:
                T.assume(base_stride % vec == 0)
                T.assume(bias_stride % vec == 0)
                T.assume(vocab % vec == 0)
                for run in T.unroll(runs):
                    col0 = part * tls.GREEDY_BLOCK + (run * threads + tx) * vec
                    if col0 < vocab:
                        for i in T.vectorized(vec):
                            a[run * vec + i] = base[row, col0 + i]
                        if biased:
                            for i in T.vectorized(vec):
                                b[run * vec + i] = bias[row, col0 + i]
                for run in T.unroll(runs):
                    col0 = part * tls.GREEDY_BLOCK + (run * threads + tx) * vec
                    if col0 < vocab:
                        for i in T.unroll(vec):
                            value = T.cast(a[run * vec + i], T.float32)
                            if biased:
                                value = T.cast(
                                    T.cast(
                                        value + T.cast(b[run * vec + i], T.float32),
                                        base_dtype,
                                    ),
                                    T.float32,
                                )
                            tls._first_keep_or_take(best, best_index, value, col0 + i)
            else:
                for j in T.unroll(passes):
                    col = part * tls.GREEDY_BLOCK + j * threads + tx
                    if col < vocab:
                        a[j] = base[row, col]
                        if biased:
                            b[j] = bias[row, col]
                for j in T.unroll(passes):
                    col = part * tls.GREEDY_BLOCK + j * threads + tx
                    if col < vocab:
                        value = T.cast(a[j], T.float32)
                        if biased:
                            value = T.cast(
                                T.cast(value + T.cast(b[j], T.float32), base_dtype),
                                T.float32,
                            )
                        tls._first_keep_or_take(best, best_index, value, col)
            tls._warp_first_argmax(best, best_index, other, other_index)
            if tx % 32 == 0:
                partial[tx // 32] = best[0]
                partial_index[tx // 32] = best_index[0]
            T.sync_threads()
            if tx == 0:
                for w in T.unroll(1, warps):
                    tls._first_keep_or_take(best, best_index, partial[w], partial_index[w])
                slot = T.cast(row, T.int64) * T.ceildiv(vocab, tls.GREEDY_BLOCK) + part
                values[slot] = best[0]
                indices[slot] = best_index[0]

    return partial_argmax


@tilelang.jit(pass_configs=tls._PASS)
def finish_tile_kernel(output: str, parts: int):
    """``tls.finish_argmax_kernel`` (a warp a row) for a number of parts fixed when
    built: lane ``k`` reads parts ``k``, ``k + 32``, ... up front (Triton's masked
    values past the end), combines them, and the lanes reduce over the next power
    of two."""
    rows, values_len, indices_len, out_len = T.dynamic(
        "rows, values_len, indices_len, out_len"
    )
    pair = output == "pair"
    out_dtype = "float32" if pair else output
    per_lane = -(-parts // 32)
    span = min(32, 1 << max(parts - 1, 0).bit_length())

    @T.prim_func
    def finish_argmax(
        values: T.Tensor((values_len,), T.float32),
        indices: T.Tensor((indices_len,), T.int32),
        out: T.Tensor((out_len,), out_dtype),
        out_rows: T.int32,
        out_stride: T.int64,
        vocab_start: T.int32,
    ):
        with T.Kernel(out_rows, threads=32) as row:
            T.import_source(tls._HEADER)
            tx = T.get_thread_binding()
            v = T.alloc_local((per_lane,), T.float32)
            ix = T.alloc_local((per_lane,), T.int32)
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            for k in T.unroll(per_lane):
                p = tx + k * 32
                slot = T.cast(row, T.int64) * parts + T.min(p, parts - 1)
                v[k] = T.if_then_else(p < parts, values[slot], -T.infinity(T.float32))
                ix[k] = T.if_then_else(p < parts, indices[slot], tls._INT32_MAX)
            best[0] = v[0]
            best_index[0] = ix[0]
            for k in T.unroll(1, per_lane):
                tls._first_keep_or_take(best, best_index, v[k], ix[k])
            if span >= 32:
                tls._first_exchange(best, best_index, other, other_index, 16)
            if span >= 16:
                tls._first_exchange(best, best_index, other, other_index, 8)
            if span >= 8:
                tls._first_exchange(best, best_index, other, other_index, 4)
            if span >= 4:
                tls._first_exchange(best, best_index, other, other_index, 2)
            if span >= 2:
                tls._first_exchange(best, best_index, other, other_index, 1)
            if tx == 0:
                if pair:
                    out[T.cast(row, T.int64) * 2] = best[0]
                    out[T.cast(row, T.int64) * 2 + 1] = T.cast(
                        best_index[0] + vocab_start, T.float32
                    )
                else:
                    out[T.cast(row, T.int64) * out_stride] = T.cast(
                        best_index[0], out_dtype
                    )

    return finish_argmax
