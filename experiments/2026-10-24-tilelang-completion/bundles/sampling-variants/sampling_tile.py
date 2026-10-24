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
