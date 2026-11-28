# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang wait for and gather of a step's Engram rows from mapped host memory.

The Engram disk reader (``engram_rows/_reader.c``) puts each of this rank's lookups'
E4M3 row and E8M0 scales in mapped host buffers, in lookup order (token-major, 24
heads), and publishes the step in the control block. Two kernels on the forward's
stream:
- the wait: one thread waits until the rows of the step this stream staged are
  published, bounded by the control block's timeout. A timed-out wait, or a reader
  that reports a failure, records the step and the time waited and stops the GPU,
  so no kernel after it reads stale rows. One wait per table and step: polling host
  memory from every gather block cost about 0.5 us a block.
- the gather: one block per (column, token) of this rank's hash columns dequantizes
  the row as b12x's Engram lookup does (the value times its E8M0 scale, 2^-127 for
  byte 0 and NaN for 255, rounded to BF16) into the token's slice of the column, the
  rank's columns side by side. A token past the step, or a row outside this rank's
  shard, loads nothing and writes zeros (the reader leaves those slots untouched).
- the release: one thread writes the step it gathered into the control block, so the
  reader may now fill the buffers with the next step's early rows (those the host
  hashed while this step ran).
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import os

import tilelang
import tilelang.language as T
import torch

from spark_ds41.tilelang import prepare

HEADS = 24
ROW = 256
SCALES = ROW // 32
CONTROL_WORDS = 64

# Debug only: the wait records each wait's length for the reader's timeline
# (SPARK_DS41F_ENGRAM_TIMELINE, common/engram.py); compiled in only when set.
_TIMING = (
    "#define ENGRAM_TIMING\n"
    if os.environ.get("SPARK_DS41F_ENGRAM_TIMELINE", "0") == "1"
    else ""
)

# Control-block word offsets, as in engram_rows/_reader.c.
_HEADER = (
    _TIMING
    + r"""
// Engram rows: the reader publishes a step by writing its sequence number into the
// control block in mapped host memory; this stream wrote the step it staged there.
#define ENGRAM_CTRL_IDS_SEQ 0
#define ENGRAM_CTRL_CONSUMED_SEQ 1
#define ENGRAM_CTRL_ROWS_SEQ 16
#define ENGRAM_CTRL_TIMEOUT_US 32
#define ENGRAM_CTRL_WAITED_US 33
#define ENGRAM_CTRL_FAILED_SEQ 34
#define ENGRAM_CTRL_GPU_WAIT_SEQ 40
#define ENGRAM_CTRL_GPU_WAIT_NS 41
#define ENGRAM_CTRL_PHASE 48
#define ENGRAM_CTRL_RETIRED_CTAS 56
#define ENGRAM_PHASE_FAILED 3

// Wait until the rows of the step this stream staged are published (cyclic
// sequence order). The clock, the timeout and the reader's phase are read only
// after 64 unanswered polls, so a wait for rows already in place costs one load. A
// timeout, or a reader that failed, records the time waited and the step, fenced,
// then stops the GPU.
__device__ __forceinline__ unsigned engram_wait_rows_cta(void *control, int is_lead_cta) {
    unsigned *ctrl = (unsigned *)control;
    unsigned target;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(target) : "l"(ctrl + ENGRAM_CTRL_IDS_SEQ) : "memory");
    unsigned long long start = 0ull, limit = 0ull;
#ifdef ENGRAM_TIMING
    unsigned long long entered = 0ull;
    if (is_lead_cta) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(entered));
    }
#else
    (void)is_lead_cta;
#endif
    for (unsigned polls = 1u;; polls++) {
        unsigned seen;
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(seen) : "l"(ctrl + ENGRAM_CTRL_ROWS_SEQ) : "memory");
        if ((int)(seen - target) >= 0) {
#ifdef ENGRAM_TIMING
            if (is_lead_cta) {
                unsigned long long left;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(left));
                unsigned long long ns = left - entered;
                unsigned waited = ns > 0xffffffffull ? 0xffffffffu : (unsigned)ns;
                asm volatile("st.relaxed.sys.global.u32 [%0], %1;"
                             :: "l"(ctrl + ENGRAM_CTRL_GPU_WAIT_NS), "r"(waited)
                             : "memory");
                asm volatile("st.release.sys.global.u32 [%0], %1;"
                             :: "l"(ctrl + ENGRAM_CTRL_GPU_WAIT_SEQ), "r"(target)
                             : "memory");
            }
#endif
            return target;
        }
        if ((polls & 63u) == 0u) {
            unsigned long long now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            unsigned phase;
            asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(phase) : "l"(ctrl + ENGRAM_CTRL_PHASE) : "memory");
            if (start == 0ull) {
                unsigned timeout;
                asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(timeout) : "l"(ctrl + ENGRAM_CTRL_TIMEOUT_US) : "memory");
                start = now;
                limit = 1000ull * (unsigned long long)timeout;
            }
            if (phase == ENGRAM_PHASE_FAILED || now - start >= limit) {
                unsigned waited = (unsigned)((now - start) / 1000ull);
                asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(ctrl + ENGRAM_CTRL_WAITED_US), "r"(waited) : "memory");
                asm volatile("fence.sc.sys;" ::: "memory");
                asm volatile("st.relaxed.sys.global.u32 [%0], %1;" :: "l"(ctrl + ENGRAM_CTRL_FAILED_SEQ), "r"(target) : "memory");
                asm volatile("fence.sc.sys;" ::: "memory");
                asm volatile("trap;");
            }
        }
    }
}

__device__ __forceinline__ void engram_wait_rows(void *control) {
    engram_wait_rows_cta(control, 1);
}

// The stream's gather of the step it staged is done (the kernels before this one have
// retired their loads of the mapped rows): tell the reader, with release semantics.
__device__ __forceinline__ void engram_release_rows(void *control) {
    unsigned *ctrl = (unsigned *)control;
    unsigned staged;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                 : "=r"(staged) : "l"(ctrl + ENGRAM_CTRL_IDS_SEQ) : "memory");
    asm volatile("st.release.sys.global.u32 [%0], %1;"
                 :: "l"(ctrl + ENGRAM_CTRL_CONSUMED_SEQ), "r"(staged) : "memory");
}

// Device-L2 synchronization state across CTAs of engram_rows_step_kernel (128B cache-line
// isolated words so only CTA 0 touches mapped-host control on wait and only the last
// finishing CTA touches mapped-host control on release).
__device__ __align__(128) unsigned engram_gpu_sync[128] = {0u};
#define ENGRAM_SYNC_COMPLETED 0
#define ENGRAM_SYNC_READY     32
#define ENGRAM_SYNC_RETIRED   64
#define ENGRAM_SYNC_STAGED    96

__device__ __forceinline__ void engram_wait_rows_step(void *control, int is_lead_cta, int total_ctas) {
    if (total_ctas <= 1) {
        unsigned target = engram_wait_rows_cta(control, 1);
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(&engram_gpu_sync[ENGRAM_SYNC_STAGED]), "r"(target) : "memory");
        return;
    }
    unsigned my_step;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(my_step) : "l"(&engram_gpu_sync[ENGRAM_SYNC_COMPLETED]) : "memory");
    my_step += 1u;
    if (is_lead_cta) {
        unsigned target = engram_wait_rows_cta(control, 1);
        asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(&engram_gpu_sync[ENGRAM_SYNC_STAGED]), "r"(target) : "memory");
        asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(&engram_gpu_sync[ENGRAM_SYNC_READY]), "r"(my_step) : "memory");
    } else {
        for (;;) {
            unsigned ready;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(ready) : "l"(&engram_gpu_sync[ENGRAM_SYNC_READY]) : "memory");
            if (ready == my_step) break;
            asm volatile("nanosleep.u32 16;");
        }
    }
}

__device__ __forceinline__ void engram_release_rows_cta(void *control, int total_ctas) {
    unsigned *ctrl = (unsigned *)control;
    if (total_ctas <= 1) {
        unsigned staged;
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(staged) : "l"(&engram_gpu_sync[ENGRAM_SYNC_STAGED]) : "memory");
        asm volatile("st.release.sys.global.u32 [%0], %1;"
                     :: "l"(ctrl + ENGRAM_CTRL_CONSUMED_SEQ), "r"(staged) : "memory");
        return;
    }
    unsigned limit = (unsigned)(total_ctas - 1);
    unsigned prev;
    asm volatile("atom.inc.release.gpu.global.u32 %0, [%1], %2;"
                 : "=r"(prev)
                 : "l"(&engram_gpu_sync[ENGRAM_SYNC_RETIRED]), "r"(limit)
                 : "memory");
    if (prev == limit) {
        unsigned my_step, staged;
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(my_step) : "l"(&engram_gpu_sync[ENGRAM_SYNC_READY]) : "memory");
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(staged) : "l"(&engram_gpu_sync[ENGRAM_SYNC_STAGED]) : "memory");
        asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(&engram_gpu_sync[ENGRAM_SYNC_COMPLETED]), "r"(my_step) : "memory");
        asm volatile("st.release.sys.global.u32 [%0], %1;"
                     :: "l"(ctrl + ENGRAM_CTRL_CONSUMED_SEQ), "r"(staged) : "memory");
    }
}

// One-shot streaming loads (L2::evict_first) for step row IDs and mapped-host
// E4M3 rows / E8M0 scales: each slot is consumed once by a single thread, so
// streaming keeps mapped-host reads from evicting resident weights in the 24 MB L2.
__device__ __forceinline__ long long engram_ld_cs_s64(const void *ptr) {
    long long val;
    asm("ld.global.cs.s64 %0, [%1];" : "=l"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ unsigned int engram_ld_cs_u8(const void *ptr) {
    unsigned int val;
    asm("ld.global.cs.u8 %0, [%1];" : "=r"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ unsigned int engram_ld_cs_u32(const void *ptr) {
    unsigned int val;
    asm("ld.global.cs.u32 %0, [%1];" : "=r"(val) : "l"(ptr));
    return val;
}
"""  # noqa: E501 (CUDA source)
)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def engram_rows_wait_kernel():
    @T.prim_func
    def engram_rows_wait(control: T.Tensor((CONTROL_WORDS,), T.int32)):
        with T.Kernel(1, threads=32):
            T.import_source(_HEADER)
            if T.get_thread_binding() == 0:
                T.call_extern(
                    "engram_wait_rows", T.address_of(control[0]), dtype="handle"
                )

    return engram_rows_wait


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def engram_rows_release_kernel():
    @T.prim_func
    def engram_rows_release(control: T.Tensor((CONTROL_WORDS,), T.int32)):
        with T.Kernel(1, threads=32):
            T.import_source(_HEADER)
            if T.get_thread_binding() == 0:
                T.call_extern(
                    "engram_release_rows", T.address_of(control[0]), dtype="handle"
                )

    return engram_rows_release


def column_span(columns: int) -> tuple[int, int, int]:
    """The hash columns of the 24-bit mask ``columns`` as (first, step, count): a
    rank reads every ``step``-th column from its ``first`` (``column_shard``)."""
    heads = [head for head in range(HEADS) if columns >> head & 1]
    step = heads[1] - heads[0] if len(heads) > 1 else 1
    if not heads or heads != list(range(heads[0], HEADS, step))[: len(heads)]:
        raise ValueError(
            f"Engram row columns must be every n-th column from the first: {columns:#x}"
        )
    return heads[0], step, len(heads)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def engram_rows_gather_kernel(
    max_lookups: int,
    shard_start: int,
    shard_end: int,
    columns: int = (1 << HEADS) - 1,
    out_cols: int = 0,
    threads: int = 64,
    vec: int = 4,
):
    assert threads * vec == ROW and 32 % vec == 0
    first, step, count = column_span(columns)
    stride_cols = out_cols or count
    assert stride_cols >= count
    tokens = T.dynamic("tokens")

    @T.prim_func
    def engram_rows_gather(
        ids: T.Tensor((tokens, HEADS), T.int64),
        num_tokens: T.Tensor((1,), T.int32),
        rows: T.Tensor((max_lookups, ROW), T.float8_e4m3fn),
        scales: T.Tensor((max_lookups, SCALES), T.uint8),
        out: T.Tensor((tokens, stride_cols * ROW), T.bfloat16),
    ):
        with T.Kernel(count, tokens, threads=threads) as (column, t):
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            head = first + column * step
            row = T.call_extern(
                "int64", "engram_ld_cs_s64", T.address_of(ids[t, head])
            )
            lookup = t * HEADS + head
            owned = (t < num_tokens[0]) & (row >= shard_start) & (row < shard_end)
            if owned:
                exponent = T.call_extern(
                    "uint32",
                    "engram_ld_cs_u8",
                    T.address_of(scales[lookup, tx * vec // 32]),
                )
                scale = T.reinterpret(T.shift_left(exponent, 23), T.float32)
                scale = T.if_then_else(exponent == 0, T.float32(2.0**-127), scale)
                scale = T.if_then_else(exponent == 255, T.float32(float("nan")), scale)
                if vec == 4:
                    packed = T.call_extern(
                        "uint32",
                        "engram_ld_cs_u32",
                        T.address_of(rows[lookup, tx * vec]),
                    )
                    for i in T.vectorized(vec):
                        byte = T.cast(
                            T.shift_right(packed, T.uint32(8 * i)) & T.uint32(0xFF),
                            T.uint8,
                        )
                        out[t, column * ROW + tx * vec + i] = T.cast(
                            T.cast(T.reinterpret(byte, T.float8_e4m3fn), T.float32)
                            * scale,
                            T.bfloat16,
                        )
                else:
                    for i in T.vectorized(vec):
                        out[t, column * ROW + tx * vec + i] = T.cast(
                            T.cast(rows[lookup, tx * vec + i], T.float32) * scale,
                            T.bfloat16,
                        )
            else:
                for i in T.vectorized(vec):
                    out[t, column * ROW + tx * vec + i] = T.bfloat16(0)

    return engram_rows_gather


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def engram_rows_step_kernel(
    max_lookups: int,
    shard_start: int,
    shard_end: int,
    columns: int = (1 << HEADS) - 1,
    out_cols: int = 0,
    vec: int = 4,
):
    """Fused multi-CTA wait + gather + release kernel for a step's Engram rows."""
    assert 32 % vec == 0
    first, step, count = column_span(columns)
    stride_cols = out_cols or count
    assert stride_cols >= count
    row_threads = ROW // vec
    rows_per_cta = next(r for r in range(min(count, 8), 0, -1) if count % r == 0)
    threads = rows_per_cta * row_threads
    col_blocks = count // rows_per_cta
    tokens = T.dynamic("tokens")

    @T.prim_func
    def engram_rows_step(
        ids: T.Tensor((tokens, HEADS), T.int64),
        num_tokens: T.Tensor((1,), T.int32),
        rows: T.Tensor((max_lookups, ROW), T.float8_e4m3fn),
        scales: T.Tensor((max_lookups, SCALES), T.uint8),
        control: T.Tensor((CONTROL_WORDS,), T.int32),
        out: T.Tensor((tokens, stride_cols * ROW), T.bfloat16),
    ):
        with T.Kernel(col_blocks, tokens, threads=threads) as (col_block, t):
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            total_ctas = col_blocks * tokens
            row_slot = tx // row_threads
            lane = tx % row_threads
            column = col_block * rows_per_cta + row_slot
            head = first + column * step
            row = T.call_extern(
                "int64", "engram_ld_cs_s64", T.address_of(ids[t, head])
            )
            n_tok = num_tokens[0]
            if tx == 0:
                T.call_extern(
                    "engram_wait_rows_step",
                    T.address_of(control[0]),
                    T.if_then_else((col_block == 0) & (t == 0), T.int32(1), T.int32(0)),
                    total_ctas,
                    dtype="handle",
                )
            T.sync_threads()
            lookup = t * HEADS + head
            owned = (t < n_tok) & (row >= shard_start) & (row < shard_end)
            if owned:
                exponent = T.call_extern(
                    "uint32",
                    "engram_ld_cs_u8",
                    T.address_of(scales[lookup, lane * vec // 32]),
                )
                scale = T.reinterpret(T.shift_left(exponent, 23), T.float32)
                scale = T.if_then_else(exponent == 0, T.float32(2.0**-127), scale)
                scale = T.if_then_else(exponent == 255, T.float32(float("nan")), scale)
                if vec == 4:
                    packed = T.call_extern(
                        "uint32",
                        "engram_ld_cs_u32",
                        T.address_of(rows[lookup, lane * vec]),
                    )
                    for i in T.vectorized(vec):
                        byte = T.cast(
                            T.shift_right(packed, T.uint32(8 * i)) & T.uint32(0xFF),
                            T.uint8,
                        )
                        out[t, column * ROW + lane * vec + i] = T.cast(
                            T.cast(T.reinterpret(byte, T.float8_e4m3fn), T.float32)
                            * scale,
                            T.bfloat16,
                        )
                else:
                    for i in T.vectorized(vec):
                        out[t, column * ROW + lane * vec + i] = T.cast(
                            T.cast(rows[lookup, lane * vec + i], T.float32) * scale,
                            T.bfloat16,
                        )
            else:
                for i in T.vectorized(vec):
                    out[t, column * ROW + lane * vec + i] = T.bfloat16(0)
            T.sync_threads()
            if tx == 0:
                T.call_extern(
                    "engram_release_rows_cta",
                    T.address_of(control[0]),
                    col_blocks * tokens,
                    dtype="handle",
                )

    return engram_rows_step


def row_kernel_calls(
    max_lookups: int,
    shard_start: int,
    shard_end: int,
    columns: int,
    out_cols: int = 0,
) -> list[prepare.TileLangCall]:
    """The kernel calls ``engram_rows_gather`` makes for a reader of ``max_lookups``
    rows a step (``rows.shape[0]``) and the rank's rows ``[shard_start, shard_end)``
    of the hash columns in the mask ``columns``."""
    _, _, count = column_span(columns)
    extra = (out_cols,) if out_cols and out_cols != count else ()
    return [
        prepare.call(
            engram_rows_step_kernel,
            max_lookups,
            shard_start,
            shard_end,
            columns,
            *extra,
        ),
    ]


def engram_rows_gather(
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    rows: torch.Tensor,
    scales: torch.Tensor,
    control: torch.Tensor,
    out: torch.Tensor,
    shard_start: int,
    shard_end: int,
    columns: int = (1 << HEADS) - 1,
) -> None:
    """Wait for the staged step's rows, dequantize the first ``num_tokens`` tokens'
    rows into ``out`` ([tokens, columns * 256] BF16: the rank's hash columns side by
    side), then release the buffers to the reader. ``ids`` ([>= tokens, 24] int64)
    are the step's row IDs; ``rows``, ``scales`` and ``control`` are the reader's
    mapped buffers; the rank owns the rows in [shard_start, shard_end) of the hash
    columns in the mask ``columns``, every n-th from the first."""
    tokens = out.shape[0]
    if tokens:
        _, _, count = column_span(columns)
        stride_cols = out.shape[1] // ROW
        extra = (stride_cols,) if stride_cols != count else ()
        kernel = engram_rows_step_kernel(
            rows.shape[0], shard_start, shard_end, columns, *extra
        )
        kernel(ids[:tokens], num_tokens, rows, scales, control, out)
    else:
        engram_rows_wait_kernel()(control)
        engram_rows_release_kernel()(control)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def engram_resident_lookup_kernel(
    shard_rows: int,
    shard_start: int,
    shard_end: int,
    columns: int = (1 << HEADS) - 1,
    out_cols: int = 0,
    threads: int = 64,
    vec: int = 4,
):
    assert threads * vec == ROW and 32 % vec == 0
    first, step, count = column_span(columns)
    stride_cols = out_cols or count
    assert stride_cols >= count
    tokens = T.dynamic("tokens")

    @T.prim_func
    def engram_resident_lookup(
        ids: T.Tensor((tokens, HEADS), T.int64),
        num_tokens: T.Tensor((1,), T.int32),
        weight: T.Tensor((shard_rows, ROW), T.float8_e4m3fn),
        scales: T.Tensor((shard_rows, SCALES), T.uint8),
        out: T.Tensor((tokens, stride_cols * ROW), T.bfloat16),
    ):
        with T.Kernel(count, tokens, threads=threads) as (column, t):
            tx = T.get_thread_binding()
            head = first + column * step
            row = ids[t, head]
            n_tok = num_tokens[0]
            owned = (
                (n_tok >= 0)
                & (n_tok <= tokens)
                & (t < n_tok)
                & (row >= shard_start)
                & (row < shard_end)
            )
            if owned:
                local_row = row - T.int64(shard_start)
                exponent = T.cast(scales[local_row, tx * vec // 32], T.uint32)
                scale = T.reinterpret(T.shift_left(exponent, 23), T.float32)
                scale = T.if_then_else(exponent == 0, T.float32(2.0**-127), scale)
                scale = T.if_then_else(exponent == 255, T.float32(float("nan")), scale)
                for i in T.vectorized(vec):
                    out[t, column * ROW + tx * vec + i] = T.cast(
                        T.cast(weight[local_row, tx * vec + i], T.float32) * scale,
                        T.bfloat16,
                    )
            else:
                for i in T.vectorized(vec):
                    out[t, column * ROW + tx * vec + i] = T.bfloat16(0)

    return engram_resident_lookup


def resident_lookup_calls(
    shard_rows: int,
    shard_start: int,
    shard_end: int,
    columns: int = (1 << HEADS) - 1,
    out_cols: int = 0,
) -> list[prepare.TileLangCall]:
    """The kernel call ``engram_resident_lookup`` makes for a resident shard of
    ``shard_rows`` rows covering ``[shard_start, shard_end)`` and the hash columns
    in the mask ``columns``."""
    _, _, count = column_span(columns)
    extra = (out_cols,) if out_cols and out_cols != count else ()
    return [
        prepare.call(
            engram_resident_lookup_kernel,
            shard_rows,
            shard_start,
            shard_end,
            columns,
            *extra,
        ),
    ]


def engram_resident_lookup(
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    shard_start: int,
    shard_end: int,
    columns: int = (1 << HEADS) - 1,
) -> None:
    """Dequantize the first ``num_tokens`` tokens' resident Engram rows from ``weight``
    and ``scales`` into ``out`` ([tokens, columns * 256] BF16) and zero any trailing
    rows up to ``out.shape[0]``."""
    tokens = out.shape[0]
    if tokens:
        _, _, count = column_span(columns)
        stride_cols = out.shape[1] // ROW
        extra = (stride_cols,) if stride_cols != count else ()
        kernel = engram_resident_lookup_kernel(
            weight.shape[0], shard_start, shard_end, columns, *extra
        )
        kernel(ids[:tokens], num_tokens, weight, scales.view(torch.uint8), out)


engram_table_gather_kernel = engram_resident_lookup_kernel
table_gather_calls = resident_lookup_calls
engram_table_gather = engram_resident_lookup
