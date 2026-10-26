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

A resident table (in device memory, or mapped host RAM) is this rank's row shard of the
table; its gather (``engram_table_gather``) reads each lookup's row from the shard and
dequantizes it the same way, B12X's Engram lookup of a resident table.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import os

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import prepare

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
#define ENGRAM_PHASE_FAILED 3

// Wait until the rows of the step this stream staged are published (cyclic
// sequence order). The clock, the timeout and the reader's phase are read only
// after 64 unanswered polls, so a wait for rows already in place costs one load. A
// timeout, or a reader that failed, records the time waited and the step, fenced,
// then stops the GPU.
__device__ __forceinline__ void engram_wait_rows(void *control) {
    unsigned *ctrl = (unsigned *)control;
    unsigned target;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(target) : "l"(ctrl + ENGRAM_CTRL_IDS_SEQ) : "memory");
    unsigned long long start = 0ull, limit = 0ull;
#ifdef ENGRAM_TIMING
    unsigned long long entered;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(entered));
#endif
    for (unsigned polls = 1u;; polls++) {
        unsigned seen;
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(seen) : "l"(ctrl + ENGRAM_CTRL_ROWS_SEQ) : "memory");
        if ((int)(seen - target) >= 0) {
#ifdef ENGRAM_TIMING
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
#endif
            return;
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


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def engram_rows_gather_kernel(
    max_lookups: int,
    shard_start: int,
    shard_end: int,
    columns: int = (1 << HEADS) - 1,
    threads: int = 64,
    vec: int = 4,
):
    assert threads * vec == ROW and 32 % vec == 0
    first, step, count = column_span(columns)
    tokens = T.dynamic("tokens")

    @T.prim_func
    def engram_rows_gather(
        ids: T.Tensor((tokens, HEADS), T.int64),
        num_tokens: T.Tensor((1,), T.int32),
        rows: T.Tensor((max_lookups, ROW), T.float8_e4m3fn),
        scales: T.Tensor((max_lookups, SCALES), T.uint8),
        out: T.Tensor((tokens, count * ROW), T.bfloat16),
    ):
        with T.Kernel(count, tokens, threads=threads) as (column, t):
            tx = T.get_thread_binding()
            head = first + column * step
            row = ids[t, head]
            lookup = t * HEADS + head
            owned = (t < num_tokens[0]) & (row >= shard_start) & (row < shard_end)
            if owned:
                exponent = T.cast(scales[lookup, tx * vec // 32], T.uint32)
                scale = T.reinterpret(T.shift_left(exponent, 23), T.float32)
                scale = T.if_then_else(exponent == 0, T.float32(2.0**-127), scale)
                scale = T.if_then_else(exponent == 255, T.float32(float("nan")), scale)
                for i in T.vectorized(vec):
                    out[t, column * ROW + tx * vec + i] = T.cast(
                        T.cast(rows[lookup, tx * vec + i], T.float32) * scale,
                        T.bfloat16,
                    )
            else:
                for i in T.vectorized(vec):
                    out[t, column * ROW + tx * vec + i] = T.bfloat16(0)

    return engram_rows_gather


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def engram_table_gather_kernel(
    shard_rows: int,
    shard_start: int,
    shard_end: int,
    threads: int = 64,
    vec: int = 4,
):
    """Every token's 24 rows from a resident table's shard: rows ``[shard_start,
    shard_end)`` of the table at local rows ``row - shard_start``. A token past the
    live count (or every token, when the count is outside ``[0, tokens]``) and a row
    outside the shard write zeros. B12X's lookup of a resident table, element for
    element."""
    assert threads * vec == ROW and 32 % vec == 0
    tokens = T.dynamic("tokens")

    @T.prim_func
    def engram_table_gather(
        ids: T.Tensor((tokens, HEADS), T.int64),
        num_tokens: T.Tensor((1,), T.int32),
        table: T.Tensor((shard_rows, ROW), T.float8_e4m3fn),
        scales: T.Tensor((shard_rows, SCALES), T.uint8),
        out: T.Tensor((tokens, HEADS * ROW), T.bfloat16),
    ):
        with T.Kernel(HEADS, tokens, threads=threads) as (head, t):
            tx = T.get_thread_binding()
            count = num_tokens[0]
            row = ids[t, head]
            owned = (
                count >= 0
                and count <= tokens
                and t < count
                and row >= shard_start
                and row < shard_end
            )
            if owned:
                local = row - shard_start
                exponent = T.cast(scales[local, tx * vec // 32], T.uint32)
                scale = T.reinterpret(T.shift_left(exponent, 23), T.float32)
                scale = T.if_then_else(exponent == 0, T.float32(2.0**-127), scale)
                scale = T.if_then_else(exponent == 255, T.float32(float("nan")), scale)
                for i in T.vectorized(vec):
                    out[t, head * ROW + tx * vec + i] = T.cast(
                        T.cast(table[local, tx * vec + i], T.float32) * scale,
                        T.bfloat16,
                    )
            else:
                for i in T.vectorized(vec):
                    out[t, head * ROW + tx * vec + i] = T.bfloat16(0)

    return engram_table_gather


def table_kernel_call(
    shard_rows: int, shard_start: int, shard_end: int
) -> prepare.TileLangCall:
    """The kernel call ``engram_table_gather`` makes for a resident shard of
    ``shard_rows`` rows holding the table's rows ``[shard_start, shard_end)``."""
    return prepare.call(engram_table_gather_kernel, shard_rows, shard_start, shard_end)


def row_kernel_calls(
    max_lookups: int, shard_start: int, shard_end: int, columns: int
) -> list[prepare.TileLangCall]:
    """The kernel calls ``engram_rows_gather`` makes for a reader of ``max_lookups``
    rows a step (``rows.shape[0]``) and the rank's rows ``[shard_start, shard_end)``
    of the hash columns in the mask ``columns``."""
    return [
        prepare.call(engram_rows_wait_kernel),
        prepare.call(
            engram_rows_gather_kernel, max_lookups, shard_start, shard_end, columns
        ),
        prepare.call(engram_rows_release_kernel),
    ]


@torch.library.custom_op("vllm::dsv41_engram_rows_gather", mutates_args=("out",))
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
    engram_rows_wait_kernel()(control)
    tokens = out.shape[0]
    if tokens:
        kernel = engram_rows_gather_kernel(
            rows.shape[0], shard_start, shard_end, columns
        )
        kernel(ids[:tokens], num_tokens, rows, scales, out)
    engram_rows_release_kernel()(control)


@engram_rows_gather.register_fake
def _engram_rows_gather_fake(
    ids,
    num_tokens,
    rows,
    scales,
    control,
    out,
    shard_start,
    shard_end,
    columns=(1 << HEADS) - 1,
):
    return None


@torch.library.custom_op("vllm::dsv41_engram_table_gather", mutates_args=("out",))
def engram_table_gather(
    ids: torch.Tensor,
    num_tokens: torch.Tensor,
    table: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    shard_start: int,
    shard_end: int,
) -> None:
    """Dequantize every token's 24 rows of a resident table into ``out`` ([tokens,
    6144] BF16, zeros past the first ``num_tokens`` tokens). ``ids`` ([tokens, 24]
    int64) are the row IDs; ``table`` ([shard_rows, 256] E4M3) and ``scales``
    ([shard_rows, 8] E8M0 bytes) are this rank's shard, which holds the table's rows
    ``[shard_start, shard_end)``."""
    kernel = engram_table_gather_kernel(table.shape[0], shard_start, shard_end)
    kernel(ids, num_tokens, table, scales.view(torch.uint8), out)


@engram_table_gather.register_fake
def _engram_table_gather_fake(
    ids, num_tokens, table, scales, out, shard_start, shard_end
):
    return None
