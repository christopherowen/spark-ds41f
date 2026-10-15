# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang gather of a step's Engram rows from mapped host memory.

The Engram disk reader (``engram_rows/_reader.c``) puts each lookup's E4M3 row and
E8M0 scales in mapped host buffers, in lookup order (token-major, 24 heads), and
publishes the step in the control block. One block per (head, token): its first
thread waits until the rows of the step this stream staged are published, bounded
by the control block's timeout; a timed-out wait records the step and the time
waited and stops the GPU, so no kernel after it reads stale rows. The block then
dequantizes the row as b12x's Engram lookup does (the value times its E8M0 scale,
2^-127 for byte 0 and NaN for 255, rounded to BF16) into the token's head slice,
or writes zeros for a token past the step or a row outside this rank's shard.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import tilelang
import tilelang.language as T
import torch

HEADS = 24
ROW = 256
SCALES = ROW // 32
CONTROL_WORDS = 64

# Control-block word offsets, as in engram_rows/_reader.c.
_HEADER = r"""
// Engram rows: the reader publishes a step by writing its sequence number into the
// control block in mapped host memory; this stream wrote the step it staged there.
#define ENGRAM_CTRL_IDS_SEQ 0
#define ENGRAM_CTRL_ROWS_SEQ 16
#define ENGRAM_CTRL_TIMEOUT_US 32
#define ENGRAM_CTRL_WAITED_US 33
#define ENGRAM_CTRL_FAILED_SEQ 34

// Wait until the rows of the step this stream staged are published (cyclic
// sequence order). The clock and the timeout are read only after 64 unanswered
// polls, so a wait for rows already in place costs one load. A timeout records the
// time waited and the step, fenced, then stops the GPU.
__device__ __forceinline__ void engram_wait_rows(void *control) {
    unsigned *ctrl = (unsigned *)control;
    unsigned target;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(target) : "l"(ctrl + ENGRAM_CTRL_IDS_SEQ) : "memory");
    unsigned long long start = 0ull, limit = 0ull;
    for (unsigned polls = 1u;; polls++) {
        unsigned seen;
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(seen) : "l"(ctrl + ENGRAM_CTRL_ROWS_SEQ) : "memory");
        if ((int)(seen - target) >= 0) {
            return;
        }
        if ((polls & 63u) == 0u) {
            unsigned long long now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            if (start == 0ull) {
                unsigned timeout;
                asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(timeout) : "l"(ctrl + ENGRAM_CTRL_TIMEOUT_US) : "memory");
                start = now;
                limit = 1000ull * (unsigned long long)timeout;
            } else if (now - start >= limit) {
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
"""


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def engram_rows_gather_kernel(
    max_lookups: int,
    shard_start: int,
    shard_end: int,
    threads: int = 64,
    vec: int = 4,
):
    assert threads * vec == ROW and 32 % vec == 0
    tokens = T.dynamic("tokens")

    @T.prim_func
    def engram_rows_gather(
        ids: T.Tensor((tokens, HEADS), T.int64),
        num_tokens: T.Tensor((1,), T.int32),
        rows: T.Tensor((max_lookups, ROW), T.float8_e4m3fn),
        scales: T.Tensor((max_lookups, SCALES), T.uint8),
        control: T.Tensor((CONTROL_WORDS,), T.int32),
        out: T.Tensor((tokens, HEADS * ROW), T.bfloat16),
    ):
        with T.Kernel(HEADS, tokens, threads=threads) as (head, t):
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            if tx == 0:
                T.call_extern("engram_wait_rows", T.address_of(control[0]), dtype="handle")
            T.sync_threads()
            row = ids[t, head]
            local = (t < num_tokens[0]) & (row >= shard_start) & (row < shard_end)
            lookup = t * HEADS + head
            exponent = T.cast(scales[lookup, tx * vec // 32], T.uint32)
            scale = T.reinterpret(T.shift_left(exponent, 23), T.float32)
            scale = T.if_then_else(exponent == 0, T.float32(2.0**-127), scale)
            scale = T.if_then_else(exponent == 255, T.float32(float("nan")), scale)
            for i in T.vectorized(vec):
                value = T.cast(rows[lookup, tx * vec + i], T.float32) * scale
                out[t, head * ROW + tx * vec + i] = T.if_then_else(
                    local, T.cast(value, T.bfloat16), T.bfloat16(0)
                )

    return engram_rows_gather


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
) -> None:
    """Wait for the staged step's rows and dequantize the first ``num_tokens``
    tokens' rows into ``out`` ([tokens, 24 * 256] BF16). ``ids`` ([>= tokens, 24]
    int64) are the step's row IDs; ``rows``, ``scales`` and ``control`` are the
    reader's mapped buffers."""
    tokens = out.shape[0]
    if tokens == 0:
        return
    kernel = engram_rows_gather_kernel(rows.shape[0], shard_start, shard_end)
    kernel(ids[:tokens], num_tokens, rows, scales, control, out)


@engram_rows_gather.register_fake
def _engram_rows_gather_fake(ids, num_tokens, rows, scales, control, out, shard_start, shard_end):
    return None
