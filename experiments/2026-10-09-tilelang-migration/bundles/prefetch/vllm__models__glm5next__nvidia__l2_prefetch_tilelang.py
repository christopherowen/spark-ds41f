# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang bulk L2 prefetch, the TileLang counterpart of the CuTe DSL kernel in
``l2_prefetch.py``.

Same work split and the same two PTX instructions: grid x block threads walk the
(ptr, bytes) segment table and issue one ``cp.async.bulk.prefetch.L2`` with an
``evict_last`` cache policy per 4 KB chunk, rounded down to 16 bytes. Cache hints
only; nothing is written.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import torch

# createpolicy has no side effects (the compiler may hoist it); the prefetch does.
_HEADER = r"""
__device__ __forceinline__ unsigned long long l2pf_policy_evict_last() {
    unsigned long long policy;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

__device__ __forceinline__ void l2pf_bulk_prefetch(long long address, int bytes,
                                                   unsigned long long policy) {
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
                 :: "l"(address), "r"(bytes), "l"(policy));
}
"""


def _kernel(grid: int, block: int, chunk: int):
    import tilelang.language as T

    nseg2 = T.dynamic("nseg2")

    @T.prim_func
    def l2_prefetch(segs: T.Tensor((nseg2,), T.int64)):
        with T.Kernel(grid, threads=block) as bx:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            tid = T.cast(bx, T.int64) * block + T.cast(tx, T.int64)
            stride = T.int64(grid * block)
            policy = T.alloc_local((1,), T.uint64)
            policy[0] = T.call_extern("l2pf_policy_evict_last", dtype=T.uint64)
            for s in T.serial(nseg2 // 2):
                base = segs[2 * s]
                nbytes = segs[2 * s + 1]
                nchunks = (nbytes + (chunk - 1)) // chunk
                # This thread's chunks: tid, tid + stride, ... below nchunks.
                for k in T.serial(T.max((nchunks - tid + (stride - 1)) // stride, T.int64(0))):
                    off = (tid + k * stride) * chunk
                    size = T.min(nbytes - off, T.int64(chunk)) // 16 * 16
                    if size > 0:
                        T.call_extern("l2pf_bulk_prefetch", base + off, T.cast(size, T.int32),
                                      policy[0], dtype="handle")

    return l2_prefetch


def compile_launcher(grid: int, block: int, chunk: int):
    """Compile once for a shape-dynamic segment table; returns
    (segs_tensor, cuda_stream_handle) -> None, like the CuTe launcher."""
    import tilelang

    kernel = tilelang.compile(_kernel(grid, block, chunk), target="cuda")

    def launch(segs: torch.Tensor, stream_handle: int) -> None:
        # TileLang launches on the current stream of the call.
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_handle, device=segs.device)):
            kernel(segs)

    return launch
