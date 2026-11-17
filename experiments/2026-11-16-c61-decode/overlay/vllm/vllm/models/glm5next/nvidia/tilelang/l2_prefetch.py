# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang bulk L2 prefetch, the TileLang counterpart of the CuTe DSL kernel in
``l2_prefetch.py``.

A single issuer thread (CTA 0, lane 0) walks the (ptr, bytes) segment table and
issues one ``cp.async.bulk.prefetch.L2`` with an ``evict_last`` cache policy per
64 KiB chunk (rounded down to 16 bytes for segment tails), pacing chunks via
``%globaltimer`` after an initial 128 KiB burst so side-stream prefetches do not
overflow the hardware TMA prefetch FIFO or stall concurrent kernels. Cache hints
only; nothing is written.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import tilelang
import tilelang.language as T
import torch

_BULK_CHUNK_BYTES = 65536
_BURST_CHUNKS = 2
_PACE_DELAY_NS = 200

# createpolicy has no side effects (the compiler may hoist it); the prefetch does.
_HEADER = r"""
#include <stdint.h>

__device__ __forceinline__ unsigned long long l2pf_policy_evict_last() {
    unsigned long long policy;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

__device__ __forceinline__ unsigned long long l2_prefetch_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void l2pf_bulk_prefetch(long long address, int bytes,
                                                   unsigned long long policy) {
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
                 :: "l"(address), "r"(bytes), "l"(policy));
}

template <int CHUNK_BYTES, int BURST_CHUNKS, int DELAY_NS>
__device__ __forceinline__ void l2_prefetch_segments_paced(
    const int64_t* __restrict__ segs,
    int nseg2) {
    if (threadIdx.x != 0) return;
    unsigned long long policy = l2pf_policy_evict_last();
    int nseg = nseg2 >> 1;
    int issued = 0;
    long long step = (long long)gridDim.x * (long long)CHUNK_BYTES;
    for (int s = 0; s < nseg; ++s) {
        long long base = (long long)segs[2 * s];
        long long nbytes = (long long)segs[2 * s + 1];
        for (long long off = (long long)blockIdx.x * (long long)CHUNK_BYTES; off < nbytes; off += step) {
            long long rem = nbytes - off;
            int size = (int)((rem < (long long)CHUNK_BYTES ? rem : (long long)CHUNK_BYTES) & ~15LL);
            if (size > 0) {
                l2pf_bulk_prefetch(base + off, size, policy);
                ++issued;
                if (DELAY_NS > 0 && issued >= BURST_CHUNKS) {
                    issued = 0;
                    if (off + step < nbytes || s + 1 < nseg) {
                        unsigned long long t0 = l2_prefetch_globaltimer();
                        while ((l2_prefetch_globaltimer() - t0) < (unsigned long long)DELAY_NS) {}
                    }
                }
            }
        }
    }
}
"""


@tilelang.jit(target="cuda")
def l2_prefetch_kernel(grid: int, block: int, chunk: int):
    """The prefetch of a shape-dynamic segment table in 64 KiB bulk-prefetch
    requests paced by ``%globaltimer``; one kernel per (grid, block, chunk)."""
    nseg2 = T.dynamic("nseg2")
    step = max(int(chunk), _BULK_CHUNK_BYTES)

    @T.prim_func
    def l2_prefetch(segs: T.Tensor((nseg2,), T.int64)):
        with T.Kernel(max(int(grid), 1), threads=32) as _bx:
            T.import_source(_HEADER)
            T.call_extern(
                f"l2_prefetch_segments_paced<{step}, {_BURST_CHUNKS}, {_PACE_DELAY_NS}>",
                segs,
                nseg2,
                dtype="handle",
            )

    return l2_prefetch


def compile_launcher(grid: int, block: int, chunk: int):
    """Compile once for a shape-dynamic segment table (a cache hit once the kernel
    was prepared); returns (segs_tensor, cuda_stream_handle) -> None, like the CuTe
    launcher."""
    kernel = l2_prefetch_kernel(grid, block, chunk)

    def launch(segs: torch.Tensor, stream_handle: int) -> None:
        # TileLang launches on the current stream of the call.
        with torch.cuda.stream(
            torch.cuda.ExternalStream(stream_handle, device=segs.device)
        ):
            kernel(segs)

    return launch
