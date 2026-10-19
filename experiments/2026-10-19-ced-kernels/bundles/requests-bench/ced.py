# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer row selection for the DeepSeek-V4.1 CED decoder."""

import tilelang
import tilelang.language as T

_PASS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


@tilelang.jit(pass_configs=_PASS)
def requests_kernel(max_requests: int):
    starts_len = T.dynamic("starts_len")
    seq_len = T.dynamic("seq_len")
    block = max(32, 1 << max_requests.bit_length())

    @T.prim_func
    def ced_requests(
        starts: T.Tensor((starts_len,), T.int32),
        seq: T.Tensor((seq_len,), T.int32),
        limits: T.Tensor((max_requests,), T.int32),
        prefix: T.Tensor((max_requests,), T.int64),
        out_starts: T.Tensor((max_requests + 1,), T.int32),
        positions: T.Tensor((max_requests,), T.int64),
        replay: T.Tensor((max_requests,), T.int64),
        counts: T.Tensor((2,), T.int32),
        nr: T.int32,
    ):
        with T.Kernel(1, threads=128):
            kept = T.alloc_fragment((block,), T.int32)
            cumulative = T.alloc_fragment((block,), T.int32)
            for r in T.Parallel(block):
                kept[r] = T.if_then_else(
                    r < nr, T.min(starts[r + 1] - starts[r], limits[r]), 0
                )
            T.copy(kept, cumulative)
            T.cumsum(cumulative, dim=0)
            for r in T.Parallel(max_requests + 1):
                out_starts[r] = T.if_then_else(
                    r < nr, cumulative[r] - kept[r], cumulative[block - 1]
                )
            for r in T.Parallel(max_requests):
                if r < nr:
                    q = starts[r + 1] - starts[r]
                    position = T.cast(seq[r], T.int64) - kept[r]
                    positions[r] = position
                    replay[r] = T.max(
                        prefix[r], T.if_then_else(q > kept[r], position, 0)
                    )
                else:
                    positions[r] = -1
                    replay[r] = 0
            for i in T.Parallel(2):
                counts[i] = T.if_then_else(i == 0, cumulative[block - 1], nr)

    return ced_requests


def requests(starts, seq, limits, prefix, out_starts, positions, replay, counts, nr):
    requests_kernel(positions.shape[0])(
        starts, seq, limits, prefix, out_starts, positions, replay, counts, nr
    )
