# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang Engram n-gram hashing for DeepSeek-V4.1, every layer in one launch.

For token ``t`` of request ``r`` (the last request whose query starts at or before
``t``) and head ``h`` of Engram layer ``l``, the sources are the compressed ids of
tokens ``t``, ``t - 1``, ``t - 2`` and ``t - 3``: query tokens while they are in the
request's query, then its three committed history tokens, then nothing. A query
token compresses through the tokenizer map unless it is an image token or out of
vocabulary; a history token unless it is the image sentinel or out of vocabulary.
From the first missing source on, every lag takes the pad id (the compressed id of
token 2). Heads 0-7 hash 2-grams, 8-15 3-grams and 16-23 4-grams::

    mixed = XOR over lags < order of source[lag] * multiplier[l, lag]
    hash  = mixed mod prime[l, h] + offset[l, h]

Tokens past the step's live count hash to -1. The arithmetic is B12X's Engram hash
(compress, request ids and hash in three Triton launches per layer, after vLLM's
metadata copy); here each block hashes one token for every layer and head, and the
first block also publishes the request and live token counts.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import tilelang
import tilelang.language as T
import torch

HEADS = 24
LAGS = 4
HISTORY = LAGS - 1
IMAGE_SENTINEL = 129264  # the history exclusion of vLLM's Engram metadata
PAD_TOKEN = 2


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def engram_hash_kernel(
    layers: int, vocab: int, ids_dtype: str = "int64", history_dtype: str = "int64"
):
    threads = (layers * HEADS + 31) // 32 * 32
    tokens = T.dynamic("tokens")
    seqs1 = T.dynamic("seqs1")  # requests + 1
    history_rows = T.dynamic("history_rows")
    history_stride = T.dynamic("history_stride")
    out_rows = T.dynamic("out_rows")

    @T.prim_func
    def engram_hash(
        ids: T.Tensor((tokens,), ids_dtype),
        image_mask: T.Tensor((tokens,), T.bool),
        starts: T.Tensor((seqs1,), T.int32),
        history: T.StridedTensor(
            (history_rows, HISTORY), (history_stride, 1), history_dtype
        ),
        token_map: T.Tensor((vocab,), T.int64),
        multipliers: T.Tensor((layers, LAGS), T.int64),
        primes: T.Tensor((layers, HEADS), T.int64),
        offsets: T.Tensor((layers, HEADS), T.int64),
        out: T.Tensor((out_rows, layers, HEADS), T.int64),
        num_seqs: T.Tensor((1,), T.int32),
        num_tokens: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(tokens, threads=threads) as t:
            tx = T.get_thread_binding()
            layer = tx // HEADS
            head = tx % HEADS
            seqs = seqs1 - 1
            live = starts[seqs]
            if t == 0 and tx == 0:
                num_seqs[0] = seqs
                num_tokens[0] = live
            if layer < layers:
                if t >= live:
                    out[t, layer, head] = T.int64(-1)
                else:
                    request = T.alloc_var(T.int32)
                    request = 0
                    for r in T.serial(seqs - 1):
                        if starts[r + 1] <= t:
                            request = r + 1
                    start = starts[request]
                    pad = token_map[PAD_TOKEN]
                    order = head // 8 + 2
                    blocked = T.alloc_var(T.bool)
                    blocked = False
                    mixed = T.alloc_var(T.int64)
                    mixed = T.int64(0)
                    for lag in T.unroll(LAGS):
                        source = T.alloc_var(T.int64)
                        source = T.int64(-1)
                        relative = t - start - lag
                        if relative >= 0:
                            raw = T.cast(ids[t - lag], T.int64)
                            if raw >= 0 and raw < vocab and not image_mask[t - lag]:
                                source = token_map[raw]
                        elif relative >= -HISTORY:
                            raw = T.cast(history[request, HISTORY + relative], T.int64)
                            if raw >= 0 and raw < vocab and raw != IMAGE_SENTINEL:
                                source = token_map[raw]
                        blocked = blocked or source == -1
                        if lag < order:
                            value = T.if_then_else(blocked, pad, source)
                            mixed = mixed ^ (value * multipliers[layer, lag])
                    prime = primes[layer, head]
                    out[t, layer, head] = mixed % prime + offsets[layer, head]

    return engram_hash


@torch.library.custom_op(
    "vllm::dsv41_tilelang_engram_hash", mutates_args=("out", "num_seqs", "num_tokens")
)
def engram_hash(
    ids: torch.Tensor,
    image_mask: torch.Tensor,
    starts: torch.Tensor,
    history: torch.Tensor,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
) -> None:
    """Hash ``ids`` ([T], int32 or int64; T may include padding past the live count
    ``starts[-1]``) into ``out`` ([>= T, layers, 24] int64) and write the request
    and live token counts. ``history`` (int32 or int64; the runner keeps int32)
    holds each request's last three committed token ids, oldest first."""
    if ids.numel() == 0:
        return
    kernel = engram_hash_kernel(
        multipliers.shape[0],
        token_map.numel(),
        str(ids.dtype).removeprefix("torch."),
        str(history.dtype).removeprefix("torch."),
    )
    kernel(
        ids,
        image_mask,
        starts,
        history,
        token_map,
        multipliers,
        primes,
        offsets,
        out,
        num_seqs,
        num_tokens,
    )


@engram_hash.register_fake
def _engram_hash_fake(
    ids,
    image_mask,
    starts,
    history,
    token_map,
    multipliers,
    primes,
    offsets,
    out,
    num_seqs,
    num_tokens,
):
    return None


def geometry_tensors(geometry, device) -> tuple[torch.Tensor, ...]:
    """Multipliers [L, 4], primes [L, 24] and offsets [L, 24] of an Engram hash
    geometry, as int64 device tensors in layer order."""
    return tuple(
        torch.tensor(values, dtype=torch.int64, device=device)
        for values in (geometry.multipliers, geometry.primes, geometry.offsets)
    )


__all__ = ["engram_hash", "engram_hash_kernel", "geometry_tensors"]
