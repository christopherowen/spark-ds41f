# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang writers of DeepSeek-V4.1's paged KV-cache records.

A token's 512 latent values become one record at its slot (``slot // page_size``
selects the page, ``slot % page_size`` the record; negative or out-of-range slots
are skipped):

- ``swa`` (528 bytes): 16 groups of 32 values as E4M3, then 16 UE8M0 scales.
  A group's scale is ``max(amax, 1e-4) * fp32(1/448)`` rounded up to a power of
  two (the exponent is bumped when the mantissa is nonzero).
- ``indexed`` (288 bytes): 32 groups of 16 values as packed E2M1 (low nibble
  first), then 32 E4M3 scales. A group's scale is ``max(amax, 6 * 2^-9) / 6``
  (IEEE division) rounded to E4M3, saturating; the values are divided by its
  decoded value (a rounded reciprocal would move exact E2M1 midpoints off their
  ties-to-even) and rounded to nearest-even E2M1.

The arithmetic is B12X's ``write_cache`` for ``cache_format="deepseek_v41"``,
step for step, so the records are the same bytes.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import tilelang
import tilelang.language as T
import torch

from .indexer import _e2m1_code

DIM = 512
RECORD_BYTES = {"swa": 528, "indexed": 288}
_GROUP = {"swa": 32, "indexed": 16}
_E4M3_MAX = 448.0


_WORD = {"swa": (4, T.uint32, torch.uint32), "indexed": (8, T.uint64, torch.uint64)}


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def cache_writer_kernel(cache_kind: str, page_size: int, slot_dtype: str = "int64"):
    """One record per token, payload stored in whole words through a word view of
    the pages (scales through the byte view). ``swa``: eight threads per 32-value
    group, four values and one 32-bit word per thread, the group's amax shuffled
    across its eight lanes. ``indexed``: one thread per 16-value group, sixteen
    values and one 64-bit word of packed E2M1."""
    threads = 128
    group_size = _GROUP[cache_kind]
    groups = DIM // group_size
    word_bytes, word_dtype, _ = _WORD[cache_kind]
    per_thread = word_bytes if cache_kind == "swa" else group_size  # values per thread
    lanes = group_size // per_thread  # threads per group: 8 or 1
    per_token = groups * lanes  # 128 or 32
    tokens_per_block = threads // per_token  # 1 or 4
    record = RECORD_BYTES[cache_kind]
    payload = DIM if cache_kind == "swa" else DIM // 2  # scale bytes follow
    tokens, kv_stride = T.dynamic("tokens, kv_stride")
    pages, page_bytes, page_stride = T.dynamic("pages, page_bytes, page_stride")
    page_words, word_stride = T.dynamic("page_words, word_stride")

    @T.prim_func
    def write(
        kv: T.StridedTensor((tokens, DIM), (kv_stride, 1), T.bfloat16),
        cache: T.StridedTensor((pages, page_bytes), (page_stride, 1), T.uint8),
        words: T.StridedTensor((pages, page_words), (word_stride, 1), word_dtype),
        slots: T.Tensor((tokens,), slot_dtype),
    ):
        with T.Kernel(T.ceildiv(tokens, tokens_per_block), threads=threads) as bx:
            tx = T.get_thread_binding()
            token = bx * tokens_per_block + tx // per_token
            item = tx % per_token
            group, lane = item // lanes, item % lanes
            col = group * group_size + lane * per_thread
            values = T.alloc_local((per_thread,), T.float32)
            amax = T.alloc_var(T.float32)
            word = T.alloc_var(word_dtype)
            amax = 0.0
            if token < tokens:
                for i in T.unroll(per_thread):
                    values[i] = T.cast(kv[token, col + i], T.float32)
                    amax = T.max(amax, T.abs(values[i]))
            if cache_kind == "swa":
                amax = T.max(amax, T.shfl_xor(amax, 1))
                amax = T.max(amax, T.shfl_xor(amax, 2))
                amax = T.max(amax, T.shfl_xor(amax, 4))
            if token < tokens:
                slot = T.cast(slots[token], T.int64)
                page = slot // page_size
                if (slot >= 0) & (page < pages):
                    base = (slot % page_size) * record
                    word = T.cast(0, word_dtype)
                    if cache_kind == "swa":
                        bits = T.reinterpret(
                            T.fmul(T.max(amax, 1e-4), T.float32(1.0 / _E4M3_MAX)),
                            T.uint32,
                        )
                        bumped = T.if_then_else(
                            (bits & 0x7FFFFF) != 0, (bits + 0x800000) & 0x7F800000, bits
                        )
                        exponent = T.cast(T.shift_right(bumped, 23) & 0xFF, T.int32)
                        # The exact reciprocal of the power-of-two scale.
                        inverse = T.reinterpret(
                            T.cast(T.shift_left(254 - exponent, 23), T.uint32),
                            T.float32,
                        )
                        for i in T.unroll(per_thread):
                            code = T.cast(T.fmul(values[i], inverse), T.float8_e4m3fn)
                            word = word | T.shift_left(
                                T.cast(T.reinterpret(code, T.uint8), word_dtype), 8 * i
                            )
                        words[page, (base + col) // word_bytes] = word
                        if lane == 0:
                            cache[page, base + payload + group] = T.cast(
                                exponent, T.uint8
                            )
                    else:
                        scale = T.max(amax, 6.0 * 2.0**-9) / 6.0
                        scale_code = T.cast(T.min(scale, _E4M3_MAX), T.float8_e4m3fn)
                        decoded = T.cast(scale_code, T.float32)
                        for i in T.unroll(per_thread):
                            code = _e2m1_code(values[i] / decoded)
                            word = word | T.shift_left(T.cast(code, word_dtype), 4 * i)
                        words[page, (base + col // 2) // word_bytes] = word
                        cache[page, base + payload + group] = T.reinterpret(
                            scale_code, T.uint8
                        )

    return write


def write_cache(
    kv: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    page_size: int,
    cache_kind: str,
) -> None:
    """Write ``kv`` ([T, 512] BF16, unit column stride) into ``cache`` (``[pages,
    page bytes]`` or ``[pages, records, record bytes]`` bytes, any page stride) at
    ``slot_mapping`` ([T] int32 or int64)."""
    if kv.shape[0] == 0:
        return
    if kv.dtype != torch.bfloat16 or kv.shape[1] != DIM or kv.stride(1) != 1:
        raise ValueError("V4.1 cache records come from [T, 512] BF16 rows")
    pages = cache.view(torch.uint8)
    if pages.dim() > 2:  # [pages, records, record bytes]
        pages = pages.flatten(1)
    if pages.shape[1] < page_size * RECORD_BYTES[cache_kind]:
        raise ValueError(
            f"cache pages hold fewer than {page_size} {cache_kind} records"
        )
    word_bytes, _, word_torch = _WORD[cache_kind]
    if pages.stride(0) % word_bytes or pages.shape[1] % word_bytes:
        raise ValueError(f"cache pages are not whole {word_bytes}-byte words")
    kernel = cache_writer_kernel(
        cache_kind, page_size, str(slot_mapping.dtype).removeprefix("torch.")
    )
    kernel(kv, pages, pages.view(word_torch), slot_mapping)


__all__ = ["RECORD_BYTES", "cache_writer_kernel", "write_cache"]
