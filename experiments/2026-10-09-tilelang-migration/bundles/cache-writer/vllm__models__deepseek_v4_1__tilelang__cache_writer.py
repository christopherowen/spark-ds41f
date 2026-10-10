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
step for step, so the records are the same bytes. One thread writes one group.
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


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def cache_writer_kernel(
    cache_kind: str, page_size: int, slot_dtype: str = "int64", threads: int = 128
):
    group_size = _GROUP[cache_kind]
    groups = DIM // group_size
    record = RECORD_BYTES[cache_kind]
    payload = DIM if cache_kind == "swa" else DIM // 2  # scale bytes follow
    tokens, kv_stride = T.dynamic("tokens, kv_stride")
    pages, page_bytes, page_stride = T.dynamic("pages, page_bytes, page_stride")

    @T.prim_func
    def write(
        kv: T.StridedTensor((tokens, DIM), (kv_stride, 1), T.bfloat16),
        cache: T.StridedTensor((pages, page_bytes), (page_stride, 1), T.uint8),
        slots: T.Tensor((tokens,), slot_dtype),
    ):
        with T.Kernel(T.ceildiv(tokens * groups, threads), threads=threads) as bx:
            item = bx * threads + T.get_thread_binding()
            token, group = item // groups, item % groups
            values = T.alloc_local((group_size,), T.float32)
            amax = T.alloc_var(T.float32)
            if token < tokens:
                slot = T.cast(slots[token], T.int64)
                page = slot // page_size
                if (slot >= 0) & (page < pages):
                    base = (slot % page_size) * record
                    amax = 0.0
                    for i in T.unroll(group_size):
                        values[i] = T.cast(kv[token, group * group_size + i], T.float32)
                        amax = T.max(amax, T.abs(values[i]))
                    if cache_kind == "swa":
                        bits = T.reinterpret(
                            T.fmul(T.max(amax, 1e-4), T.float32(1.0 / _E4M3_MAX)),
                            T.uint32,
                        )
                        bumped = T.if_then_else(
                            (bits & 0x7FFFFF) != 0,
                            (bits + 0x800000) & 0x7F800000,
                            bits,
                        )
                        exponent = T.cast(T.shift_right(bumped, 23) & 0xFF, T.int32)
                        # The exact reciprocal of the power-of-two scale.
                        inverse = T.reinterpret(
                            T.cast(T.shift_left(254 - exponent, 23), T.uint32),
                            T.float32,
                        )
                        for i in T.unroll(group_size):
                            code = T.cast(T.fmul(values[i], inverse), T.float8_e4m3fn)
                            cache[page, base + group * group_size + i] = T.reinterpret(
                                code, T.uint8
                            )
                        cache[page, base + payload + group] = T.cast(exponent, T.uint8)
                    else:
                        scale = T.max(amax, 6.0 * 2.0**-9) / 6.0
                        scale_code = T.cast(T.min(scale, _E4M3_MAX), T.float8_e4m3fn)
                        decoded = T.cast(scale_code, T.float32)
                        for i in T.unroll(group_size // 2):
                            lo = _e2m1_code(values[2 * i] / decoded)
                            hi = _e2m1_code(values[2 * i + 1] / decoded)
                            cache[page, base + group * (group_size // 2) + i] = T.cast(
                                lo | (hi << 4), T.uint8
                            )
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
    kernel = cache_writer_kernel(
        cache_kind, page_size, str(slot_mapping.dtype).removeprefix("torch.")
    )
    kernel(kv, pages, slot_mapping)


__all__ = ["RECORD_BYTES", "cache_writer_kernel", "write_cache"]
