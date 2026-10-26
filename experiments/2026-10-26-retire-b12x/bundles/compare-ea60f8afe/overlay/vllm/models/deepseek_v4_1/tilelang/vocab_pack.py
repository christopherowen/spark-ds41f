# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact 12-bit storage for BF16 vocabulary heads, packed by TileLang at load.

A BF16 value is a sign bit, an 8-bit exponent and a 7-bit mantissa. Each value
here keeps its sign and mantissa in one byte and its exponent in a 4-bit code:
code ``c`` in 1..15 means exponent ``base + c - 1`` for one 15-exponent window
chosen per tensor, and code 0 means exponent 0 (zero and subnormal values keep
their bits exactly). The rare values whose exponent falls outside the window
are exceptions: their slot holds +0 and their exact BF16 bits sit in a per-row
list (``row_offsets`` indexes it by row; each entry is the value's bits in the
low half and its column in the high half), in column order within a row.

Row layout, ``K + K / 2`` bytes: the ``K`` sign-and-mantissa bytes, then the
codes. Byte ``j`` of each 128-column group's 64 code bytes holds the code of
column ``j`` in its low nibble and of column ``j + 64`` in its high nibble.
``vocab.packed_vocab_projection`` reads this layout.

The format, the window choice and the packing are B12X's
(``b12x.gemm.packed_bf16_vocab_projection``, spark-ds41f's B12X patch 0012), and
the packed tensors are B12X's byte for byte. ``pack`` streams the BF16 weight
through three TileLang passes (exponent histogram, per-row exception counts,
then the packed bytes and exceptions) and allocates nothing large but its
result; ``pack(..., in_place=True)`` writes the result into the weight's own
storage instead. The write pass decodes every value it encodes and compares it
with the source bits; ``pack`` refuses unless all of them match, so a packed
tensor is the checkpoint tensor, not an approximation.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
import importlib.util
from dataclasses import dataclass

import tilelang
import tilelang.language as T
import torch

GROUP = 128  # columns per code group
WINDOW = 15  # exponents a 4-bit code covers (code 0 is exponent 0)
WARP = 32
# Rows one histogram block counts; rows (one a warp) one count or pack block covers.
_HISTOGRAM_ROWS = 16
_HISTOGRAM_THREADS = 256
_ROW_WARPS = 4
# In place, rows are packed through a staging block of this many rows.
_BLOCK_ROWS = 1_024

_PASS_CONFIGS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


@dataclass(frozen=True)
class PackedBf16Weight:
    """A packed ``[out_features, in_features]`` BF16 weight."""

    values: torch.Tensor
    exponent_base: int
    exception_rows: torch.Tensor
    exception_entries: torch.Tensor
    row_offsets: torch.Tensor
    exceptions: int

    @property
    def out_features(self) -> int:
        return int(self.values.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.values.shape[1]) * 2 // 3


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def exponent_histogram_kernel(K: int):
    """Count every value's 8-bit exponent into ``histogram`` [256] (added to it)."""
    rows, threads = _HISTOGRAM_ROWS, _HISTOGRAM_THREADS
    N = T.dynamic("N")

    @T.prim_func
    def exponent_histogram(
        weight: T.Tensor((N, K), T.int16),
        histogram: T.Tensor((256,), T.int32),
    ):
        with T.Kernel(T.ceildiv(N, rows), threads=threads) as bx:
            tx = T.get_thread_binding()
            counts = T.alloc_shared((256,), T.int32)
            T.fill(counts, 0)
            T.sync_threads()
            for step in T.serial(T.ceildiv(rows * K, threads)):
                index = step * threads + tx
                row = bx * rows + index // K
                if index < rows * K and row < N:
                    bits = T.cast(weight[row, index % K], T.int32) & 0xFFFF
                    T.atomic_add(counts[(bits >> 7) & 0xFF], 1)
            T.sync_threads()
            if counts[tx] != 0:
                T.atomic_add(histogram[tx], counts[tx])

    return exponent_histogram


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def exception_count_kernel(K: int):
    """``counts[row]``: the row's values outside the window from ``base`` (one warp
    a row)."""
    N = T.dynamic("N")

    @T.prim_func
    def exception_count(
        weight: T.Tensor((N, K), T.int16),
        counts: T.Tensor((N,), T.int32),
        base: T.int32,
    ):
        with T.Kernel(T.ceildiv(N, _ROW_WARPS), threads=_ROW_WARPS * WARP) as bx:
            tx = T.get_thread_binding()
            lane = tx % WARP
            row = bx * _ROW_WARPS + tx // WARP
            if row < N:
                total = T.alloc_var(T.int32)
                total = 0
                for chunk in T.serial(K // WARP):
                    bits = T.cast(weight[row, chunk * WARP + lane], T.int32) & 0xFFFF
                    exponent = (bits >> 7) & 0xFF
                    outside = exponent != 0 and (
                        exponent < base or exponent >= base + WINDOW
                    )
                    total = total + T.if_then_else(outside, 1, 0)
                for step in T.unroll(5):
                    total = total + T.shfl_xor(total, 16 >> step)
                if lane == 0:
                    counts[row] = total

    return exception_count


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def pack_rows_kernel(K: int):
    """Pack rows ``row_start + r`` into row ``r`` of ``values``, one warp a row.

    A warp takes each 128-column group as four 32-column parts in column order;
    a part's exceptions take the next slots of the row's list in lane order (a
    warp prefix sum), so the list is in column order, as ``row_offsets`` (the
    exception counts' prefix sum) expects. Every encoded value is decoded and
    compared with its source bits; ``mismatches`` counts the values that differ
    and the rows whose exceptions do not fill their slots exactly.
    """
    width = K + K // 2
    N = T.dynamic("N")
    N1 = T.dynamic("N1")  # N + 1
    R = T.dynamic("R")
    E = T.dynamic("E")

    @T.prim_func
    def pack_rows(
        weight: T.Tensor((N, K), T.int16),
        values: T.Tensor((R, width), T.uint8),
        row_offsets: T.Tensor((N1,), T.int32),
        exception_rows: T.Tensor((E,), T.int32),
        exception_entries: T.Tensor((E,), T.int32),
        mismatches: T.Tensor((1,), T.int32),
        base: T.int32,
        row_start: T.int32,
    ):
        with T.Kernel(T.ceildiv(R, _ROW_WARPS), threads=_ROW_WARPS * WARP) as bx:
            tx = T.get_thread_binding()
            lane = tx % WARP
            local = bx * _ROW_WARPS + tx // WARP
            if local < R:
                row = row_start + local
                codes = T.alloc_local((4,), T.int32)
                written = T.alloc_var(T.int32)
                written = row_offsets[row]
                bad = T.alloc_var(T.int32)
                bad = 0
                inclusive = T.alloc_var(T.int32)
                for group in T.serial(K // GROUP):
                    for part in T.unroll(GROUP // WARP):
                        column = group * GROUP + part * WARP + lane
                        bits = T.cast(weight[row, column], T.int32) & 0xFFFF
                        exponent = (bits >> 7) & 0xFF
                        outside = exponent != 0 and (
                            exponent < base or exponent >= base + WINDOW
                        )
                        code = T.if_then_else(
                            exponent != 0 and not outside, exponent - base + 1, 0
                        )
                        sign_mantissa = ((bits >> 8) & 0x80) | (bits & 0x7F)
                        low = T.if_then_else(outside, 0, sign_mantissa)
                        decoded = (
                            ((low & 0x80) << 8)
                            | (T.if_then_else(code == 0, 0, code + base - 1) << 7)
                            | (low & 0x7F)
                        )
                        differs = not outside and decoded != bits
                        bad = bad + T.if_then_else(differs, 1, 0)
                        flag = T.if_then_else(outside, 1, 0)
                        inclusive = flag
                        for step in T.unroll(5):
                            prior = T.shfl_up(inclusive, 1 << step)
                            inclusive = inclusive + T.if_then_else(
                                lane >= (1 << step), prior, 0
                            )
                        if outside:
                            slot = written + inclusive - 1
                            exception_rows[slot] = row
                            exception_entries[slot] = (column << 16) | bits
                        written = written + T.shfl_sync(inclusive, WARP - 1)
                        values[local, column] = T.cast(low, T.uint8)
                        codes[part] = code
                    values[local, K + group * (GROUP // 2) + lane] = T.cast(
                        codes[0] | (codes[2] << 4), T.uint8
                    )
                    values[local, K + group * (GROUP // 2) + WARP + lane] = T.cast(
                        codes[1] | (codes[3] << 4), T.uint8
                    )
                bad = bad + T.if_then_else(written != row_offsets[row + 1], 1, 0)
                for step in T.unroll(5):
                    bad = bad + T.shfl_xor(bad, 16 >> step)
                if lane == 0 and bad != 0:
                    T.atomic_add(mismatches[0], bad)

    return pack_rows


def is_supported(device=None) -> bool:
    """True on SM120/SM121 GPUs with TileLang available."""
    if not torch.cuda.is_available():
        return False
    device = torch.device(device) if device is not None else torch.device("cuda")
    if device.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (
        major == 12
        and minor in (0, 1)
        and importlib.util.find_spec("tilelang") is not None
    )


def _check_weight(weight: torch.Tensor) -> None:
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise TypeError("packing needs a 2-D torch.bfloat16 weight")
    if not weight.is_cuda or not weight.is_contiguous():
        raise ValueError("packing needs a contiguous CUDA weight")
    if weight.shape[0] == 0 or weight.shape[1] == 0 or weight.shape[1] % GROUP:
        raise ValueError(
            f"packing needs rows and K divisible by {GROUP}, got {tuple(weight.shape)}"
        )


def pack(weight: torch.Tensor, *, in_place: bool = False) -> PackedBf16Weight:
    """Pack ``weight`` exactly; raise if any value would not round-trip.

    Out of place, it allocates the result plus the 256-bin histogram, the
    mismatch counter and a row-count prefix sum. ``in_place=True`` writes the
    packed rows into the first three quarters of ``weight``'s own storage, so
    the only other allocation is one 1,024-row staging block (7.5 MiB at
    K=5,120); ``weight`` must not be read as BF16 afterwards. A packed row is
    shorter than a BF16 row, so each block's packed bytes land only on rows
    already read: blocks are packed into staging while their own rows are
    intact, then copied into place, in row order.
    """
    _check_weight(weight)
    out_features, in_features = map(int, weight.shape)
    width = in_features + in_features // 2
    device = weight.device
    bits = weight.view(torch.int16)
    histogram = torch.zeros(256, dtype=torch.int32, device=device)
    exponent_histogram_kernel(in_features)(bits, histogram)
    hist = histogram.to(torch.int64).cpu()
    hist[0] = 0
    windows = hist.unfold(0, WINDOW, 1).sum(1)
    windows[0] = -1
    base = int(windows.argmax())  # the lowest of equally good windows
    row_offsets = torch.zeros(out_features + 1, dtype=torch.int32, device=device)
    exception_count_kernel(in_features)(bits, row_offsets[1:], base)
    row_offsets[1:] = torch.cumsum(row_offsets[1:], 0, dtype=torch.int32)
    exceptions = int(row_offsets[-1])
    # Kernels take a valid pointer even when the row offsets select nothing.
    rows = torch.zeros(max(exceptions, 1), dtype=torch.int32, device=device)
    entries = torch.zeros_like(rows)
    mismatches = torch.zeros(1, dtype=torch.int32, device=device)
    pack_rows = pack_rows_kernel(in_features)
    if in_place:
        values = (
            weight.view(torch.uint8)
            .view(-1)[: out_features * width]
            .view(out_features, width)
        )
        staging = torch.empty(
            (min(_BLOCK_ROWS, out_features), width), dtype=torch.uint8, device=device
        )
        for start in range(0, out_features, _BLOCK_ROWS):
            count = min(_BLOCK_ROWS, out_features - start)
            pack_rows(
                bits,
                staging[:count],
                row_offsets,
                rows,
                entries,
                mismatches,
                base,
                start,
            )
            values[start : start + count].copy_(staging[:count])
        del staging
    else:
        values = torch.empty((out_features, width), dtype=torch.uint8, device=device)
        pack_rows(bits, values, row_offsets, rows, entries, mismatches, base, 0)
    if int(mismatches) != 0:
        raise ValueError(
            f"packed BF16 weight does not round-trip ({int(mismatches)} mismatches)"
        )
    return PackedBf16Weight(values, base, rows, entries, row_offsets, exceptions)


def unpack(packed: PackedBf16Weight) -> torch.Tensor:
    """The exact BF16 weight that ``packed`` stores (tests and inspection; it
    allocates freely)."""
    count, width = packed.values.shape
    k = width * 2 // 3
    low = packed.values[:, :k].to(torch.int32)
    codes = packed.values[:, k:].to(torch.int32).view(count, k // GROUP, 1, GROUP // 2)
    code = torch.cat((codes & 0xF, codes >> 4), dim=2).reshape(count, k)
    exponent = torch.where(code == 0, 0, code + packed.exponent_base - 1)
    bits = ((low & 0x80) << 8) | (exponent << 7) | (low & 0x7F)
    if packed.exceptions:
        entries = packed.exception_entries[: packed.exceptions]
        bits[
            packed.exception_rows[: packed.exceptions].long(), (entries >> 16).long()
        ] = entries & 0xFFFF
    return bits.to(torch.int16).view(torch.bfloat16)


__all__ = [
    "GROUP",
    "PackedBf16Weight",
    "WINDOW",
    "exception_count_kernel",
    "exponent_histogram_kernel",
    "is_supported",
    "pack",
    "pack_rows_kernel",
    "unpack",
]
