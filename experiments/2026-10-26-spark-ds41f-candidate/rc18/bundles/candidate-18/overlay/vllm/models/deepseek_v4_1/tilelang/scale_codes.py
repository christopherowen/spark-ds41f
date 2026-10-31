# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UE8M0 scale codes for the routed experts.

4-bit codes are a nibble per exponent, low nibble first, one uint16 per four
exponents. 2-bit codes are the same spread with two-bit fields, four a byte,
low bits first. This module is torch only, so a CPU test can load it without
TileLang.
"""

from typing import NamedTuple

import torch


class ExpertScaleCodes(NamedTuple):
    """One tensor of experts, each expert entirely 2-bit or entirely 4-bit.

    ``narrow`` is uint8 ``[E]``: 1 when every row of that expert has
    ``max - min <= 3``, else 0. ``codes`` and ``base`` are ``pack_scale_codes``
    of the whole tensor, so a wide expert's codes are those bytes.
    ``codes_2bit`` is contiguous uint8, four deltas a byte; a wide expert's
    slice is zero (not a clipped 2-bit code).
    """

    narrow: torch.Tensor
    base: torch.Tensor
    codes: torch.Tensor
    codes_2bit: torch.Tensor


def _row_deltas(
    exponents: torch.Tensor, max_delta: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row minimum ``base`` and padded 4-tuples of uint8 deltas ``e - base``."""
    e = exponents.view(torch.uint8)
    base = e.amin(dim=-1).contiguous()
    deltas = e - base.unsqueeze(-1)
    span = int(deltas.max()) if deltas.numel() else 0
    if span > max_delta:
        bits = 2 if max_delta == 3 else 4
        raise ValueError(
            f"a row's scale exponents span {span + 1} values; {bits}-bit codes hold {max_delta + 1}"
        )
    pad = -e.shape[-1] % 4
    if pad:
        deltas = torch.nn.functional.pad(deltas, (0, pad), value=0)
    return deltas.view(*deltas.shape[:-1], -1, 4), base


def _pack_nibbles(d: torch.Tensor) -> torch.Tensor:
    lo = (d[..., 0] | (d[..., 1] << 4)).to(torch.int16)
    hi = (d[..., 2] | (d[..., 3] << 4)).to(torch.int16)
    return (lo | (hi << 8)).view(torch.uint16).contiguous()


def _pack_crumbs(d: torch.Tensor) -> torch.Tensor:
    return (
        d[..., 0] | (d[..., 1] << 2) | (d[..., 2] << 4) | (d[..., 3] << 6)
    ).contiguous()


def pack_scale_codes(exponents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., K / 32]`` UE8M0 bytes as a base exponent per row and 4-bit codes.

    Returns ``(codes, base)``: ``codes[..., ceil(K / 128)]`` (uint16) holds, for each
    word of ``pack_scale_words``, its four exponents minus the row's ``base[...]``
    (uint8, the row's smallest exponent), a nibble each, low nibble first. Spreading a
    code's nibbles into bytes and adding ``base`` to each gives that word byte for byte
    (``expand_scale_codes``); padding exponents (rows whose count is not a multiple of
    four) code as 0 and are unread. The routed experts' scales span at most 8 values
    in a row (DeepSeek-V4.1-Flash, both checkpoints sampled), so the codes are exact;
    a row spanning more than 16 is refused rather than approximated.
    """
    d, base = _row_deltas(exponents, 15)
    return _pack_nibbles(d), base


def expand_scale_codes(codes: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """The uint32 words ``pack_scale_codes`` encodes; padding bytes read as ``base``."""
    c = codes.view(torch.int16).to(torch.int32) & 0xFFFF
    spread = (
        (c & 0xF)
        | ((c >> 4) & 0xF) << 8
        | ((c >> 8) & 0xF) << 16
        | ((c >> 12) & 0xF) << 24
    )
    words = spread + base.to(torch.int32).unsqueeze(-1) * 0x01010101
    return words.to(torch.int32).view(torch.uint32)


def pack_scale_codes_2bit(exponents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., K / 32]`` UE8M0 bytes as a base exponent per row and 2-bit codes.

    Returns ``(codes, base)``: ``codes[..., ceil(count / 4)]`` (uint8) holds the same
    four exponents as one ``pack_scale_codes`` word, two bits each, low bits first.
    A row whose delta exceeds 3 is refused, not clipped. Padding deltas (a count not
    a multiple of four) are 0, so those bytes expand to ``base``.
    """
    d, base = _row_deltas(exponents, 3)
    return _pack_crumbs(d), base


def expand_scale_codes_2bit(codes: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """The uint32 words ``pack_scale_codes_2bit`` encodes; padding bytes read as ``base``.

    The same spread-and-add as ``expand_scale_codes``, with shifts of 2. On rows
    whose delta fits in 2 bits, the words equal ``expand_scale_codes`` of the 4-bit pack.
    """
    c = codes.view(torch.int8).to(torch.int32) & 0xFF
    spread = (
        (c & 0x3)
        | ((c >> 2) & 0x3) << 8
        | ((c >> 4) & 0x3) << 16
        | ((c >> 6) & 0x3) << 24
    )
    words = spread + base.to(torch.int32).unsqueeze(-1) * 0x01010101
    return words.to(torch.int32).view(torch.uint32)


def pack_expert_scale_codes(exponents: torch.Tensor) -> ExpertScaleCodes:
    """Choose 2-bit or 4-bit codes per expert.

    ``exponents`` is ``[E, rows, K / 32]`` (or ``[E, K / 32]``, one row). An expert
    whose every row has delta <= 3 is narrow: flag 1 and 2-bit codes. Any wider
    row keeps that expert on 4-bit codes, flag 0; the other rows of that expert
    stay 4-bit too. A row with delta > 15 raises, as ``pack_scale_codes`` does.
    """
    e = exponents.view(torch.uint8)
    if e.ndim < 2:
        raise ValueError("scale exponents must be [experts, rows, K / 32]")
    d, base = _row_deltas(e, 15)
    expert_delta = d.reshape(d.shape[0], -1).amax(dim=1)
    narrow = (expert_delta <= 3).to(torch.uint8).contiguous()
    codes = _pack_nibbles(d)
    codes_2bit = torch.zeros(codes.shape, dtype=torch.uint8, device=codes.device)
    mask = narrow.bool()
    if mask.any().item():
        codes_2bit[mask] = _pack_crumbs(d[mask])
    return ExpertScaleCodes(
        narrow,
        base,
        codes,
        codes_2bit.contiguous(),
    )


def scale_word(code: int, base: int, narrow: bool) -> int:
    """The uint32 scale word the coded GEMM writes for one row of one K block.

    ``narrow`` uses the 2-bit fields (shifts of 2). Otherwise the fields are
    the 4-bit nibbles. Both add the row base to every byte.
    """
    code &= 0xFF if narrow else 0xFFFF
    if narrow:
        spread = (
            (code & 0x3)
            | ((code >> 2) & 0x3) << 8
            | ((code >> 4) & 0x3) << 16
            | ((code >> 6) & 0x3) << 24
        )
    else:
        spread = (
            (code & 0xF)
            | ((code >> 4) & 0xF) << 8
            | ((code >> 8) & 0xF) << 16
            | ((code >> 12) & 0xF) << 24
        )
    return (spread + (base * 0x01010101)) & 0xFFFFFFFF


def tile_byte_codes(codes: torch.Tensor, block_N: int) -> torch.Tensor:
    """``[E, N, words]`` uint8 codes in the coded GEMM's tile order.

    The shape matches ``tile_codes``: ``[E * N / block_N * words, block_N]``, each
    tile's codes for one K block contiguous. The dtype stays uint8, so a narrow
    expert's bytes are contiguous. Packing the same fields into the low byte of a
    uint16 would leave those bytes strided by two, and a half-width load would
    not be half the traffic.
    """
    if codes.dtype != torch.uint8:
        raise TypeError(f"2-bit scale codes must be uint8, got {codes.dtype}")
    codes = codes.contiguous()
    experts, n, words = codes.shape
    if n % block_N:
        raise ValueError(f"N={n} is not a multiple of block_N={block_N}")
    return (
        codes.view(torch.int8)
        .view(experts, n // block_N, block_N, words)
        .permute(0, 1, 3, 2)
        .reshape(-1, block_N)
        .contiguous()
        .view(torch.uint8)
    )


def overlay_byte_codes(
    codes: torch.Tensor, byte_codes: torch.Tensor, narrow: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One compact buffer for both widths: ``(codes, bytes)`` over the same storage.

    ``codes`` is ``tile_codes``' uint16 ``[rows, block_N]``, ``byte_codes`` is
    ``tile_byte_codes``' uint8 ``[rows, block_N]`` and ``narrow`` the per-expert
    flags; each expert's ``per = rows // experts`` rows are contiguous in both.
    A narrow expert takes one ``[per, block_N]``-byte unit (its 2-bit codes); a
    wide expert takes two units (its ``[per, block_N]`` uint16 4-bit codes as
    ``[2 * per, block_N]`` bytes), starting at unit ``2 * e - narrow[:e].sum()``.
    ``bytes`` is that storage as uint8 ``[units * per, block_N]`` and ``codes``
    is ``bytes.view(torch.uint16)`` (``[units * per, block_N // 2]``), so narrow
    experts take half the bytes of wide ones and no expert holds both forms.
    """
    if codes.dtype != torch.uint16 or byte_codes.dtype != torch.uint8:
        raise TypeError("expected uint16 4-bit codes and uint8 2-bit codes")
    if codes.shape != byte_codes.shape or not codes.is_contiguous():
        raise ValueError("the two code layouts must be the same contiguous rows")
    rows, block_N = codes.shape
    if block_N % 2:
        raise ValueError(f"block_N={block_N} must be even")
    experts = narrow.numel()
    if rows % experts:
        raise ValueError(f"{rows} code rows do not split into {experts} experts")
    mask = narrow.bool()
    if not mask.any().item():
        compact = codes.view(2 * rows, block_N // 2)
        return compact, compact.view(torch.uint8)
    per = rows // experts
    units = codes.view(torch.uint8).view(experts, 2, per, block_N)
    units[mask, 0] = byte_codes.view(experts, per, block_N)[mask]
    keep = torch.ones((experts, 2), dtype=torch.bool, device=codes.device)
    keep[mask, 1] = False
    compact_bytes = units[keep].reshape(-1, block_N).contiguous()
    return compact_bytes.view(torch.uint16), compact_bytes
