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
    e = exponents.view(torch.uint8)
    base = e.amin(dim=-1)
    deltas = e.to(torch.int32) - base.unsqueeze(-1).to(torch.int32)
    span = int(deltas.max()) if deltas.numel() else 0
    if span > 15:
        raise ValueError(
            f"a row's scale exponents span {span + 1} values; 4-bit codes hold 16"
        )
    pad = -e.shape[-1] % 4
    if pad:
        deltas = torch.nn.functional.pad(deltas, (0, pad), value=0)
    d = deltas.view(*deltas.shape[:-1], -1, 4)
    codes = d[..., 0] | (d[..., 1] << 4) | (d[..., 2] << 8) | (d[..., 3] << 12)
    return codes.to(torch.int16).view(torch.uint16).contiguous(), base.contiguous()


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
    e = exponents.view(torch.uint8)
    base = e.amin(dim=-1)
    deltas = e.to(torch.int32) - base.unsqueeze(-1).to(torch.int32)
    span = int(deltas.max()) if deltas.numel() else 0
    if span > 3:
        raise ValueError(
            f"a row's scale exponents span {span + 1} values; 2-bit codes hold 4"
        )
    pad = -e.shape[-1] % 4
    if pad:
        deltas = torch.nn.functional.pad(deltas, (0, pad), value=0)
    d = deltas.view(*deltas.shape[:-1], -1, 4)
    codes = d[..., 0] | (d[..., 1] << 2) | (d[..., 2] << 4) | (d[..., 3] << 6)
    return codes.to(torch.uint8).contiguous(), base.contiguous()


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
    codes, base = pack_scale_codes(e)
    deltas = e.to(torch.int32) - base.unsqueeze(-1).to(torch.int32)
    row_delta = deltas.amax(dim=-1)
    if row_delta.ndim == 1:
        expert_delta = row_delta
    else:
        expert_delta = row_delta.reshape(row_delta.shape[0], -1).amax(dim=1)
    narrow = (expert_delta <= 3).to(torch.uint8)
    codes_2bit = torch.zeros(codes.shape, dtype=torch.uint8, device=codes.device)
    mask = narrow.bool()
    if mask.any().item():
        packed, _packed_base = pack_scale_codes_2bit(e[mask])
        codes_2bit[mask] = packed
    return ExpertScaleCodes(
        narrow.contiguous(),
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
