# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU identity of per-expert 2-bit UE8M0 scale codes.

A narrow expert (every row's max - min <= 3) packs four 2-bit deltas a byte,
low bits first. Expanding those bytes is the 4-bit expand, including the pad
byte when the exponent count is not a multiple of four. A wide expert stays
on ``pack_scale_codes``. The loader reads the pure module, so the test does
not import TileLang or launch a kernel.
"""

import ast
import importlib.util
from pathlib import Path

import pytest
import torch

_SCALE_CODES = (
    Path(__file__).resolve().parents[3]
    / "vllm/models/deepseek_v4_1/tilelang/scale_codes.py"
)


def _load_scale_codes():
    spec = importlib.util.spec_from_file_location("ds41_scale_codes", _SCALE_CODES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_codes = _load_scale_codes()
pack_scale_codes = _codes.pack_scale_codes
expand_scale_codes = _codes.expand_scale_codes
pack_scale_codes_2bit = _codes.pack_scale_codes_2bit
expand_scale_codes_2bit = _codes.expand_scale_codes_2bit
pack_expert_scale_codes = _codes.pack_expert_scale_codes
tile_byte_codes = _codes.tile_byte_codes
overlay_byte_codes = _codes.overlay_byte_codes
scale_word = _codes.scale_word

_BLOCK_N = 128


def _experts(k: int) -> torch.Tensor:
    """Two experts, 128 rows. Expert 0 spans 4 values; expert 1's first row spans 6."""
    n_exp = k // 32
    rows = _BLOCK_N
    base = 110
    exponents = torch.full((2, rows, n_exp), base, dtype=torch.uint8)
    exponents[0] = base + (torch.arange(n_exp) % 4).to(torch.uint8)
    exponents[1, 0, 1] = base + 5
    return exponents


@pytest.mark.parametrize("k", [512, 576])
def test_narrow_expert_round_trips_and_wide_expert_keeps_4bit_codes(k):
    """Expert 0 is 2-bit and exact, including K=576's pad. Expert 1 stays 4-bit.

    Expert 1 has one row with delta 5 and the rest at the row base. The whole
    expert stays on 4-bit codes, and those codes are ``pack_scale_codes``.
    Its 2-bit slice is zero: the delta is refused, not clipped into the byte.
    """
    exponents = _experts(k)
    n_exp = k // 32
    got = pack_expert_scale_codes(exponents)
    whole, whole_base = pack_scale_codes(exponents)
    wide_codes, wide_base = pack_scale_codes(exponents[1])

    assert got.narrow.dtype == torch.uint8
    assert got.narrow.tolist() == [1, 0]
    assert got.codes_2bit.dtype == torch.uint8
    assert got.codes_2bit.is_contiguous()
    assert torch.equal(got.codes, whole)
    assert torch.equal(got.base, whole_base)
    assert torch.equal(got.codes[1], wide_codes)
    assert torch.equal(got.base[1], wide_base)
    assert torch.equal(got.codes_2bit[1], torch.zeros_like(got.codes_2bit[1]))
    with pytest.raises(ValueError, match="2-bit codes hold 4"):
        pack_scale_codes_2bit(exponents[1])

    narrow_words = expand_scale_codes_2bit(got.codes_2bit[0], got.base[0])
    assert torch.equal(narrow_words, expand_scale_codes(got.codes[0], got.base[0]))
    raw = narrow_words.view(torch.uint8).reshape(exponents.shape[1], -1)
    assert torch.equal(raw[:, :n_exp], exponents[0])
    pad = raw[:, n_exp:]
    assert torch.equal(pad, got.base[0].unsqueeze(1).expand_as(pad))
    # Low bits first: deltas 0, 1, 2, 3 -> 0 | 1<<2 | 2<<4 | 3<<6.
    assert got.codes_2bit[0, 0, 0].item() == 228
    if k == 576:
        # 18 exponents: the last byte is deltas 0, 1 and two pad zeros.
        assert n_exp == 18
        assert got.codes_2bit[0, 0, -1].item() == (0 | (1 << 2))
        assert pad.shape[1] == 2

    tiled = tile_byte_codes(got.codes_2bit, _BLOCK_N)
    words = got.codes_2bit.shape[-1]
    assert tiled.dtype == torch.uint8
    assert tiled.is_contiguous()
    assert tiled.stride(-1) == 1
    assert tiled.numel() == got.codes.numel()
    assert tiled.nbytes * 2 == got.codes.nbytes
    narrow_tile = tiled[:words]
    restored = narrow_tile.reshape(words, _BLOCK_N).permute(1, 0).contiguous()
    assert torch.equal(restored, got.codes_2bit[0])
    assert torch.equal(tiled[words:], torch.zeros_like(tiled[words:]))


@pytest.mark.parametrize("k", [128, 192, 256, 512, 576, 640])
def test_code_words_match_the_gemm_k_blocks(k):
    """One code word per K block the grouped GEMM loads, including a 64-wide tail.

    The kernel's block count is ``K // 128`` plus one when ``K % 128 == 64``.
    The packer's last dimension is that count, so a tile row
    ``(expert * n_blocks + bx) * k_blocks + ko`` is one K block's codes.
    """
    exponents = torch.full((1, 32, k // 32), 100, dtype=torch.uint8)
    packed = pack_expert_scale_codes(exponents)
    k_blocks = k // 128 + (k % 128 == 64)
    assert packed.codes.shape[-1] == k_blocks
    assert packed.codes_2bit.shape[-1] == k_blocks
    assert packed.narrow.tolist() == [1]


def test_two_bit_pack_refuses_a_delta_above_3():
    """Delta 4 spans 5 values. The 2-bit pack raises instead of keeping the low bits."""
    exponents = torch.tensor([[[10, 14, 10, 10]]], dtype=torch.uint8)
    with pytest.raises(ValueError, match="2-bit codes hold 4"):
        pack_scale_codes_2bit(exponents)
    packed = pack_expert_scale_codes(exponents)
    assert packed.narrow.tolist() == [0]
    assert torch.equal(packed.codes, pack_scale_codes(exponents)[0])
    assert torch.equal(packed.codes_2bit, torch.zeros_like(packed.codes_2bit))


def test_a_row_spanning_more_than_16_values_is_refused():
    """K=576. Delta 16 spans 17 values and still raises, as ``pack_scale_codes`` does."""
    exponents = torch.full((2, 4, 18), 100, dtype=torch.uint8)
    exponents[1, 2, 3] = 116
    with pytest.raises(ValueError, match="4-bit codes hold 16"):
        pack_expert_scale_codes(exponents)
    with pytest.raises(ValueError, match="4-bit codes hold 16"):
        pack_scale_codes(exponents)


def test_kernel_row_index_selects_that_experts_codes():
    """``(expert * n_blocks + bx) * words + ko`` is that tile's block_N codes.

    K=576 is 18 exponents, padded to 5 words, the tail shape. A wide expert's
    2-bit slice stays zero, so its kernel row must not be read as codes.
    """
    gen = torch.Generator().manual_seed(4)
    exponents = torch.randint(90, 94, (2, 256, 18), generator=gen, dtype=torch.uint8)
    exponents[1, 0, 0] = exponents[1, 0, 0] + 8  # one wide expert
    packed = pack_expert_scale_codes(exponents)
    assert packed.narrow.tolist() == [1, 0]
    tiled = tile_byte_codes(packed.codes_2bit, _BLOCK_N)
    n_blocks, words = 256 // _BLOCK_N, packed.codes_2bit.shape[-1]
    assert words == 5
    for expert, bx, ko in ((0, 0, 0), (0, 1, 4), (1, 0, 3)):
        row = (expert * n_blocks + bx) * words + ko
        expect = packed.codes_2bit[expert, bx * _BLOCK_N : (bx + 1) * _BLOCK_N, ko]
        assert torch.equal(tiled[row], expect)
    assert torch.equal(
        tiled[n_blocks * words :], torch.zeros_like(tiled[n_blocks * words :])
    )


def test_the_moe_fixture_range_keeps_one_narrow_and_one_wide_expert():
    """The GPU expert test draws 118..121, then pins expert 1's first row at
    118 and 123. That expert is wide for any other draw; the rest stay narrow.
    """
    gen = torch.Generator().manual_seed(1)
    exps = torch.randint(118, 122, (4, 8, 16), generator=gen)
    exps[1, 0, 0] = 118
    exps[1, 0, 1] = 118 + 5
    packed = pack_expert_scale_codes(exps.to(torch.uint8))
    assert packed.narrow.tolist() == [1, 0, 1, 1]
    # Every seed: the pinned row spans 6 values, so expert 1 cannot be narrow.
    for seed in range(20):
        draw = torch.randint(
            118, 122, (2, 4, 8), generator=torch.Generator().manual_seed(seed)
        )
        draw[1, 0, 0] = 118
        draw[1, 0, 1] = 118 + 5
        got = pack_expert_scale_codes(draw.to(torch.uint8))
        assert got.narrow.tolist() == [1, 0], seed


def test_both_code_layouts_share_the_gemm_row():
    """One row index, ``(expert * n_blocks + bx) * words + ko``, is that tile
    in both layouts. A wide expert's 2-bit slice is zeros; its 4-bit slice is
    not, so the GEMM must not read one layout with the other's row.
    """
    import ast

    moe = (
        Path(__file__).resolve().parents[3]
        / "vllm/models/deepseek_v4_1/tilelang/moe.py"
    )
    src = moe.read_text()
    tree = ast.parse(src)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "tile_codes"
    )
    ns = {"torch": torch}
    exec(ast.get_source_segment(src, fn), ns)
    tile_codes = ns["tile_codes"]
    gen = torch.Generator().manual_seed(0)
    exps = torch.randint(90, 94, (3, 256, 18), generator=gen, dtype=torch.uint8)
    exps[1, 0, 0] = 90
    exps[1, 0, 1] = 95
    packed = pack_expert_scale_codes(exps)
    assert packed.narrow.tolist() == [1, 0, 1]
    narrow = tile_byte_codes(packed.codes_2bit, _BLOCK_N)
    wide = tile_codes(packed.codes, _BLOCK_N)
    n_blocks, words = 256 // _BLOCK_N, packed.codes.shape[-1]
    assert words == 5
    for expert, bx, ko in ((0, 0, 0), (0, 1, 4), (1, 0, 3), (2, 1, 2)):
        row = (expert * n_blocks + bx) * words + ko
        logical = slice(bx * _BLOCK_N, (bx + 1) * _BLOCK_N)
        assert torch.equal(narrow[row], packed.codes_2bit[expert, logical, ko])
        assert torch.equal(wide[row], packed.codes[expert, logical, ko])
    wide_rows = slice(n_blocks * words, 2 * n_blocks * words)
    assert torch.equal(narrow[wide_rows], torch.zeros_like(narrow[wide_rows]))
    assert not torch.equal(wide[wide_rows], torch.zeros_like(wide[wide_rows]))


def test_the_gemm_scale_word_matches_both_expands():
    """The word the kernel writes is the packer's word, for a narrow and a wide code."""
    narrow_code, base = 0b11100100, 100  # deltas 0, 1, 2, 3
    wide_code = 0x1234  # nibbles 4, 3, 2, 1
    got = scale_word(narrow_code, base, True)
    ref = expand_scale_codes_2bit(
        torch.tensor([narrow_code], dtype=torch.uint8), torch.tensor([base])
    )
    assert got == int(ref.view(torch.int32))
    got = scale_word(wide_code, base, False)
    ref = expand_scale_codes(
        torch.tensor([wide_code], dtype=torch.uint16), torch.tensor([base])
    )
    assert got == int(ref.view(torch.int32))
    # Deltas sit in the low bits of each byte. base 252 plus 0,1,2,3 is
    # 252..255: the sum fills the byte and must not carry into the next.
    top = scale_word(0b11100100, 252, True)
    assert top == 0xFFFEFDFC
    assert scale_word(0, 255, True) == 0xFFFFFFFF
    # The same byte read as 4-bit nibbles is a different word. The GEMM
    # must store the narrow expand inside the narrow branch.
    assert scale_word(0b11100100, 100, True) != scale_word(0b11100100, 100, False)


def test_apply_passes_narrow_codes_in_the_kernel_argument_order():
    """The MoE launch tuple is the narrow GEMM's arguments, tail included.

    ``(tiles, tail, 4-bit codes, 2-bit codes, narrow, base)``. ``_present``
    drops a missing tail, which is the no-tail kernel's argument list.
    """
    import ast

    src = (
        Path(__file__).resolve().parents[3]
        / "vllm/model_executor/layers/fused_moe/tilelang.py"
    ).read_text()
    tree = ast.parse(src)
    tuples = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 6:
            continue
        text = ast.unparse(node)
        if "self.w1_scale" in text and "narrow13" in text:
            tuples.append(text)
    assert tuples, "apply() does not pass the 2-bit codes"
    assert tuples[0] == ("(w1, tail13, self.w1_scale, byte13, narrow13, base13)")


def _words_each_thread_writes(codes, base, narrow, block_n, threads=128):
    """The uint32 words the GEMM's per-thread expand stores, one per row and K block.

    A narrow expert: threads ``tx < block_N // 16`` each expand rows
    ``tx * 16 + i``. A wide expert: threads ``tx < block_N // 8`` each expand
    rows ``tx * 8 + i``. Every row is written once.
    """
    rows, words = codes.shape
    assert rows % block_n == 0
    out = torch.empty(rows, words, dtype=torch.int64)
    seen = torch.zeros(rows, dtype=torch.int32)
    tiles = rows // block_n
    for tile in range(tiles):
        origin = tile * block_n
        if narrow:
            for tx in range(threads):
                if tx >= block_n // 16:
                    continue
                for i in range(16):
                    row = origin + tx * 16 + i
                    seen[row] += 1
                    for ko in range(words):
                        out[row, ko] = scale_word(
                            int(codes[row, ko]), int(base[row]), True
                        )
        else:
            for tx in range(threads):
                if tx >= block_n // 8:
                    continue
                for i in range(8):
                    row = origin + tx * 8 + i
                    seen[row] += 1
                    for ko in range(words):
                        out[row, ko] = scale_word(
                            int(codes[row, ko]), int(base[row]), False
                        )
    assert torch.equal(seen, torch.ones(rows, dtype=torch.int32))
    return out


@pytest.mark.parametrize("block_n", [64, 128])
def test_each_thread_expands_its_own_rows_to_the_packer_words(block_n):
    """The thread that loads a code expands that row, and the word matches the packer.

    Bases differ by row, so a shared base would not match. K=576 keeps the
    padded last word. Expert 0 is narrow; expert 1 is wide.
    """
    gen = torch.Generator().manual_seed(7)
    exps = torch.randint(100, 104, (2, 256, 18), generator=gen, dtype=torch.uint8)
    exps[1, 0, 0] = 100
    exps[1, 0, 1] = 106
    packed = pack_expert_scale_codes(exps)
    assert packed.narrow.tolist() == [1, 0]
    narrow_words = _words_each_thread_writes(
        packed.codes_2bit[0], packed.base[0], True, block_n
    )
    wide_words = _words_each_thread_writes(
        packed.codes[1], packed.base[1], False, block_n
    )
    narrow_ref = expand_scale_codes_2bit(packed.codes_2bit[0], packed.base[0])
    wide_ref = expand_scale_codes(packed.codes[1], packed.base[1])
    assert torch.equal(
        narrow_words & 0xFFFFFFFF,
        narrow_ref.view(torch.int32).to(torch.int64) & 0xFFFFFFFF,
    )
    assert torch.equal(
        wide_words & 0xFFFFFFFF, wide_ref.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    )
    # The tiled load the kernel issues is those same rows, in thread order.
    tiled = tile_byte_codes(packed.codes_2bit[:1], block_n)
    words = packed.codes_2bit.shape[-1]
    for bx, ko in ((0, 0), (1, words - 1)):
        row = bx * words + ko
        for tx in range(block_n // 16):
            got = tiled[row, tx * 16 : tx * 16 + 16]
            expect = packed.codes_2bit[
                0, bx * block_n + tx * 16 : bx * block_n + tx * 16 + 16, ko
            ]
            assert torch.equal(got, expect), (bx, ko, tx)


def _moe_functions(*names: str):
    moe = (
        Path(__file__).resolve().parents[3]
        / "vllm/models/deepseek_v4_1/tilelang/moe.py"
    )
    src = moe.read_text()
    tree = ast.parse(src)
    ns = {"torch": torch, "FP4_TMA_K": 128}
    wanted = set(names)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(ast.get_source_segment(src, node), ns)
    missing = [name for name in names if name not in ns]
    assert not missing, missing
    return tuple(ns[name] for name in names)


@pytest.mark.parametrize("n, k", [(512, 512), (384, 576)])
def test_the_mixed_launch_shapes_match_the_narrow_kernel(n, k):
    """The GPU bit-match's tensors are the narrow GEMM's arguments.

    K=512 has no tail. K=576 keeps a 64-wide tail. A mismatch here fails the
    launch before any multiply.
    """
    tile_expert_weights, tile_codes = _moe_functions(
        "tile_expert_weights", "tile_codes"
    )
    experts, block_n = 32, 128
    gen = torch.Generator().manual_seed(41)
    exps = torch.randint(
        100, 104, (experts, n, k // 32), generator=gen, dtype=torch.uint8
    )
    exps[1, 0, 0] = 100
    exps[1, 0, 1] = 106
    packed = pack_expert_scale_codes(exps)
    assert packed.narrow[1].item() == 0
    weights = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8)
    tiles, tails = tile_expert_weights(weights.view(torch.int8), block_n)
    byte_codes = tile_byte_codes(packed.codes_2bit, block_n)
    wide_codes = tile_codes(packed.codes, block_n)
    k_full, tail = divmod(k, 128)
    k_blocks = k_full + (tail == 64)
    n_blocks = n // block_n
    rows = experts * n_blocks * k_blocks
    assert tiles.shape == (experts * n_blocks * k_full, block_n, 64)
    if tail == 0:
        assert tails is None
    else:
        assert tails.shape == (experts * n_blocks, block_n, tail // 2)
    assert wide_codes.shape == (rows, block_n) and wide_codes.dtype == torch.uint16
    assert byte_codes.shape == (rows, block_n) and byte_codes.dtype == torch.uint8
    codes, byte_codes = overlay_byte_codes(wide_codes, byte_codes, packed.narrow)
    units = int((2 - packed.narrow.to(torch.int64)).sum().item())
    code_rows = units * n_blocks * k_blocks
    assert codes.shape == (code_rows, block_n // 2) and codes.dtype == torch.uint16
    assert byte_codes.shape == (code_rows, block_n) and byte_codes.dtype == torch.uint8
    assert packed.narrow.shape == (experts,) and packed.base.shape == (experts, n)


def test_one_buffer_holds_each_experts_one_form():
    """Each expert keeps only its own code width in one compact buffer.

    ``overlay_byte_codes`` returns uint16 and uint8 views of one storage where a
    narrow expert takes one ``[per, block_N]``-byte unit (its ``tile_byte_codes``
    rows) and a wide expert takes two units (its ``tile_codes`` rows), starting
    at unit ``2 * e - narrow[:e].sum()``. K=576 has the padded fifth word.
    """
    (tile_codes,) = _moe_functions("tile_codes")
    gen = torch.Generator().manual_seed(5)
    exps = torch.randint(90, 94, (3, 256, 18), generator=gen, dtype=torch.uint8)
    exps[1, 0, 0] = 90
    exps[1, 0, 1] = 95
    packed = pack_expert_scale_codes(exps)
    assert packed.narrow.tolist() == [1, 0, 1]
    wide = tile_codes(packed.codes, _BLOCK_N)
    narrow = tile_byte_codes(packed.codes_2bit, _BLOCK_N)
    codes, byte_codes = overlay_byte_codes(wide.clone(), narrow, packed.narrow)
    per = wide.shape[0] // 3
    assert codes.dtype == torch.uint16 and codes.shape == (4 * per, _BLOCK_N // 2)
    assert byte_codes.dtype == torch.uint8
    assert byte_codes.shape == (4 * per, _BLOCK_N)
    assert byte_codes.data_ptr() == codes.data_ptr()
    assert codes.untyped_storage().nbytes() == 4 * per * _BLOCK_N
    for expert in range(3):
        src = slice(expert * per, (expert + 1) * per)
        slot = 2 * expert - int(packed.narrow[:expert].sum().item())
        if packed.narrow[expert]:
            assert torch.equal(byte_codes[slot * per : (slot + 1) * per], narrow[src])
        else:
            got = codes[slot * per : (slot + 2) * per].view(per, _BLOCK_N)
            assert torch.equal(got, wide[src])


def test_an_all_wide_tensor_keeps_its_4bit_codes():
    """With no narrow expert the overlay keeps all 4-bit codes without copying."""
    (tile_codes,) = _moe_functions("tile_codes")
    exps = torch.full((2, 128, 16), 100, dtype=torch.uint8)
    exps[:, 0, 0] = 108
    packed = pack_expert_scale_codes(exps)
    assert packed.narrow.tolist() == [0, 0]
    wide = tile_codes(packed.codes, _BLOCK_N)
    narrow = tile_byte_codes(packed.codes_2bit, _BLOCK_N)
    codes, _ = overlay_byte_codes(wide.clone(), narrow, packed.narrow)
    assert torch.equal(codes.view_as(wide), wide)


def test_an_all_narrow_tensor_uses_half_the_4bit_bytes():
    """With every expert narrow the compact buffer takes half the 4-bit bytes."""
    (tile_codes,) = _moe_functions("tile_codes")
    gen = torch.Generator().manual_seed(9)
    exps = torch.randint(90, 94, (4, 128, 18), generator=gen, dtype=torch.uint8)
    packed = pack_expert_scale_codes(exps)
    assert packed.narrow.tolist() == [1, 1, 1, 1]
    wide = tile_codes(packed.codes, _BLOCK_N)
    narrow = tile_byte_codes(packed.codes_2bit, _BLOCK_N)
    codes, byte_codes = overlay_byte_codes(wide.clone(), narrow, packed.narrow)
    assert codes.untyped_storage().nbytes() * 2 == wide.nbytes
    assert torch.equal(byte_codes, narrow)


def test_the_narrow_kernel_reads_2bit_codes_as_double_width_rows():
    """The GEMM's 2-bit spec is the compact ``[code_rows, block_N]`` byte view."""
    src = (
        Path(__file__).resolve().parents[3]
        / "vllm/models/deepseek_v4_1/tilelang/moe.py"
    ).read_text()
    spec = "T.Tensor((code_rows, block_N), T.uint8)"
    assert f"sfb8_spec = {spec}" in src
    assert "overlay_byte_codes(" in src


def test_the_tail_pad_byte_is_the_row_base_and_the_real_exponents_match():
    """K=576's last word has two real exponents and two pad bytes.

    The uncoded scale pack pads those bytes with 127. The coded expand stores the
    row base there (delta 0). The 18 real exponents match. A 64-wide tail
    reads the two real exponents; the bit-match depends on the MMA not
    reading the pad.
    """
    scale_pack = (
        Path(__file__).resolve().parents[3]
        / "vllm/models/deepseek_v4_1/tilelang/scale_pack.py"
    ).read_text()
    assert "value=127" in scale_pack
    gen = torch.Generator().manual_seed(3)
    base = torch.randint(100, 120, (4, 32, 1), generator=gen)
    exps = (base + torch.randint(0, 4, (4, 32, 18), generator=gen)).to(torch.uint8)
    exps[1, 0, 0] = 100
    exps[1, 0, 1] = 110
    packed = pack_expert_scale_codes(exps)
    assert packed.narrow.tolist() == [1, 0, 1, 1]
    narrow = expand_scale_codes_2bit(packed.codes_2bit, packed.base)
    wide = expand_scale_codes(packed.codes, packed.base)
    got = torch.where(
        packed.narrow.view(-1, 1, 1).bool(),
        narrow.view(torch.uint8).reshape(4, 32, -1),
        wide.view(torch.uint8).reshape(4, 32, -1),
    )
    raw = exps
    pad = -raw.shape[-1] % 4
    ref = torch.nn.functional.pad(raw, (0, pad), value=127)
    assert torch.equal(got[..., :18], ref[..., :18])
    assert torch.equal(
        got[..., 18:], packed.base.unsqueeze(-1).expand_as(got[..., 18:])
    )
    assert not torch.equal(got[..., 18:], ref[..., 18:])
    # FP8 x unpacked E2M1 is mxf8f6f4: one byte per K32 atom. A 64-wide tail
    # at k_start 0 reads bytes 0 and 1 of the word. Bytes 2 and 3 are the pad.
    mma = (
        Path(__file__).resolve().parents[4]
        / "tilelang/tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py"
    )
    if mma.is_file():
        src = mma.read_text()
        assert (
            "return (_k_start + self.micro_size_k * ki) // int(sf_granularity_k) % 4"
            in src
        )
        assert '("mxf8f6f4", 1, "ue8m0")' in src
        assert "atom_k=32" in src
    bytes_read = [(0 + 32 * ki) // 32 % 4 for ki in range(2)]
    assert bytes_read == [0, 1]


def test_the_narrow_scale_store_is_inside_the_device_branch():
    """A name bound in a device if does not exist after it.

    The 2-bit store has to be a statement of ``if Narrow[expert] != 0``, not a
    store after that if. Otherwise the store keeps the 4-bit formula.
    """
    import ast

    src = (
        Path(__file__).resolve().parents[3]
        / "vllm/models/deepseek_v4_1/tilelang/moe.py"
    ).read_text()
    tree = ast.parse(src)

    def is_narrow_test(test: ast.AST) -> bool:
        return isinstance(test, ast.Compare) and "Narrow" in ast.unparse(test)

    def mentions(nodes: list[ast.AST], name: str) -> bool:
        for node in nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id == name:
                    return True
        return False

    found = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If) and is_narrow_test(node.test)):
            continue
        assert not mentions(node.orelse, "code8")
        if not (mentions(node.body, "code8") and mentions(node.body, "SFB_shared")):
            continue
        # Each thread expands its own row; the 2-bit store is a statement of the
        # expert test itself.
        stores = [
            child
            for child in node.body
            if isinstance(child, ast.Assign)
            and "SFB_shared" in ast.unparse(child.targets[0])
        ]
        assert stores, "the 2-bit store is a statement of the expert test"
        assert all("code8" in ast.unparse(store.value) for store in stores)
        found = True
    assert found


def test_process_weights_replaces_w13_before_preparing_w2(monkeypatch):
    """``TileLangExperts.process_weights_after_loading`` releases ``w13_weight``
    and ``w13_weight_scale`` before preparing ``w2_weight`` so both raw and tiled
    projections are never held live simultaneously."""
    import weakref
    from types import SimpleNamespace
    from vllm.model_executor.layers.fused_moe import tilelang as tl_moe

    refs = {}
    w13 = torch.ones((2, 8, 4), dtype=torch.int8)
    s13 = torch.ones((2, 8, 2), dtype=torch.uint8)
    w2 = torch.ones((2, 4, 4), dtype=torch.int8)
    s2 = torch.ones((2, 4, 2), dtype=torch.uint8)
    refs["w13"] = weakref.ref(w13)
    refs["s13"] = weakref.ref(s13)
    calls = []

    class _Plan:
        def __init__(self, *args, **kwargs):
            pass

        def prepare_weight(self, w, scale):
            calls.append(len(calls))
            if len(calls) == 2:
                # Raw w13_weight and w13_weight_scale must already be released
                # before w2_weight is prepared.
                assert refs["w13"]() is None
                assert refs["s13"]() is None
            return (
                torch.zeros_like(w),
                None,
                torch.zeros((2, 2), dtype=torch.uint16),
                torch.zeros((2, 4), dtype=torch.uint8),
                torch.ones((2,), dtype=torch.uint8),
                torch.zeros((2, 2), dtype=torch.uint8),
            )

    monkeypatch.setattr(tl_moe, "_kernels", lambda: SimpleNamespace(MoeExpertsV41Plan=_Plan))
    monkeypatch.setattr(tl_moe, "replace_parameter", lambda layer, name, t: setattr(layer, name, t))
    layer = SimpleNamespace(
        apply_router_weight_on_input=False,
        w13_weight=w13,
        w13_weight_scale=s13,
        w2_weight=w2,
        w2_weight_scale=s2,
    )
    del w13, s13, w2, s2
    experts = tl_moe.TileLangExperts.__new__(tl_moe.TileLangExperts)
    experts.quant_config = SimpleNamespace(
        gemm1_clamp_limit=None,
        _w1=SimpleNamespace(scale=None),
        _w2=SimpleNamespace(scale=None),
    )
    experts.moe_config = SimpleNamespace(
        experts_per_token=2, swiglu_limit=7.0, in_dtype=torch.bfloat16
    )
    experts.fold_shared = True
    experts.process_weights_after_loading(layer)
    assert calls == [0, 1]
