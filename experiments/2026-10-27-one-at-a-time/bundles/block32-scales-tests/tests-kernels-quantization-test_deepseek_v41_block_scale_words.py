# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU identity of the packed 32x32 block-scale grid.

``pack_scale_words(grid, rows=N)`` repeats each block row onto 32 output rows.
The MXFP8 GEMM stores ``pack_scale_words(grid)`` and loads ``SFB[row // 32]``.
The loader below reads that function from its file, so the test does not import
TileLang or launch a kernel.
"""

import ast
import importlib.util
from pathlib import Path

import torch

_SCALE_PACK = (
    Path(__file__).resolve().parents[3]
    / "vllm/models/deepseek_v4_1/tilelang/scale_pack.py"
)


def _load_pack_scale_words():
    spec = importlib.util.spec_from_file_location("ds41_scale_pack", _SCALE_PACK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pack_scale_words


pack_scale_words = _load_pack_scale_words()


def test_block_scale_words_match_the_per_row_broadcast():
    """expanded[row] == block[row // 32], including the 576-wide 4-byte pad."""
    n = 128
    # 512 is a whole number of scale words. 576 is 18 exponents, padded to 20.
    for k, exponents, padded in ((512, 16, 16), (576, 18, 20)):
        gen = torch.Generator(device="cpu").manual_seed(k)
        grid = torch.randint(
            0, 256, (n // 32, k // 32), generator=gen, dtype=torch.uint8
        )
        expanded = pack_scale_words(grid, rows=n)
        block = pack_scale_words(grid)
        assert grid.shape == (n // 32, exponents)
        assert expanded.shape == (n, padded // 4)
        assert block.shape == (n // 32, padded // 4)
        assert torch.equal(expanded, block.repeat_interleave(32, dim=0))
        for row in range(n):
            assert torch.equal(expanded[row], block[row // 32]), row
        if padded != exponents:
            pad = expanded.view(torch.uint8).reshape(n, padded)[:, exponents:]
            assert torch.equal(pad, torch.full_like(pad, 127))


def _block_scale_words(block_k: int) -> int:
    """``gemm.block_scale_words``: words a K block reads, including a straddle."""
    return (block_k + block_k % 128 + 127) // 128


def _const(node):
    return node.value if isinstance(node, ast.Constant) else None


def _block_ks(node):
    """block_K integers in a config, including ``dict(block_K=...)`` calls."""
    found = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "dict"
        ):
            for kw in child.keywords:
                if kw.arg == "block_K":
                    block_k = _const(kw.value)
                    if block_k is not None:
                        found.append(block_k)
        if not isinstance(child, ast.Dict):
            continue
        for key, value in zip(child.keys, child.values):
            if _const(key) == "block_K":
                block_k = _const(value)
                if block_k is not None:
                    found.append(block_k)
    return found


def _serving_tiles():
    """(K, block_K) pairs the decode and split-K tables actually launch."""
    gemm = (
        Path(__file__).resolve().parents[3]
        / "vllm/models/deepseek_v4_1/tilelang/gemm.py"
    )
    tree = ast.parse(gemm.read_text())
    tiles = set()
    for node in tree.body:
        if not (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in ("DECODE_FP8_CONFIGS", "SPLIT_FP8")
        ):
            continue
        table = node.value
        for key, value in zip(table.keys, table.values):
            if not isinstance(key, ast.Tuple) or len(key.elts) < 2:
                continue
            k = _const(key.elts[1])
            if k is None:
                continue
            for block_k in _block_ks(value):
                tiles.add((k, block_k))
    # Shapes with no sweep entry take pick_block_K: 128, else the widest
    # multiple of 64 up to 192 that divides K. 576 is the straddle.
    for k in (64, 192, 256, 384, 512, 576, 640, 768, 1024, 1280, 2048, 4096, 5120, 6144):
        for block_k in (128, *range(192, 0, -64)):
            if k % block_k == 0:
                tiles.add((k, block_k))
                break
    return tiles


def test_every_serving_k_tile_loads_the_scale_words_it_consumes():
    """A K tile's scale words stay inside the row, with no clamp filling a gap.

    ``SFB[row // 32, min(word0 + w, sf_words - 1)]`` repeats the last word when
    the index runs past the row. That is exact only when the words the tile
    consumes are already in range. A block that straddles a 128-element boundary
    needs both words in the shared buffer.
    """
    tiles = _serving_tiles()
    # These are sweep entries, not the default 128-wide tile. A walker that
    # misses ``dict(block_K=...)`` would drop them.
    assert {(576, 64), (576, 192), (1280, 256), (5120, 256)} <= tiles
    short = []
    for k, block_k in sorted(tiles):
        assert k % block_k == 0, (k, block_k)
        sf_words = (k + 127) // 128
        block_words = _block_scale_words(block_k)
        for ko in range(k // block_k):
            start = ko * block_k
            first = start // 128
            last = (start + block_k - 1) // 128
            if last >= sf_words or last - first + 1 > block_words:
                short.append((k, block_k, ko, first, last, block_words, sf_words))
            # mxf8f6f4 reads one byte per K32 atom:
            # (k_start + 32 * ki) // 32 % 4 inside word word0.
            k_start = start % 128
            word0 = start // 128
            for ki in range(block_k // 32):
                byte = (k_start + 32 * ki) // 32 % 4
                exponent = (word0 + (k_start + 32 * ki) // 128) * 4 + byte
                if exponent >= k // 32:
                    short.append((k, block_k, ko, ki, exponent))
    assert short == []


def _sfb_indexes(path: Path) -> list[tuple[str, ast.AST]]:
    """First index of every ``SFB[...]`` load, with the function that owns it."""
    tree = ast.parse(path.read_text())
    found = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Subscript):
                continue
            if not isinstance(child.value, ast.Name) or child.value.id != "SFB":
                continue
            index = child.slice
            if isinstance(index, ast.Tuple) and index.elts:
                index = index.elts[0]
            found.append((node.name, index))
    return found


def _is_floor_div_32(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.FloorDiv)
        and isinstance(node.right, ast.Constant)
        and node.right.value == 32
    )


def test_every_sfb_load_clamps_the_row_then_divides_by_32():
    """The GEMM reads the packed grid. Clamping after the divide would treat
    ``N - 1`` as a block-row index and walk off the last block.

    Both loads name the clamped row, then take ``row // 32``. A load that
    indexes ``SFB`` by the output row itself is the old per-row broadcast.
    """
    gemm = (
        Path(__file__).resolve().parents[3]
        / "vllm/models/deepseek_v4_1/tilelang/gemm.py"
    )
    loads = _sfb_indexes(gemm)
    assert [name for name, _ in loads] == ["_mxfp8_gemm", "_mxfp8_gemm_partials"]
    tree = ast.parse(gemm.read_text())
    clamps = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Assign)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
                and child.targets[0].id == "row"
            ):
                clamps.setdefault(node.name, child.value)
    for name, index in loads:
        assert _is_floor_div_32(index), name
        assert isinstance(index.left, ast.Name) and index.left.id == "row"
        clamp = clamps[name]
        assert isinstance(clamp, ast.Call)
        func = clamp.func
        assert isinstance(func, ast.Attribute) and func.attr == "min"
        # Clamp the output row, then divide. A divide inside the clamp, or a
        # second divide of an already-blocked index, reads the wrong block row.
        assert not any(isinstance(child, ast.FloorDiv) for child in ast.walk(clamp))

    linear = (
        Path(__file__).resolve().parents[3]
        / "vllm/models/deepseek_v4_1/tilelang/linear.py"
    )
    tree = ast.parse(linear.read_text())
    store = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_block32_scale_words":
            store = node
    assert store is not None
    calls = [
        child
        for child in ast.walk(store)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "pack_scale_words"
    ]
    assert len(calls) == 1
    assert calls[0].keywords == []
