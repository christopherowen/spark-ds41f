# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 indexer's score kernel over an index cache whose slots past each
row's visible positions were never written, for ``compute-sanitizer --tool
initcheck``: the sanitizer reports a read of any of them.

usage: compute-sanitizer --tool initcheck --error-exitcode 1 \\
           python deepseek_v41_indexer_reads.py [--module PATH]

Run with ``PYTORCH_NO_CUDA_MEMORY_CACHING=1``, so every tensor is its own fresh
allocation and its unwritten bytes stay uninitialized. Every input but the pool is
written whole, and the scores are written before the kernel runs; the pool's keys and
scale words are written only at the visible positions of each row's pages. The extents
end inside a 64-row tile, on pages of 128 and 256 rows, in dense and candidate mode.
``--module`` scores with another copy of ``tilelang/indexer.py`` (the release's, as a
positive control: before its read bound it loads the last tile's rows past the extent).
"""

import argparse
import importlib.util

import torch

from vllm.models.deepseek_v4_1.tilelang import indexer

DEVICE = "cuda"
# (page size, visible positions per row): extents inside a tile, a tile long, and past
# several pages; rows of one case have different extents.
CASES = [(128, (1, 37, 64, 300)), (256, (100, 1061, 4133))]


def module(path: str | None):
    if path is None:
        return indexer
    package = "vllm.models.deepseek_v4_1.tilelang"
    spec = importlib.util.spec_from_file_location(f"{package}._release_indexer", path)
    loaded = importlib.util.module_from_spec(spec)
    loaded.__package__ = package
    spec.loader.exec_module(loaded)
    return loaded


def run_case(mod, page_size: int, extents: tuple[int, ...], candidates: bool) -> None:
    rows, heads, dim = len(extents), mod.INDEX_HEADS, mod.INDEX_DIM
    gen = torch.Generator(device=DEVICE).manual_seed(page_size + rows)
    logical = max(-(-e // page_size) for e in extents)
    width = logical * page_size
    pool_pages = logical * rows + 1
    nbytes = mod.page_bytes(page_size)
    # Never written but at the visible positions below (a fresh allocation).
    pool = torch.empty((pool_pages, nbytes), dtype=torch.uint8, device=DEVICE)
    page_table = (
        torch.randperm(pool_pages - 1, generator=gen, device=DEVICE)[: logical * rows]
        .add(1)
        .view(rows, logical)
        .to(torch.int32)
    )
    for r, extent in enumerate(extents):
        positions = torch.arange(extent, device=DEVICE)
        pages = page_table[r, positions // page_size].long()
        offsets = positions % page_size
        keys = torch.randint(
            0, 256, (extent, dim // 2), generator=gen, device=DEVICE, dtype=torch.uint8
        )
        scales = torch.randint(
            120, 130, (extent, 4), generator=gen, device=DEVICE, dtype=torch.uint8
        )
        for byte in range(dim // 2):
            pool[pages, offsets * (dim // 2) + byte] = keys[:, byte]
        for byte in range(4):
            pool[pages, page_size * (dim // 2) + offsets * 4 + byte] = scales[:, byte]
    q = torch.randint(
        0, 256, (rows, heads, dim // 2), generator=gen, device=DEVICE, dtype=torch.uint8
    )
    q_scales = torch.randint(
        120, 130, (rows, heads, 4), generator=gen, device=DEVICE, dtype=torch.uint8
    )
    weights = (torch.randn((rows, heads), generator=gen, device=DEVICE) / 64).bfloat16()
    lengths = torch.tensor(extents, dtype=torch.int32, device=DEVICE)
    active = torch.tensor([width], dtype=torch.int32, device=DEVICE)
    args = [
        q.view(torch.int8),
        q_scales.view(torch.uint32).reshape(rows, -1),
        weights,
        pool.view(torch.int8),
        pool.view(torch.uint32),
        page_table,
        lengths,
        active,
    ]
    if candidates and hasattr(mod, "_candidate_block"):
        # Block ids as the candidate source writes them: ascending, the block holding
        # the row's last position always kept (its rows past the end lie past the
        # count, unwritten), one block masked, -1 padding past them.
        block = mod.CANDIDATE_BLOCK
        chosen = torch.full(
            (rows, width // block), -1, dtype=torch.int32, device=DEVICE
        )
        counts = []
        for r, extent in enumerate(extents):
            last = (extent - 1) // block
            n = max(1, (last + 1) * 2 // 3)
            picked = torch.randperm(last, generator=gen, device=DEVICE)[: n - 1]
            ids = torch.cat([picked.sort().values, torch.tensor([last], device=DEVICE)])
            chosen[r, : ids.numel()] = ids.to(torch.int32)
            if ids.numel() > 2:
                chosen[r, ids.numel() // 2] = -1
            counts.append(block * (ids.numel() - 1) + extent - block * last)
        args += [chosen, torch.tensor(counts, dtype=torch.int32, device=DEVICE)]
    elif candidates:
        # A position-list module (the release's): ascending visible positions, some
        # masked; the counts end inside a tile.
        chosen = torch.full((rows, width), -1, dtype=torch.int32, device=DEVICE)
        counts = []
        for r, extent in enumerate(extents):
            n = max(1, extent * 2 // 3)
            picked = torch.randperm(extent, generator=gen, device=DEVICE)[:n]
            chosen[r, :n] = picked.sort().values.to(torch.int32)
            chosen[r, n // 2] = -1
            counts.append(n)
        args += [chosen, torch.tensor(counts, dtype=torch.int32, device=DEVICE)]
    scores = torch.zeros((rows, width), dtype=torch.bfloat16, device=DEVICE)
    plan = mod.IndexerV41Plan(width, page_size, candidates=candidates)
    plan.score(*args, scores)
    torch.accelerator.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--module", default=None)
    args = parser.parse_args()
    mod = module(args.module)
    for page_size, extents in CASES:
        for candidates in (False, True):
            run_case(mod, page_size, extents, candidates)
    print("scored", len(CASES) * 2, "cases")


if __name__ == "__main__":
    main()
