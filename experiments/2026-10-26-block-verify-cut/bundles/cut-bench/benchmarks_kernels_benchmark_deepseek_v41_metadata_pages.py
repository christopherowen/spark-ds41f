# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired per-call timing of DeepSeek-V4.1's sparse-MLA page tables over each row's
live pages (``tilelang.metadata``'s ``decode_metadata`` and ``pages``) against the
kernels they replace, which wrote every column of the table.

At ``--max-model-len`` 1M and 256-token blocks a page table has 4,096 columns; a
row reads the pages of its visible entries. The replaced kernels are kept here as
they were at vLLM 26382f5cf (``_baseline_*``). Shapes: decode steps of one and 16
requests (1 + 5 drafted rows each) at 4K, 64K and 1M-token contexts, and a
16,384-row prefill chunk at the start of a prompt and 48K into it; both main page
sizes (256 entries at ratio 1, 128 at ratio 2); for ``pages``, the indexer's decode
chunk and a 256-row prefill chunk.

Each shape is measured on eight fresh inputs. On each, both kernels' outputs are
first checked: the window metadata equal, the page tables equal on every live
column. Then CUDA graphs of 100 calls of both run back to back, alternating which
goes first, and the shape's 95% interval comes from the eight per-input
differences. The gate (``--gate``) is that neither kernel is slower at any shape
(the interval's lower bound above zero).
"""

import argparse
import json
import math
import statistics

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import metadata
from vllm.models.deepseek_v4_1.tilelang.metadata import _PASS, _threads, _window_macro

DEVICE = "cuda"
CALLS = 100
BLOCK_TOKENS = 256
MAIN_WIDTH = (1 << 20) // BLOCK_TOKENS  # 1M-token max_model_len
WINDOW = SWA_WIDTH = 128
SWA_PAGE = BLOCK_TOKENS  # the window part reads the same table
DRAFT_ROWS = 6

# name: (requests, rows a request, context tokens at the step's last row)
DECODE = {
    "c1 4K": (1, DRAFT_ROWS, 4096),
    "c1 64K": (1, DRAFT_ROWS, 65536),
    "c1 1M": (1, DRAFT_ROWS, 1 << 20),
    "c16 4K": (16, DRAFT_ROWS, 4096),
    "c16 64K": (16, DRAFT_ROWS, 65536),
    "prefill 16K": (1, 16384, 16384),
    "prefill 48K-64K": (1, 16384, 65536),
}
PAGES = {
    "decode c1 64K": (1, DRAFT_ROWS, 65536),
    "decode c16 64K": (16, DRAFT_ROWS, 65536),
    "prefill 256 rows at 64K": (1, 256, 65536),
}


@tilelang.jit(pass_configs=_PASS)
def _baseline_decode_metadata_kernel(
    page: int, window: int, width: int, draft: bool, replay: bool, main_width: int
):
    """``decode_metadata_kernel`` at 26382f5cf: every main page-table column."""
    rows = T.dynamic("rows")
    src_rows = T.dynamic("src_rows")
    table_rows = T.dynamic("table_rows")
    table_cols = T.dynamic("table_cols")
    table_stride = T.dynamic("table_stride")
    main_src_rows = T.dynamic("main_src_rows")
    main_rows = T.dynamic("main_rows")
    main_cols = T.dynamic("main_cols")
    main_stride = T.dynamic("main_stride")
    req_count = T.dynamic("req_count")
    starts_len = T.dynamic("starts_len")
    replay_len = T.dynamic("replay_len")
    threads, per_thread = _threads(width)
    window_row = _window_macro(page, window, width, draft, replay, threads, per_thread)
    main_per_thread = (main_width + threads - 1) // threads

    @T.prim_func
    def metadata_decode(
        positions: T.Tensor((src_rows,), T.int64),
        reqs: T.Tensor((src_rows,), T.int32),
        table: T.StridedTensor((table_rows, table_cols), (table_stride, 1), T.int32),
        swa: T.Tensor((rows, width), T.int32),
        lengths: T.Tensor((rows,), T.int32),
        top_lengths: T.Tensor((rows,), T.int32),
        visible: T.Tensor((src_rows,), T.int32),
        starts: T.Tensor((starts_len,), T.int32),
        request_positions: T.Tensor((req_count,), T.int64),
        replay_start: T.Tensor((replay_len,), T.int64),
        main_reqs: T.Tensor((main_src_rows,), T.int32),
        main_table: T.StridedTensor((main_rows, main_cols), (main_stride, 1), T.int32),
        main_pages: T.Tensor((rows, main_width), T.int32),
        offset: T.int32,
        main_read_width: T.int32,
    ):
        with T.Kernel(rows, 2, threads=threads) as (row, part):
            tx = T.get_thread_binding()
            info = T.alloc_local((2,), T.int64)
            if part == 0:
                window_row(
                    row, tx, info, positions, reqs, table, swa, lengths, top_lengths,
                    visible, starts, request_positions, replay_start, offset,
                )  # fmt: skip
            else:
                req = main_reqs[offset + row]
                for k in T.serial(main_per_thread):
                    col = k * threads + tx
                    if col < main_width:
                        page_id = T.if_then_else(
                            (req >= 0) & (col < main_read_width),
                            main_table[req, col],
                            -1,
                        )
                        main_pages[row, col] = T.if_then_else(page_id > 0, page_id, -1)

    return metadata_decode


@tilelang.jit(pass_configs=_PASS)
def _baseline_pages_kernel(width: int):
    """``pages_kernel`` at 26382f5cf: every column."""
    rows = T.dynamic("rows")
    src_rows = T.dynamic("src_rows")
    table_rows = T.dynamic("table_rows")
    table_cols = T.dynamic("table_cols")
    table_stride = T.dynamic("table_stride")

    @T.prim_func
    def metadata_pages(
        reqs: T.Tensor((src_rows,), T.int32),
        table: T.StridedTensor((table_rows, table_cols), (table_stride, 1), T.int32),
        out: T.Tensor((rows, width), T.int32),
        offset: T.int32,
        read_width: T.int32,
    ):
        with T.Kernel(rows, T.ceildiv(width, 128), threads=128) as (row, cb):
            col = cb * 128 + T.get_thread_binding()
            if col < width:
                req = reqs[offset + row]
                page_id = T.if_then_else(
                    (req >= 0) & (col < read_width), table[req, col], -1
                )
                out[row, col] = T.if_then_else(page_id > 0, page_id, -1)

    return metadata_pages


def _step(requests: int, rows_each: int, context: int, ratio: int, seed: int):
    """One step's per-row metadata, as the builders leave it: ``rows_each`` rows a
    request ending at position ``context - 1``, each row's visible entries at the
    compression ratio, and every request's block table."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    rows = requests * rows_each
    row = torch.arange(rows, device=DEVICE)
    req = (row // rows_each).to(torch.int32)
    pos = (context - rows_each + row % rows_each).to(torch.int64)
    visible = ((pos + 1) // ratio).to(torch.int32)
    starts = torch.arange(0, rows + 1, rows_each, dtype=torch.int32, device=DEVICE)
    table = torch.randint(
        1, 1 << 20, (16, MAIN_WIDTH), generator=gen, device=DEVICE, dtype=torch.int32
    )
    table[:, 7] = 0  # a hole
    return dict(
        rows=rows,
        pos=pos,
        req=req,
        visible=visible,
        starts=starts,
        request_positions=pos[::rows_each].contiguous(),
        table=table,
    )


def _live_mask(step, page: int, width: int) -> torch.Tensor:
    live = (step["visible"].long() + page - 1) // page
    cols = torch.arange(width, device=DEVICE)
    return cols[None, :] < live.clamp_max(width)[:, None]


def decode_case(rows_each, requests, context, main_page, seed):
    """Both decode-metadata kernels on one step, each with its outputs."""
    step = _step(requests, rows_each, context, BLOCK_TOKENS // main_page, seed)
    rows = step["rows"]

    def outputs():
        return (
            torch.full((rows, SWA_WIDTH), 7, dtype=torch.int32, device=DEVICE),
            torch.full((rows,), 7, dtype=torch.int32, device=DEVICE),
            torch.full((rows,), 7, dtype=torch.int32, device=DEVICE),
            torch.full((rows, MAIN_WIDTH), 7, dtype=torch.int32, device=DEVICE),
        )

    ours_out, theirs_out = outputs(), outputs()
    common = (
        step["pos"], step["req"], step["table"], *ours_out[:3], step["visible"],
        step["starts"], step["request_positions"],
    )  # fmt: skip

    def ours():
        metadata.decode_metadata(
            *common, step["req"], step["table"], ours_out[3], 0, rows, MAIN_WIDTH,
            page=SWA_PAGE, window=WINDOW, draft=False, main_page=main_page,
        )  # fmt: skip

    baseline = _baseline_decode_metadata_kernel(
        SWA_PAGE, WINDOW, SWA_WIDTH, False, False, MAIN_WIDTH
    )

    def theirs():
        baseline(
            step["pos"], step["req"], step["table"], *theirs_out[:3], step["visible"],
            step["starts"], step["request_positions"], step["request_positions"],
            step["req"], step["table"], theirs_out[3], 0, MAIN_WIDTH,
        )  # fmt: skip

    def check():
        for a, b in zip(ours_out[:3], theirs_out[:3]):
            if not torch.equal(a, b):
                raise SystemExit("decode metadata: the window metadata differ")
        live = _live_mask(step, main_page, MAIN_WIDTH)
        if not torch.equal(ours_out[3][live], theirs_out[3][live]):
            raise SystemExit("decode metadata: a live page differs")

    return ours, theirs, check


def pages_case(rows_each, requests, context, page, seed):
    """Both index page-table kernels on one indexer chunk."""
    step = _step(requests, rows_each, context, BLOCK_TOKENS // page, seed)
    rows = step["rows"]
    ours_out = torch.full((rows, MAIN_WIDTH), 7, dtype=torch.int32, device=DEVICE)
    theirs_out = torch.full_like(ours_out, 7)
    baseline = _baseline_pages_kernel(MAIN_WIDTH)

    def ours():
        metadata.pages(
            step["req"], step["visible"], step["table"], ours_out, 0, rows, MAIN_WIDTH,
            page=page,
        )  # fmt: skip

    def theirs():
        baseline(step["req"], step["table"], theirs_out, 0, MAIN_WIDTH)

    def check():
        live = _live_mask(step, page, MAIN_WIDTH)
        if not torch.equal(ours_out[live], theirs_out[live]):
            raise SystemExit("pages: a live page differs")

    return ours, theirs, check


def graph(function):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(CALLS):
            function()
    return capture


def elapsed(capture) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS  # microseconds a call


def measure(make, args, samples: int = 8) -> dict:
    ours_us, theirs_us, differences = [], [], []
    for sample in range(samples):
        ours_call, theirs_call, check = make(*args, seed=sample)
        ours_call()
        theirs_call()
        torch.cuda.synchronize()
        check()
        ours, theirs = graph(ours_call), graph(theirs_call)
        pair = (ours, theirs) if sample % 2 == 0 else (theirs, ours)
        times = {id(capture): elapsed(capture) for capture in pair}
        ours_us.append(times[id(ours)])
        theirs_us.append(times[id(theirs)])
        differences.append(times[id(ours)] - times[id(theirs)])
        del ours, theirs, pair
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(samples)  # t(7), 95%
    return {
        "live_pages_us": statistics.median(ours_us),
        "every_column_us": statistics.median(theirs_us),
        "difference_us": mean,
        "interval_us": [mean - half, mean + half],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    results = {}
    for main_page in (256, 128):
        for name, (requests, rows_each, context) in DECODE.items():
            results[f"decode_metadata page {main_page} {name}"] = measure(
                decode_case, (rows_each, requests, context, main_page)
            )
    for name, (requests, rows_each, context) in PAGES.items():
        results[f"pages {name}"] = measure(
            pages_case, (rows_each, requests, context, 128)
        )
    print(json.dumps(results, indent=1))
    if args.gate:
        slower = [name for name, r in results.items() if r["interval_us"][0] > 0]
        if slower:
            raise SystemExit(f"live-page tables slower at {', '.join(slower)}")


if __name__ == "__main__":
    main()
