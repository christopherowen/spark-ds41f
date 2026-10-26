# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired timing of DeepSeek-V4.1's sparse-MLA metadata builder in one launch
(``tilelang.metadata.builder``) against the two launches it replaces, the
``requests`` and ``tokens`` kernels as they were at vLLM 26382f5cf (kept here as
``_baseline_*``).

Every KV-cache group's builder runs once a step, eagerly, before the step's graph:
26 groups at a DS4.1 decode step (target and drafter), so the pair was 52 eager
launches a step. Shapes: decode steps of one and 16 requests (1 + 5 drafted rows
each) and a 16,384-row prefill chunk, for a paged group at ratio 1 and 2 and the
sliding-window ring.

Each shape is measured on eight fresh inputs. On each, both forms' outputs are first
checked bit for bit. Then 100 calls of each run back to back, alternating which goes
first, twice: eagerly, timed on the host from the first launch to the end of the
last kernel (what a step's preparation pays), and as CUDA graphs (the device time
alone). Each shape's 95% interval comes from the eight per-input differences. The
gate (``--gate``) is that the single launch is slower in neither (the interval's
lower bound above zero).
"""

import argparse
import json
import math
import statistics
import time

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import metadata
from vllm.models.deepseek_v4_1.tilelang.metadata import _BLOCK, _PASS, _search_steps

DEVICE = "cuda"
CALLS = 100
MAX_REQUESTS = 16  # the TP4 recipe's max_num_seqs
CAPACITY = 16384  # its max_num_batched_tokens
TABLE_COLS = 4096
DRAFT_ROWS = 6
# name: (requests, rows a request)
STEPS = {"decode c1": (1, DRAFT_ROWS), "decode c16": (16, DRAFT_ROWS)}
STEPS["prefill 16K"] = (1, 16384)
# name: (page, ratio, circular)
GROUPS = {
    "paged ratio 1": (256, 1, False),
    "paged ratio 2": (128, 2, False),
    "window ring": (128, 1, True),
}


@tilelang.jit(pass_configs=_PASS)
def _baseline_requests_kernel(max_requests: int):
    """``requests_kernel`` at 26382f5cf."""
    starts_len = T.dynamic("starts_len")
    seq_len = T.dynamic("seq_len")

    @T.prim_func
    def metadata_requests(
        starts: T.Tensor((starts_len,), T.int32),
        seq: T.Tensor((seq_len,), T.int32),
        out_starts: T.Tensor((max_requests + 1,), T.int32),
        positions: T.Tensor((max_requests,), T.int64),
        counts: T.Tensor((2,), T.int32),
        nr: T.int32,
        nt: T.int32,
    ):
        with T.Kernel(T.ceildiv(max_requests + 1, _BLOCK), threads=_BLOCK) as bx:
            r = bx * _BLOCK + T.get_thread_binding()
            if r <= max_requests:
                start = T.if_then_else(r <= nr, starts[r], nt)
                end = T.if_then_else(r < nr, starts[r + 1], nt)
                out_starts[r] = start
                if r < max_requests:
                    positions[r] = T.if_then_else(
                        r < nr,
                        T.Cast(T.int64, seq[r] - (end - start)),
                        T.int64(-1),
                    )
            if r == 0:
                counts[0] = nt
                counts[1] = nr

    return metadata_requests


@tilelang.jit(pass_configs=_PASS)
def _baseline_tokens_kernel(max_requests: int, page: int, ratio: int, circular: bool):
    """``tokens_kernel`` at 26382f5cf."""
    cover = T.dynamic("cover")
    starts_len = T.dynamic("starts_len")
    seq_len = T.dynamic("seq_len")
    table_rows = T.dynamic("table_rows")
    table_cols = T.dynamic("table_cols")
    table_stride = T.dynamic("table_stride")
    slot_rows = T.dynamic("slot_rows")
    steps = _search_steps(max_requests)

    @T.prim_func
    def metadata_tokens(
        starts: T.Tensor((starts_len,), T.int32),
        seq: T.Tensor((seq_len,), T.int32),
        table: T.StridedTensor((table_rows, table_cols), (table_stride, 1), T.int32),
        input_slots: T.Tensor((slot_rows,), T.int64),
        positions: T.Tensor((cover,), T.int64),
        reqs: T.Tensor((cover,), T.int32),
        slots: T.Tensor((cover,), T.int64),
        visible: T.Tensor((cover,), T.int32),
        nr: T.int32,
        nt: T.int32,
    ):
        with T.Kernel(T.ceildiv(cover, _BLOCK), threads=_BLOCK) as bx:
            t = bx * _BLOCK + T.get_thread_binding()
            bounds = T.alloc_local((2,), T.int32)
            if t < cover:
                bounds[0] = 0
                bounds[1] = nr
                for _ in T.serial(steps):
                    if bounds[0] < bounds[1]:
                        mid = (bounds[0] + bounds[1]) // 2
                        end_mid = T.if_then_else(mid < nr, starts[mid + 1], nt)
                        if end_mid <= t:
                            bounds[0] = mid + 1
                        else:
                            bounds[1] = mid
                lo = bounds[0]
                live = (t < nt) & (lo < nr)
                start = T.if_then_else(live, starts[lo], 0)
                end = T.if_then_else(live, starts[lo + 1], 0)
                sq = T.if_then_else(live, seq[lo], 0)
                pos = T.Cast(T.int64, sq) - T.Cast(T.int64, end - start) + (t - start)
                input_slot = T.if_then_else(t < nt, input_slots[t], T.int64(-1))
                if circular:
                    valid0 = live & (pos >= 0)
                else:
                    valid0 = live & (pos >= 0) & (input_slot >= 0)
                logical = T.truncdiv(pos, ratio)
                page_col = T.int64(0) if circular else T.truncdiv(logical, page)
                valid1 = valid0 & (page_col < table_cols)
                block = T.if_then_else(
                    valid1, T.Cast(T.int64, table[lo, page_col]), T.int64(0)
                )
                valid = valid1 & (block > 0)
                slot = block * page + T.truncmod(logical, page)
                emit = valid & (T.truncmod(pos + 1, ratio) == 0)
                if circular:
                    emit = emit & (t >= end - page)
                positions[t] = T.if_then_else(valid, pos, T.int64(-1))
                reqs[t] = T.if_then_else(valid, lo, -1)
                slots[t] = T.if_then_else(emit, slot, T.int64(-1))
                visible[t] = T.if_then_else(
                    valid, T.Cast(T.int32, T.truncdiv(pos + 1, ratio)), 0
                )

    return metadata_tokens


def _inputs(requests: int, rows_each: int, seed: int):
    """A step's builder inputs: ``rows_each`` rows a request, contexts of 4K-64K,
    every row with a slot."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    nr, nt = requests, requests * rows_each
    qsl = torch.full((MAX_REQUESTS + 1,), nt, dtype=torch.int32, device=DEVICE)
    qsl[: nr + 1] = torch.arange(0, nt + 1, rows_each, dtype=torch.int32)
    seq = torch.zeros(MAX_REQUESTS, dtype=torch.int32, device=DEVICE)
    seq[:nr] = torch.randint(
        4096, 65536, (nr,), generator=gen, device=DEVICE, dtype=torch.int32
    ).clamp_min(rows_each)
    table = torch.randint(
        1, 1 << 20, (MAX_REQUESTS, TABLE_COLS), generator=gen, device=DEVICE,
        dtype=torch.int32,
    )  # fmt: skip
    slots = torch.randint(
        0, 1 << 30, (CAPACITY,), generator=gen, device=DEVICE, dtype=torch.int64
    )
    return qsl, seq, table, slots, nr, nt


def _outputs():
    return (
        torch.full((MAX_REQUESTS + 1,), 7, dtype=torch.int32, device=DEVICE),
        torch.full((MAX_REQUESTS,), 7, dtype=torch.int64, device=DEVICE),
        torch.full((2,), 7, dtype=torch.int32, device=DEVICE),
        torch.full((CAPACITY,), 7, dtype=torch.int64, device=DEVICE),
        torch.full((CAPACITY,), 7, dtype=torch.int32, device=DEVICE),
        torch.full((CAPACITY,), 7, dtype=torch.int64, device=DEVICE),
        torch.full((CAPACITY,), 7, dtype=torch.int32, device=DEVICE),
    )


def case(requests, rows_each, page, ratio, circular, seed):
    qsl, seq, table, slots, nr, nt = _inputs(requests, rows_each, seed)
    ours_out, theirs_out = _outputs(), _outputs()
    requests_kernel = _baseline_requests_kernel(MAX_REQUESTS)
    tokens_kernel = _baseline_tokens_kernel(MAX_REQUESTS, page, ratio, circular)
    cover = min(CAPACITY, max(1, -(-nt // _BLOCK)) * _BLOCK)

    def ours():
        metadata.builder(
            qsl, seq, table, slots, *ours_out, nr, nt, page=page, ratio=ratio,
            circular=circular,
        )  # fmt: skip

    def theirs():
        out_starts, request_pos, counts, pos, reqs, slot_out, visible = theirs_out
        requests_kernel(qsl, seq, out_starts, request_pos, counts, nr, nt)
        tokens_kernel(
            qsl, seq, table, slots, pos[:cover], reqs[:cover], slot_out[:cover],
            visible[:cover], nr, nt,
        )  # fmt: skip

    def check():
        for a, b in zip(ours_out, theirs_out):
            if not torch.equal(a, b):
                raise SystemExit("builder: the outputs differ from the pair's")

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


def graph_us(capture) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS


def eager_us(function) -> float:
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(CALLS):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - begin) * 1e6 / CALLS


def interval(differences):
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(len(differences))
    return mean, [mean - half, mean + half]


def measure(args, samples: int = 8) -> dict:
    rows = {"graph": ([], [], []), "eager": ([], [], [])}
    for sample in range(samples):
        ours, theirs, check = case(*args, seed=sample)
        ours()
        theirs()
        torch.cuda.synchronize()
        check()
        captures = {"ours": graph(ours), "theirs": graph(theirs)}
        order = ("ours", "theirs") if sample % 2 == 0 else ("theirs", "ours")
        timed = {name: graph_us(captures[name]) for name in order}
        eager = {name: eager_us(ours if name == "ours" else theirs) for name in order}
        for mode, times in (("graph", timed), ("eager", eager)):
            one, pair, diff = rows[mode]
            one.append(times["ours"])
            pair.append(times["theirs"])
            diff.append(times["ours"] - times["theirs"])
        del captures
    result = {}
    for mode, (one, pair, diff) in rows.items():
        mean, bounds = interval(diff)
        result[mode] = {
            "one_launch_us": statistics.median(one),
            "two_launches_us": statistics.median(pair),
            "difference_us": mean,
            "interval_us": bounds,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    results = {
        f"{step} {group}": measure((*STEPS[step], *GROUPS[group]))
        for step in STEPS
        for group in GROUPS
    }
    print(json.dumps(results, indent=1))
    if args.gate:
        slower = [
            f"{name} ({mode})"
            for name, result in results.items()
            for mode, r in result.items()
            if r["interval_us"][0] > 0
        ]
        if slower:
            raise SystemExit(f"one-launch builder slower at {', '.join(slower)}")


if __name__ == "__main__":
    main()
