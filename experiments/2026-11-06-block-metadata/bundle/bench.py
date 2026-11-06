"""Paired component screen. Requires an exclusive, thermally guarded GPU window.

The control is the unreleased candidate-40 source. No released serving build is
re-benchmarked. Captured GPU time and host-plus-device preparation time are
different metrics; neither is reported as full-step or end-to-end serving time.
"""
import argparse
import json
import math
from pathlib import Path
import resource
import statistics
import time

import torch

from incremental import IncrementalBlockTables
from test_gpu import Case, test_locked_control_source
from vllm.v1.worker.gpu.block_table import BlockTables


WIDTHS = [16384] * 4 + [4096] * 6 + [1024] * 3 + [64, 32, 1, 1]
BLOCKS = [64] * 4 + [256] * 6 + [1024] * 3 + [1] * 4


class FullCase(Case):
    def __init__(self, mode, concurrency, context):
        self.concurrency, self.rows = concurrency, concurrency
        self.active = list(range(concurrency))
        self.context = context
        self.counts = [min((context + b - 1) // b, w - 1) if g < 13 else 1
                       for g, (b, w) in enumerate(zip(BLOCKS, WIDTHS))]
        args = dict(block_sizes=BLOCKS, kernel_block_sizes=BLOCKS,
                    max_num_blocks_per_group=WIDTHS, max_num_reqs=16,
                    max_num_batched_tokens=128, device=torch.device("cuda"))
        self.control = BlockTables(**args)
        self.candidate = IncrementalBlockTables(**args, mode=mode)
        self.pointers = [t.data_ptr() for t in self.candidate.input_block_tables]
        self.initial = {s: self.make_ids(s, 0, 0) for s in range(16)}
        self.replacement = {s: self.make_ids(s, 0, 1) for s in range(16)}
        self.delta = tuple([990001 + g] if g < 13 else [] for g in range(17))
        self.ids = {}
        self.reset()

    def make_ids(self, slot, live, generation):
        start = 1000000 * (generation + 1) + slot * 20000
        return tuple([start + 3 * i for i in range(n)] for n in self.counts)

    def reset(self):
        for slot in range(16):
            self.replace(slot, self.initial[slot])
        self.publish()
        self.check()


def summarize(control, candidate):
    delta = [b - a for a, b in zip(control, candidate)]
    mean = statistics.fmean(delta)
    half = 2.365 * statistics.stdev(delta) / math.sqrt(8)
    before, after = statistics.median(control), statistics.median(candidate)
    return {"before_median_us": before, "after_median_us": after,
            "median_change_percent": 100 * (after / before - 1),
            "paired_mean_change_us": mean, "paired_95_interval_us": [mean - half, mean + half],
            "control_samples_us": control, "candidate_samples_us": candidate}


def capture(tables, mappings, rows):
    for idx in mappings:
        tables.gather_block_tables(idx, rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(32):
            for idx in mappings:
                tables.gather_block_tables(idx, rows)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph, 32 * len(mappings)


def gpu_pair(case, event):
    indices = [case.active]
    if event == "reorder":
        indices.append(list(reversed(case.active)) if case.concurrency > 1 else [1])
    mappings = [torch.tensor(x, dtype=torch.int32, device="cuda") for x in indices]
    graphs = [capture(t, mappings, case.rows) for t in (case.control, case.candidate)]
    values = [[], []]
    for sample in range(8):
        for arm in ([0, 1] if sample % 2 == 0 else [1, 0]):
            graph, calls = graphs[arm]
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            values[arm].append(start.elapsed_time(end) * 1000 / (10 * calls))
    case.check()
    return summarize(*values)


def preparation_pair(case, event):
    idx = torch.tensor(case.active, dtype=torch.int32, device="cuda")
    cpu, completion = [[], []], [[], []]
    for sample in range(8):
        # Restore both arms outside the measured segment. Each measured call is
        # followed by retirement before any UVA slot is reused.
        case.reset()
        for arm in ([0, 1] if sample % 2 == 0 else [1, 0]):
            tables = (case.control, case.candidate)[arm]
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            if event == "page_crossing":
                for slot in case.active:
                    tables.append_block_ids(slot, case.delta, overwrite=False)
            elif event == "overwrite":
                for slot in case.active:
                    tables.append_block_ids(slot, case.replacement[slot], overwrite=True)
            tables.apply_staged_writes()
            tables.gather_block_tables(idx, case.rows)
            enqueued = time.perf_counter_ns()
            torch.cuda.synchronize()
            finished = time.perf_counter_ns()
            cpu[arm].append((enqueued - start) / 1000)
            completion[arm].append((finished - start) / 1000)
        case.check()
    return {"host_enqueue": summarize(*cpu), "host_and_device_completion": summarize(*completion)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    test_locked_control_source()
    records = []
    for mode in ("row", "chunk", "span"):
        for context in (32768, 65536, 262144, 524288, 1048576):
            for concurrency in (1, 8, 16):
                case = FullCase(mode, concurrency, context)
                result = {"mode": mode, "context_label": context, "concurrency": concurrency,
                          "live_kernel_blocks": case.counts,
                          "extra_gpu_cache_bytes": case.candidate.gather_state.numel() * 8,
                          "epoch_host_bytes_including_pool": 16 * 8 * (1 + len(case.candidate.epochs.pool._uva_bufs)),
                          "gpu": {}, "preparation": {}}
                for event in ("steady", "reorder"):
                    result["gpu"][event] = gpu_pair(case, event)
                for event in ("steady", "page_crossing", "overwrite"):
                    result["preparation"][event] = preparation_pair(case, event)
                records.append(result)
                # Persist every completed case, even if a later case fails.
                receipt = {"kind": "component screen, not a serving benchmark",
                           "control_source": "5f926df6d3126266a9c88c82c06544060008c33c",
                           "group_widths": WIDTHS,
                           "geometry_note": "17-group runner-benchmark geometry; live counts scale by explicit block sizes, four state groups keep one ID. Not an observed serving allocation.",
                           "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "torch_peak_allocated_bytes_two_arms": torch.cuda.max_memory_allocated(),
                           "records": records}
                args.output.write_text(json.dumps(receipt, indent=2) + "\n")
                print(json.dumps({"completed_cases": len(records), "case_result": result}), flush=True)
                del case
    print("45 component cases completed; no serving-performance conclusion", flush=True)


if __name__ == "__main__":
    main()
