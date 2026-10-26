"""Whether the DSpark Markov bias projection's weight stays in L2 across a step's drafts.

usage: markov_l2.py [--reps 30]

A step drafts five tokens one after another. Each draft projects the previous token's
Markov embedding (rank 256) to the vocabulary through ``markov_w2``, a BF16
``[vocab, 256]`` weight: about 16.5 MB at a TP4 rank's shard (32,320 rows), 22.1 MB at
TP3's (43,136) and 66 MB whole (129,280). GB10's L2 holds 24 MB. Between drafts only an
argmax-sized kernel runs.

Per shape, the production projection (the BF16 head's, ``vocab.prepare_bf16_head`` and
``project_head``, one row, FP32 out, as the drafter's logits processor runs it) is timed
with CUDA events: once after a 128 MiB read that evicts L2 (cold), five times
back to back after that read with an argmax between (draft by draft), and the argmax
alone. If drafts 2-5 take about as long as the cold one, every draft streams the weight
from DRAM; if they are much faster, the weight stays in L2 and there is nothing to gain.

The bound of keeping it there: the same five drafts after the L2 prefetch kernel
(``cp.async.bulk.prefetch.L2`` with an ``evict_last`` policy, the decode prefetcher's,
128 threads x ``VLLM_L2_PREFETCH_GRID`` CTAs) has loaded the weight, and that
prefetch's own time.

Whether L2 holds the weight at all: ``stream_bound.py``'s ideal read kernel (48 CTAs,
32 KB chunks) reads the same weight five times back to back after the eviction, with and
without the prefetch first. Reads 2-5 much faster than the first mean L2 keeps the bytes
and the projection's own latency is what holds the drafts.

Prints one JSON line per shape: microseconds a launch (medians over the reps), cold and
per draft, after the prefetch, the prefetch, the five reads either way, and the bytes'
time at 266.6 GB/s.
"""

import argparse
import json
import statistics

import torch

RATE = 266.6e9
RANK = 256
SHAPES = {"tp4": 32320, "tp3": 43136, "tp1": 129280}


class _Projection:
    """A BF16 head: its weight, and the kernels ``prepare_bf16_head`` binds."""

    def __init__(self, weight):
        self.weight = weight


def main():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.models.deepseek_v4_1.tilelang.vocab import prepare_bf16_head, project_head
    from vllm.models.glm5next.nvidia import l2_prefetch as l2pf
    from vllm.models.glm5next.nvidia.tilelang.l2_prefetch import l2_prefetch_kernel

    from stream_bound import CTAS, THREADS, expert_stream

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(0)
    evict = torch.ones(128 << 20, dtype=torch.uint8, device="cuda").view(torch.int32)
    with set_current_vllm_config(VllmConfig()):
        layers = {}
        for name, n in SHAPES.items():
            weight = (torch.randn((n, RANK), generator=gen, device="cuda") * 0.02).bfloat16()
            layer = _Projection(weight)
            prepare_bf16_head(layer, torch.float32)
            layers[name] = layer
        prefetch_kernel = l2_prefetch_kernel(l2pf._GRID, l2pf._BLOCK, l2pf._CHUNK_BYTES)
        readers = {name: expert_stream(n * RANK // 2) for name, n in SHAPES.items()}
        sink = torch.empty((CTAS * THREADS,), dtype=torch.int32, device="cuda")
        row = torch.zeros((1,), dtype=torch.int32, device="cuda")
        prepare.freeze("markov_l2.py: every kernel was prepared before the first launch")
        for name, layer in layers.items():
            n = layer.weight.shape[0]
            x = torch.randn((1, RANK), generator=gen, device="cuda").bfloat16()
            logits = [None]

            def project():
                logits[0] = project_head(layer, x)

            def argmax():
                torch.argmax(logits[0], dim=-1)

            for _ in range(3):  # warm
                project()
                argmax()
            torch.cuda.synchronize()
            cold, drafts, arg = [], [[] for _ in range(5)], []
            for _ in range(args.reps):
                evict.add_(1)
                events = [torch.cuda.Event(enable_timing=True) for _ in range(12)]
                events[0].record()
                project()
                events[1].record()
                torch.cuda.synchronize()
                cold.append(events[0].elapsed_time(events[1]) * 1000)
                evict.add_(1)
                marks = [torch.cuda.Event(enable_timing=True) for _ in range(11)]
                marks[0].record()
                for d in range(5):
                    project()
                    marks[2 * d + 1].record()
                    argmax()
                    marks[2 * d + 2].record()
                torch.cuda.synchronize()
                for d in range(5):
                    start = marks[2 * d]  # after the previous argmax
                    drafts[d].append(start.elapsed_time(marks[2 * d + 1]) * 1000)
                    arg.append(marks[2 * d + 1].elapsed_time(marks[2 * d + 2]) * 1000)
            segs = torch.tensor([layer.weight.data_ptr(), layer.weight.numel() * 2],
                                dtype=torch.int64, device="cuda")
            prefetched, prefetch = [[] for _ in range(5)], []
            for _ in range(args.reps):
                evict.add_(1)
                marks = [torch.cuda.Event(enable_timing=True) for _ in range(12)]
                marks[0].record()
                prefetch_kernel(segs)
                marks[1].record()
                for d in range(5):
                    project()
                    marks[2 * d + 2].record()
                    argmax()
                    marks[2 * d + 3].record()
                torch.cuda.synchronize()
                prefetch.append(marks[0].elapsed_time(marks[1]) * 1000)
                for d in range(5):
                    prefetched[d].append(marks[2 * d + 1].elapsed_time(marks[2 * d + 2]) * 1000)
            words = layer.weight.view(torch.int32).view(1, -1)
            reads = {False: [[] for _ in range(5)], True: [[] for _ in range(5)]}
            for prefetched_first in (False, True):
                for _ in range(args.reps):
                    evict.add_(1)
                    if prefetched_first:
                        prefetch_kernel(segs)
                    marks = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
                    marks[0].record()
                    for d in range(5):
                        readers[name](words, row, sink)
                        marks[d + 1].record()
                    torch.cuda.synchronize()
                    for d in range(5):
                        reads[prefetched_first][d].append(marks[d].elapsed_time(marks[d + 1]) * 1000)
            nbytes = n * RANK * 2
            print(json.dumps(dict(
                shape=name, rows=n, weight_mb=round(nbytes / 1e6, 2),
                bytes_us=round(nbytes / RATE * 1e6, 1),
                cold_us=round(statistics.median(cold), 1),
                draft_us=[round(statistics.median(d), 1) for d in drafts],
                argmax_us=round(statistics.median(arg), 1),
                prefetched_draft_us=[round(statistics.median(d), 1) for d in prefetched],
                prefetch_us=round(statistics.median(prefetch), 1),
                read_us=[round(statistics.median(d), 1) for d in reads[False]],
                prefetched_read_us=[round(statistics.median(d), 1) for d in reads[True]],
            )), flush=True)


if __name__ == "__main__":
    main()
