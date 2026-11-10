#!/usr/bin/env python3
"""#995 guard (round 3): device time of the two guarded kernels with the launch latency amortized.
Usage: k32995.py --label L --out OUT.jsonl
The 24 cells of benchmark_copy_drafts.py's copy case (same construction and seeds; its timed()
imported unmodified: CUDA-graph replay, 64 MiB flush before each replay, median), copy_match run once
for the sources, then one CUDA graph of 32 back-to-back (copy_alloc, copy_apply) pairs, 51 replays:
the per-pair microseconds (median / 32). A single pair's replay is ~8 us, mostly launch latency, and
its medians sit on the event timer's quantum; 32 pairs a replay resolve the kernels' own time."""
import argparse, importlib.util, json
import torch
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.spec_decode import tilelang_copy as tlc

p = argparse.ArgumentParser(); p.add_argument("--label", required=True); p.add_argument("--out", required=True)
a = p.parse_args()
dev = torch.device("cuda")
spec = importlib.util.spec_from_file_location("bcd", "benchmarks/kernels/benchmark_copy_drafts.py")
bcd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bcd)
VOCAB, LOOKAHEAD, BLOCK, INITIAL, MIN_MATCH, PAIRS = 129280, 16, 5, 8, bcd.MIN_MATCH, 32
fh = open(a.out, "w")
for uva in (True, False):
    for num_reqs in (1, 8, 16):
        for context in bcd.CONTEXTS:
            g = torch.Generator(device="cpu").manual_seed(num_reqs * 7 + context)
            shape = (16, context + 64)
            if uva:
                buf = StagedWriteTensor(shape, dtype=torch.int32, device=dev, uva_instead_of_gpu=True); hist = buf.gpu
            else:
                buf = None; hist = torch.empty(shape, dtype=torch.int32, device=dev)
            total_len = torch.zeros(16, dtype=torch.int32, device=dev)
            for slot in range(num_reqs):
                tokens = torch.randint(0, VOCAB, (context,), generator=g, dtype=torch.int32)
                start = int(torch.randint(0, context // 4 + 1, (1,), generator=g))
                tail = min(600, context // 2)
                tokens[context - tail:] = tokens[start:start + tail]
                hist[slot, :context].copy_(tokens.to(hist.device)); total_len[slot] = context
            idx = torch.arange(num_reqs, dtype=torch.int64, device=dev)
            sources = torch.empty((num_reqs, tlc.CHUNKS, tlc.num_levels(MIN_MATCH)), dtype=torch.int32, device=dev)
            int_state = torch.zeros((16, 2), dtype=torch.int32, device=dev)
            float_state = torch.zeros((16, 2), dtype=torch.float32, device=dev)
            dspark = torch.zeros((num_reqs, BLOCK), dtype=torch.int64, device=dev)
            dconf = torch.full((num_reqs, BLOCK), 0.5, device=dev)
            ones = torch.ones(num_reqs, dtype=torch.int32, device=dev)
            chosen = torch.empty((num_reqs, tlc.CHOSEN_COLUMNS), dtype=torch.int32, device=dev)
            estimates = torch.empty(num_reqs, dtype=torch.float32, device=dev)
            outs = (torch.empty((num_reqs, LOOKAHEAD), dtype=torch.int64, device=dev),
                    torch.empty((num_reqs, LOOKAHEAD), device=dev), torch.empty((num_reqs, LOOKAHEAD), device=dev),
                    torch.empty(num_reqs, dtype=torch.int32, device=dev))
            tlc.copy_match(idx, hist, total_len, sources, MIN_MATCH, LOOKAHEAD); torch.cuda.synchronize()
            def pairs():
                for _ in range(PAIRS):
                    tlc.copy_alloc(idx, total_len, sources, ones, ones, ones, int_state, float_state,
                                   chosen, estimates, outs[3], MIN_MATCH, LOOKAHEAD, BLOCK, INITIAL)
                    tlc.copy_apply(idx, hist, chosen, estimates, int_state, dspark, dconf, *outs[:3])
            med, low = bcd.timed(pairs, replays=51)
            r = {"label": a.label, "history": "uva" if uva else "device", "reqs": num_reqs, "ctx": context,
                 "pair_us": round(med / PAIRS, 4), "pair_min_us": round(low / PAIRS, 4), "graph_med_us": round(med, 3)}
            fh.write(json.dumps(r) + "\n"); fh.flush(); print(json.dumps(r), flush=True)
            del buf, hist
            torch.cuda.empty_cache()
fh.close()
