#!/usr/bin/env python3
"""Benchmark CopyDrafter copy_match latency across 4K-1M contexts on UVA memory."""

import argparse
import json
from pathlib import Path
import statistics
import torch

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.spec_decode import tilelang_copy

VOCAB = 129280
LOOKAHEAD = 16
CONTEXTS = (4096, 16384, 65536, 131072, 262144, 524288, 1048576)
REQ_COUNTS = (1, 8, 16)
SCENARIOS = (
    ("no_match", 16),
    ("recent_match_8", 8),
    ("recent_match_16", 16),
    ("recent_match_32", 16),
    ("recent_match_256", 16),
    ("deep_match_256", 16),
    ("boilerplate_16", 16),
)
FLUSH = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")


def timed(launch, replays: int = 21, flush: bool = True) -> tuple[float, float]:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    samples = []
    for _ in range(replays):
        if flush:
            FLUSH.zero_()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0)
    return round(statistics.median(samples), 2), round(min(samples), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_ctx = max(CONTEXTS)
    shape = (max(REQ_COUNTS), max_ctx + 64)
    history_buf = StagedWriteTensor(
        shape,
        dtype=torch.int32,
        device=torch.device("cuda"),
        uva_instead_of_gpu=True,
    )
    history = history_buf.gpu
    offsets16 = torch.arange(-16, 0).unsqueeze(0)

    with out_path.open("w", encoding="utf-8") as fh:
        for scenario, min_match in SCENARIOS:
            levels = tilelang_copy.num_levels(min_match)
            for num_reqs in REQ_COUNTS:
                idx = torch.arange(num_reqs, dtype=torch.int64, device="cuda")
                sources = torch.empty(
                    (num_reqs, tilelang_copy.CHUNKS, levels),
                    dtype=torch.int32,
                    device="cuda",
                )
                for ctx in CONTEXTS:
                    g = torch.Generator(device="cpu").manual_seed(
                        1234 + num_reqs * 17 + ctx
                    )
                    total_len = torch.zeros(
                        max(REQ_COUNTS), dtype=torch.int32, device="cuda"
                    )
                    for slot in range(num_reqs):
                        tokens = torch.randint(
                            10, VOCAB, (ctx,), generator=g, dtype=torch.int32
                        )
                        if scenario == "no_match":
                            pass
                        elif scenario == "recent_match_8":
                            pos = max(32, ctx - 200)
                            tokens[pos - 8 : pos] = tokens[ctx - 8 : ctx]
                            tokens[pos - 9] = (tokens[ctx - 9] + 1) % VOCAB
                        elif scenario == "recent_match_16":
                            pos = max(32, ctx - 500)
                            tokens[pos - 16 : pos] = tokens[ctx - 16 : ctx]
                            tokens[pos - 17] = (tokens[ctx - 17] + 1) % VOCAB
                        elif scenario == "recent_match_32":
                            pos = max(64, ctx - 1000)
                            tokens[pos - 32 : pos] = tokens[ctx - 32 : ctx]
                            tokens[pos - 33] = (tokens[ctx - 33] + 1) % VOCAB
                        elif scenario == "recent_match_256":
                            pos = max(300, ctx - 1000)
                            tokens[pos - 256 : pos] = tokens[ctx - 256 : ctx]
                        elif scenario == "deep_match_256":
                            start = int(
                                torch.randint(
                                    0, ctx // 4 + 1, (1,), generator=g
                                )
                            )
                            tail = min(600, ctx // 2)
                            tokens[ctx - tail :] = tokens[start : start + tail]
                        elif scenario == "boilerplate_16":
                            phrase = tokens[ctx - 16 : ctx].clone()
                            guard = int((tokens[ctx - 17].item() + 1) % VOCAB)
                            idxs = torch.arange(32, ctx - 320, 48)
                            tokens[idxs - 17] = guard
                            tokens[idxs.unsqueeze(1) + offsets16] = (
                                phrase.unsqueeze(0)
                            )
                        history[slot, :ctx].copy_(tokens.to(history.device))
                        total_len[slot] = ctx

                    def launch():
                        tilelang_copy.copy_match(
                            idx, history, total_len, sources, min_match, LOOKAHEAD
                        )

                    med, low = timed(launch)
                    rec = {
                        "label": args.label,
                        "scenario": scenario,
                        "min_match": min_match,
                        "reqs": num_reqs,
                        "ctx": ctx,
                        "uva": True,
                        "med_us": med,
                        "min_us": low,
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
