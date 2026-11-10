"""#957 part 2 (0797): the sparse-MLA decode plan (stage 2) at TP3 and TP4 rank shapes.

usage: python3 /h/h957p2.py --boot K --out FILE

The PR's bench_sparse_mla_tp3 workload (screen_tp3_decode.py at e464d364): SWA width 128,
index width 512, page 128, max_rows 48, q/KV random, sinks 0, positions 1024, with the
container's own SparseMlaV41Plan (candidate-54's, or 0797's). Heads 24 (TP3) and 16 (TP4).
Two index-length mixes: the PR's [128, 256, 384, 512] (inactive segments in 3 of 4 rows)
and all 512 (every segment active, a long context). The plan runs inside the decode CUDA
graph in serving, so CUDA-graph replay (20 calls a graph) decides; KV bytes are drawn from
0..119 (the PR drew 0..254), so no FP8 code or scale is NaN and outputs compare bit for bit; CUPTI kernel time and
eager timing are beside it. Output bytes are hashed so the arms can be compared.
"""

import argparse
import json

import torch

from omlib import env_info, eager_us, gpu_warmup, graph_hot_us, h, kernel_us

ROWS = (4, 6, 8, 12, 16, 24, 32, 48)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from vllm.models.deepseek_v4_1.tilelang.sparse_mla import (
        HEAD_DIM, IDX_RECORD_BYTES, SWA_RECORD_BYTES, SparseMlaV41Plan)
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda", 0)
    init_workspace_manager(device)
    warm = gpu_warmup()
    out = {"boot": args.boot, "env": env_info(), "gpu_warmup_ms": warm, "plans": {}, "cells": {}}
    swa_width, idx_width, page_size, max_rows, num_pages = 128, 512, 128, 48, 64
    for tp, heads in ((3, 24), (4, 16)):
        plan = SparseMlaV41Plan(heads, swa_width, idx_width, page_size, swa_page_size=page_size,
                                max_rows=max_rows, device=device)
        out["plans"][f"tp{tp}"] = {"heads": heads, "split_max_rows": getattr(plan, "split_max_rows", None),
                                   "decode_max_rows": getattr(plan, "decode_max_rows", None)}
        g = torch.Generator(device=device).manual_seed(5700 + heads)
        q = torch.randn((max_rows, heads, HEAD_DIM), generator=g, dtype=torch.float32, device=device).bfloat16()
        swa_kv = torch.randint(0, 120, (num_pages, page_size * SWA_RECORD_BYTES), generator=g, dtype=torch.uint8,
                               device=device)
        idx_kv = torch.randint(0, 120, (num_pages, page_size * IDX_RECORD_BYTES), generator=g, dtype=torch.uint8,
                               device=device)
        swa_indices = torch.ones((max_rows, swa_width), dtype=torch.int32, device=device)
        swa_lens = torch.full((max_rows,), 128, dtype=torch.int32, device=device)
        idx_indices = torch.ones((max_rows, idx_width), dtype=torch.int32, device=device)
        idx_page_table = torch.ones((max_rows, 32), dtype=torch.int32, device=device)
        sinks = torch.zeros((heads,), dtype=torch.float32, device=device)
        positions = torch.full((max_rows,), 1024, dtype=torch.int64, device=device)
        cos_sin = torch.ones((4096, 64), dtype=torch.float32, device=device)
        o = torch.empty((max_rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
        partial = torch.empty((max_rows, 5, heads, HEAD_DIM), dtype=torch.float32, device=device)
        stats = torch.empty((max_rows, 5, heads, 2), dtype=torch.float32, device=device)
        mixes = {
            "mixed": torch.tensor([128, 256, 384, 512] * (max_rows // 4), dtype=torch.int32, device=device),
            "full": torch.full((max_rows,), 512, dtype=torch.int32, device=device),
        }
        for mix, idx_lens in mixes.items():
            for r in ROWS:
                def run(r=r, idx_lens=idx_lens):
                    plan.run(q[:r], swa_kv, swa_indices[:r], swa_lens[:r], sinks, positions[:r], cos_sin, o[:r],
                             idx_cache=idx_kv, idx_indices=idx_indices[:r], idx_lengths=idx_lens[:r],
                             idx_page_table=idx_page_table[:r], scratch=(partial[:r], stats[:r]))
                o.zero_()
                run()
                torch.cuda.synchronize()
                cell = {
                    "route": plan.kernel_for(r),
                    "out_hash": h(o[:r]),
                    "graph_hot": graph_hot_us(run),
                    "kernel": kernel_us(run),
                    "eager_us": eager_us(run),
                }
                cell["kernel"].pop("kernels", None)
                key = f"tp{tp}_{mix}_m{r}"
                out["cells"][key] = cell
                print(key, cell["route"], cell["graph_hot"]["median_us"], cell["kernel"]["per_call_us"],
                      cell["eager_us"], cell["out_hash"], flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
