"""Host time of BlockTables.apply_staged_writes() on a decode step that allocates
no block, in the release test's seven-group geometry (test_gpu_tilelang_inputs's
_block_tables), with the step's append_block_ids(overwrite=False, empty) for each
request first, as update_requests does."""
import argparse
import json
import statistics
import time

import torch

from vllm.v1.worker.gpu.block_table import BlockTables

p = argparse.ArgumentParser()
p.add_argument("--label", required=True)
p.add_argument("--output", required=True)
p.add_argument("--iters", type=int, default=20000)
args = p.parse_args()

dev = torch.device("cuda")
tables = BlockTables(
    block_sizes=[256, 128, 256, 64, 8, 256, 16],
    max_num_reqs=16,
    max_num_batched_tokens=16384,
    max_num_blocks_per_group=[4096, 33, 1024, 7, 1, 64, 3],
    device=dev,
    kernel_block_sizes=[64, 128, 256, 64, 8, 128, 16],
)
empty = tuple([] for _ in range(7))
for r in range(16):
    tables.append_block_ids(r, tuple([r + 1] for _ in range(7)), overwrite=True)
tables.apply_staged_writes()
torch.cuda.synchronize()
out = {"label": args.label, "groups": 7, "iters": args.iters, "rows": {}}
for num_reqs in (1, 8, 16):
    def step():
        for r in range(num_reqs):
            tables.append_block_ids(r, empty, overwrite=False)
        tables.apply_staged_writes()

    for _ in range(2000):
        step()
    samples = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(args.iters):
            step()
        samples.append((time.perf_counter_ns() - t0) / args.iters / 1000)
    out["rows"][f"c{num_reqs}"] = {
        "mean_us": round(statistics.mean(samples), 3),
        "min_us": round(min(samples), 3),
        "max_us": round(max(samples), 3),
        "samples_us": [round(s, 3) for s in samples],
    }
    print(args.label, f"c{num_reqs}", out["rows"][f"c{num_reqs}"])
json.dump(out, open(args.output, "w"), indent=2)
