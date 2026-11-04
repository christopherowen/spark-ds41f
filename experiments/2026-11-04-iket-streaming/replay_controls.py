#!/usr/bin/env python3
"""Numerical replays for the two formerly unbounded phase manifests."""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["bounds"])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--stream", action="store_true")
parser.add_argument("--native", action="store_true")
args = parser.parse_args()
import torch

collector = None
if args.stream:
    from stream.collector import enable

    collector = enable(
        args.out / "stream", ring_bytes=4096, chunk_bytes=16384, patched=True
    )
elif args.native:
    from stream.phase import load

    iket = load(args.out)
    iket.enable(
        args.out / "native-sources",
        ctas=1,
        iters=1,
        warps=1,
        granularity="phase",
        payloads=True,
    )
from vllm.v1.worker.gpu.sample.tilelang_bias import apply_logit_bias
from vllm.v1.worker.gpu.tilelang_inputs import compute_slot_mappings


def tensor(value, dtype):
    return torch.tensor(value, dtype=dtype, device="cuda")


i32, i64 = torch.int32, torch.int64
vocab = 129280
base = torch.full((2, vocab), -float("inf"), dtype=torch.float32)
base[:, 0] = 3
base[0, -1] = 7
expected = base.clone()
expected[0, 0] = -float("inf")
expected[1, 0] = 5
logits = base.cuda()
for _ in range(10):
    logits.copy_(base)
    apply_logit_bias(
        logits,
        tensor([0, 1], i32),
        tensor([0, 0], i64),
        tensor([0, 0], i32),
        tensor([[0], [0]], i32),
        tensor([1, 1], i32),
        tensor([[0], [0]], i32),
        tensor([[2.0], [2.0]], torch.float32),
        tensor([2, 2], i32),
        tensor([1, 1], i32),
        tensor([1, 1], i32),
        tensor([[0], [0]], i32),
        True,
    )
    assert torch.equal(logits.cpu(), expected)

requests, live, padded = 16, 96, 128
idx_cpu = torch.arange(requests - 1, -1, -1, dtype=i32)
positions_cpu = torch.arange(padded, dtype=i64) % 6 + 128
starts = tensor(list(range(0, live + 1, 6)), i32)
tables_cpu = [
    torch.arange(128, dtype=i32).reshape(16, 8) + offset for offset in [10, 1010]
]
tables = [table.cuda() for table in tables_cpu]
expected_slots = torch.full((2, padded), -1, dtype=i64)
for group, table in enumerate(tables_cpu):
    for token in range(live):
        expected_slots[group, token] = int(
            table[int(idx_cpu[token // 6]), 2]
        ) * 64 + int(positions_cpu[token] % 64)
slots = torch.empty((2, padded), dtype=i64, device="cuda")
for _ in range(10):
    slots.fill_(666)
    compute_slot_mappings(
        idx_cpu.cuda(),
        starts,
        positions_cpu.cuda(),
        tensor([table.data_ptr() for table in tables], i64),
        tensor([8, 8], i64),
        torch.full((2, requests), 8, dtype=i32, device="cuda"),
        tensor([64, 64], i32),
        tensor([64, 64], i32),
        tensor([True, True], torch.bool),
        tensor([False, False], torch.bool),
        tensor([1, 1], i32),
        slots,
        0,
        1,
        1,
    )
    assert torch.equal(slots.cpu(), expected_slots)
torch.cuda.synchronize()
result = {
    "operation": "bounds",
    "stream": args.stream,
    "native": args.native,
    "bias_sha256": hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),
    "slots_sha256": hashlib.sha256(slots.cpu().numpy().tobytes()).hexdigest(),
}
if collector:
    collector.close()
    from stream.export import export

    result["trace"] = export(args.out / "stream", args.out / "trace.json")
source = args.out / ("stream/sources" if args.stream else "native-sources/sources")
if source.exists():
    bounds = {}
    for file in source.glob("*.json"):
        manifest = json.loads(file.read_text())
        values = [s["events_per_warp_bound"] for s in manifest["sites"] if "range" in s]
        assert values and all(isinstance(value, int) for value in values)
        bounds[manifest["kernel"]] = max(values)
    result["bounds"] = bounds
(args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
