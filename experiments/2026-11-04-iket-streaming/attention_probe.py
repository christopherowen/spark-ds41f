#!/usr/bin/env python3
"""Replay real vision attention with a known uniform-softmax reference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["attention"])
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
import torch
from stream.collector import enable
from stream.export import records

collector = enable(args.out / "stream", patched=True)
from vllm.models.deepseek_v4_1.tilelang.vision import attention_kernel

rows, heads, dim = 8704, 16, 64
qkv = torch.zeros((rows, 3 * heads * dim), device="cuda", dtype=torch.bfloat16)
values = (torch.arange(heads * dim) % 13 + 1).float().div(16).bfloat16()
qkv[:, 2 * heads * dim :] = values.cuda()
out = torch.empty((rows, heads * dim), device="cuda", dtype=torch.bfloat16)
kernel = attention_kernel(heads, dim)
for _ in range(5):
    kernel(qkv, out)
torch.cuda.synchronize()
assert torch.equal(out.cpu(), values.expand(rows, -1))
collector.close()
counts = Counter()
for row in records(args.out / "stream"):
    counts["records"] += 1
    counts[f"kind_{row[9]}"] += 1
assert counts["kind_0"] == counts["kind_3"] and counts["kind_1"] == counts["kind_2"]
result = {
    "operation": "attention",
    "sha256": hashlib.sha256(out.float().cpu().numpy().tobytes()).hexdigest(),
    "trace": dict(counts),
}
(args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
