#!/usr/bin/env python3
"""Replay the ATen scale-code preparation sequence, without loading a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["smoke", "index", "overlay", "pack", "tile"])
parser.add_argument("--large", action="store_true")
parser.add_argument("--out", required=True)
parser.add_argument("--wait-file")
parser.add_argument("--no-finalize", action="store_true")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

import torch

torch.cuda.init()
print(json.dumps({"stage": "cuda-ready", "torch": torch.__version__, "device": torch.cuda.get_device_name(),
                  "module_loading": os.environ.get("CUDA_MODULE_LOADING", "default")}), flush=True)
if args.wait_file:
    (out / "ready").write_text("ready\n")
    deadline = time.monotonic() + 120
    while not Path(args.wait_file).exists():
        if time.monotonic() > deadline:
            raise TimeoutError("debugger did not release workload")
        time.sleep(0.05)


def stage(name):
    torch.cuda.synchronize()
    print(json.dumps({"stage": name}), flush=True)


E, N, K = (384, 1152, 160) if args.large else (4, 64, 16)
if args.operation == "smoke":
    x = torch.arange(4096, device="cuda", dtype=torch.int32)
    result = x + 1
    assert torch.equal(result, torch.arange(1, 4097, device="cuda", dtype=torch.int32))
elif args.operation == "index":
    values = torch.full((E, N, K // 4), 0x55, device="cuda", dtype=torch.uint8)
    mask = torch.ones(E, device="cuda", dtype=torch.bool)
    stage("index-inputs")
    selected = values[mask]
    stage("gather")
    result = torch.zeros_like(values)
    result[mask] = selected
    stage("scatter")
    assert torch.equal(values, result)
elif args.operation == "tile":
    from vllm.models.deepseek_v4_1.tilelang.moe import tile_expert_weights
    k = 5120 if args.large else 512
    weights = torch.full((E, N, k // 2), 0x33, device="cuda", dtype=torch.int8)
    stage("tile-inputs")
    result, tail = tile_expert_weights(weights, 32)
    stage("tile-complete")
    assert tail is None
    assert result.numel() == weights.numel()
    assert bool((result == 0x33).all())
else:
    from vllm.models.deepseek_v4_1.tilelang.scale_codes import (
        narrow_code_offsets, overlay_byte_codes, pack_expert_scale_codes,
    )
    if args.operation == "pack":
        exponents = torch.full((E, N, K), 127, device="cuda", dtype=torch.uint8)
        stage("pack-inputs")
        packed = pack_expert_scale_codes(exponents)
        stage("pack-complete")
        assert bool((packed.narrow == 1).all())
        assert bool((packed.base == 127).all())
        result = packed.codes_2bit
        assert not bool(result.any())
    else:
        rows = E * (N // 32) * (K // 4)
        codes = torch.full((rows, 32), 0, device="cuda", dtype=torch.uint16)
        byte_codes = torch.full((rows, 32), 0x55, device="cuda", dtype=torch.uint8)
        narrow = torch.ones(E, device="cuda", dtype=torch.uint8)
        stage("overlay-inputs")
        code_view, result = overlay_byte_codes(codes, byte_codes, narrow)
        stage("overlay-complete")
        offsets = narrow_code_offsets(narrow)
        assert code_view.data_ptr() == result.data_ptr()
        assert result.numel() == E * N * (K // 4)
        assert bool((result == 0x55).all())
        assert torch.equal(offsets.to(torch.int32), 2 * torch.arange(E, device="cuda") + 1)
stage("validated")
raw = result.cpu().contiguous().numpy().tobytes()
receipt = {"operation": args.operation, "large": args.large, "dimensions": [E, N, K],
           "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "no_finalize": args.no_finalize}
(out / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
if args.no_finalize:
    # Explicitly excludes profiler exit/export from this loading-only test.
    os._exit(0)
