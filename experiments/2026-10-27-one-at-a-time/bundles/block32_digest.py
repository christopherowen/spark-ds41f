"""Output digests of the block-32 FP8 projections, for a bit-for-bit comparison of two trees.

Run once on candidate 9's image and once with a change's files overlaid; the two runs
must print the same digests. Inputs are seeded on the GPU, so both runs see the same
weights, scales and activations. Each line is one (shape, rows) case; the last line
is the digest over all of them.
"""

import hashlib
import json

import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1.tilelang.linear import TileLangBlock32Rows
from vllm.v1.worker.workspace import use_preallocated_workspace

DEVICE = "cuda"
# TP4 rank shapes (N, K) of the block-32 projections, and the first row each GEMM reads.
SHAPES = [
    ("Q-B", 8192, 1280, 0),
    ("fused Q-A/KV", 1792, 5120, 0),
    ("DSpark context KV", 1792, 5120, 1280),
    ("shared gate/up", 1152, 5120, 0),
    ("shared down", 5120, 576, 0),
    ("indexer Q-B", 4096, 1280, 0),
    ("WO-B", 5120, 2048, 0),
]
ROWS = (*range(1, 17), 24, 32, 33, 64, 65, 96, 128, 200, 300, 1024)


def gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def weight(n, k, seed):
    g = gen(seed)
    w = (torch.randn((n, k), generator=g, device=DEVICE) * 2).to(torch.float8_e4m3fn)
    exps = torch.randint(110, 131, (n // 32, k // 32), generator=g, device=DEVICE).to(torch.uint8)
    return w, exps


def main():
    total = hashlib.sha256()
    scratch = torch.empty(64 << 20, dtype=torch.uint8, device=DEVICE)
    with set_current_vllm_config(VllmConfig()):
        for i, (label, n, k, start) in enumerate(SHAPES):
            layer = nn.Module()
            layer.weight, layer.weight_scale_inv = weight(n, k, 100 + i)
            gemm = TileLangBlock32Rows(layer, start)
            gemm.declare_kernels()
            x = torch.randn((max(ROWS), k), generator=gen(200 + i), device=DEVICE).bfloat16()
            with use_preallocated_workspace(scratch):
                for rows in ROWS:
                    out = gemm(x[:rows])
                    torch.cuda.synchronize()
                    digest = hashlib.sha256(out.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                    total.update(digest.encode())
                    print(json.dumps({"case": label, "n": n - start, "k": k, "rows": rows,
                                      "sha256": digest}), flush=True)
    print(json.dumps({"summary": "block32 digests", "cases": len(SHAPES) * len(ROWS),
                      "sha256": total.hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
