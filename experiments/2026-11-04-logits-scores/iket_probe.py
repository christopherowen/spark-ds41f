#!/usr/bin/env python3
"""Record actual IKET loads/stores for masked and unmasked selected logits."""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

import torch  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang import iket  # noqa: E402

iket.enable(args.out, ctas=1, iters=1, warps=1, granularity="phase")
from vllm.v1.worker.gpu.sample import tilelang_logprob  # noqa: E402 -- enable before compilation.

receipts = []
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    for masked in (False, True):
        logits = torch.arange(3 * 257, device="cuda", dtype=torch.float32).view(3, 257)
        logits = (logits / 8).to(dtype)
        ids = torch.arange(33, device="cuda").expand(3, -1)
        valid = (ids % 3 != 0) if masked else None
        expected = logits.gather(1, ids).float()
        if masked:
            expected.masked_fill_(~valid, -torch.inf)
        actual = tilelang_logprob.gather_logits(logits, ids, valid)
        assert torch.equal(actual, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = tilelang_logprob.gather_logits(logits, ids, valid)
        graph.replay()
        assert torch.equal(replayed, expected)
        receipts.append(
            dict(
                dtype=str(dtype),
                masked=masked,
                sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),
            )
        )
torch.cuda.synchronize()
(args.out / "numerics.json").write_text(json.dumps(receipts, indent=2) + "\n")
