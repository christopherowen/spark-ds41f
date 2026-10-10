"""Port C3: the rank-order reduce-scatter's local sum at TP4 sequence-parallel
chunk sizes (each rank owns ceil(T / 4) rows of 5120 BF16; T from 205 to 8192).

1. Bits: the TileLang sum equals FP32 additions in rank order rounded once (the
   one-shot all-reduce's arithmetic) on parts that are views of one exchange
   buffer, with this rank's chunk read in place.
2. Repeatability: graph replays, warm and L2-evicted, give the same bits.
3. Time per call under CUDA graphs, warm and cold, beside the torch sequence it
   replaces (separate conversions and additions) and a BF16 add chain (about the
   arithmetic NCCL's reduce-scatter does on the GPU).

Exits 1 if bits differ or a replay differs.
"""
import sys

import torch

sys.path.insert(0, "/b")
from kbench import DEVICE, REPLAYS, per_call, repeatable  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.collectives import rank_order_sum  # noqa: E402

WORLD, HIDDEN = 4, 5120
GEN = torch.Generator(device=DEVICE).manual_seed(5)
failures = []


def reference(parts):
    total = parts[0].float()
    for part in parts[1:]:
        total += part.float()
    return total.bfloat16()


print("rows per rank: us warm/cold  tilelang | torch fp32 sequence | bf16 add chain", flush=True)
for tokens in (205, 512, 1024, 2048, 4144, 8192):
    rows = -(-tokens // WORLD)
    exchange = torch.randn((WORLD, rows, HIDDEN), device=DEVICE, generator=GEN).bfloat16()
    own = torch.randn((rows, HIDDEN), device=DEVICE, generator=GEN).bfloat16()
    rank = 1
    parts = [own if r == rank else exchange[r] for r in range(WORLD)]
    out = torch.empty_like(own)
    rank_order_sum(parts, out)
    torch.cuda.synchronize()
    if not torch.equal(out.view(torch.int16), reference(parts).view(torch.int16)):
        failures.append(f"{tokens} tokens: bits differ")
    if repeatable(lambda parts=parts, out=out: rank_order_sum(parts, out), lambda out=out: [out]):
        failures.append(f"{tokens} tokens: not repeatable over {REPLAYS} replays")
    chain = torch.empty_like(own)

    def bf16_chain(parts=parts, chain=chain):
        torch.add(parts[0], parts[1], out=chain)
        chain.add_(parts[2])
        chain.add_(parts[3])

    t = per_call(lambda parts=parts, out=out: rank_order_sum(parts, out), rows)
    r = per_call(lambda parts=parts: reference(parts), rows)
    c = per_call(bf16_chain, rows)
    print(f"  {rows:5d} ({tokens:5d} tokens): {t[0]:7.1f}/{t[1]:7.1f} | {r[0]:7.1f}/{r[1]:7.1f} | {c[0]:7.1f}/{c[1]:7.1f}",
          flush=True)

for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
