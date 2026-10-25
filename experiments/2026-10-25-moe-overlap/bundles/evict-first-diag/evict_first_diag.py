#!/usr/bin/env python3
"""Which part of the evict-first change raises CUDA_ERROR_ILLEGAL_INSTRUCTION on GB10.

The evict-first bundle's first grouped GEMM (no K tails) left the context with an illegal
instruction (2026-10-25, dgx4). Two hints changed: the TMA weight tiles' (TileLang's
``tl::tma_load<CacheHintSm90::EVICT_FIRST>``, a constant policy word) and the scale codes'
``cp.async`` (a ``createpolicy`` word). This runs the release's GEMM and three variants
(either hint alone, both), each in a fresh process (the error kills the context), at a
gate/up shape (no tails) and a down shape (K tails), and compares each variant's output
bits with the release's. It also prints the words ``createpolicy`` makes on this GPU, beside
TileLang's constants (EVICT_NORMAL 0x1000000000000000, EVICT_FIRST 0x12F0000000000000,
EVICT_LAST 0x14F0000000000000).

Mounts: /b/moe_release.py (the release's moe.py), /b/moe_evict_first.py (the change's).
"""
import json
import subprocess
import sys
from pathlib import Path

RELEASE = Path("/b/moe_release.py").read_text()
CHANGE = Path("/b/moe_evict_first.py").read_text()
TMA_HINT = '                eviction_policy="evict_first",\n'
TMA_DEFAULT = "                barrier=loaded[s],\n            )"
assert CHANGE.count(TMA_HINT) == 1 and RELEASE.count(TMA_DEFAULT) == 1
VARIANTS = {
    "release": RELEASE,
    "tma hint only": RELEASE.replace(
        TMA_DEFAULT, "                barrier=loaded[s],\n" + TMA_HINT + "            )"
    ),
    "cp.async hint only": CHANGE.replace(TMA_HINT, ""),
    "both (the change)": CHANGE,
}

RUN = r'''
import importlib.util, json, sys
import torch
path, out = sys.argv[1], sys.argv[2]
package = "vllm.models.deepseek_v4_1.tilelang"
spec = importlib.util.spec_from_file_location(package + ".moe_variant", path)
moe = importlib.util.module_from_spec(spec)
moe.__package__ = package
spec.loader.exec_module(moe)
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang.gemm import pack_scale_codes
from vllm.models.deepseek_v4_1.tilelang.linear import _calls
import tile_kernels
results = {}
for n, k, gather in ((512, 512, True), (384, 576, False)):
    experts, topk, rows, block_N = 32, 6, 12, 128
    gen = torch.Generator(device="cuda").manual_seed(22)
    packed = torch.randint(0, 256, (experts, n, k // 2), generator=gen, device="cuda", dtype=torch.uint8)
    base = torch.randint(110, 126, (experts, n, 1), generator=gen, device="cuda")
    exps = (base + torch.randint(0, 8, (experts, n, k // 32), generator=gen, device="cuda")).to(torch.uint8)
    tiles, tails = moe.tile_expert_weights(packed.view(torch.int8), block_N)
    codes, row_base = pack_scale_codes(exps)
    dispatch = moe.MoeDispatchV41Plan(experts, topk, 32, k, torch.int32, torch.bfloat16)
    prepare.declare(lambda: _calls(dispatch.kernels()), bind=dispatch.bind)
    ids = torch.stack([torch.randperm(experts, generator=gen, device="cuda")[:topk] for _ in range(rows)]).to(torch.int32)
    weights = torch.rand((rows, topk), generator=gen, device="cuda")
    specs = dispatch.workspace_specs(rows)
    layout = dispatch.dispatch(ids, weights, moe.carve(torch.empty(moe.workspace_bytes(specs), dtype=torch.uint8, device="cuda"), specs))
    order, tile_expert = layout["order"], layout["tile_expert"]
    a_rows = rows if gather else order.shape[0]
    x = (torch.randn((a_rows, k), generator=gen, device="cuda") * 0.5).bfloat16()
    xq, sf = tile_kernels.quant.per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True)
    gemm = moe.grouped_fp8_fp4_gemm(experts, n, k, topk, gather, 32, block_N, coded_scales=True)
    c = torch.zeros((order.shape[0], n), dtype=torch.bfloat16, device="cuda")
    extra = (tails,) if tails is not None else ()
    gemm(xq, sf.contiguous().view(torch.uint32), tiles, *extra, moe.tile_codes(codes, block_N), row_base, order, tile_expert, c)
    torch.cuda.synchronize()
    results[f"{n}x{k}"] = c[order >= 0].view(torch.int16).cpu()
torch.save(results, out)
'''

POLICIES = r'''
import json
import tilelang
import tilelang.language as T
import torch
HEADER = """
__device__ __forceinline__ long long diag_policy(int kind) {
  unsigned long long p;
  if (kind == 0) asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  else if (kind == 1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  else asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return (long long)p;
}
"""
@tilelang.jit(target="cuda")
def policies():
    @T.prim_func
    def diag_policies(out: T.Tensor((3,), T.int64)):
        with T.Kernel(1, threads=32) as bx:
            T.import_source(HEADER)
            tx = T.get_thread_binding()
            if tx < 3:
                out[tx] = T.call_extern("int64", "diag_policy", tx)
    return diag_policies
out = torch.zeros(3, dtype=torch.int64, device="cuda")
policies()(out)
torch.cuda.synchronize()
print(json.dumps({name: hex(v & 0xFFFFFFFFFFFFFFFF) for name, v in zip(("evict_normal", "evict_first", "evict_last"), out.tolist())}))
'''


def main() -> int:
    policy = subprocess.run([sys.executable, "-c", POLICIES], capture_output=True, text=True)
    print("createpolicy words:", policy.stdout.strip() or policy.stderr.strip()[-2000:], flush=True)
    outputs = {}
    for index, (name, source) in enumerate(VARIANTS.items()):
        path, out = Path(f"/tmp/moe_variant_{index}.py"), f"/tmp/moe_variant_{index}.pt"
        path.write_text(source)
        run = subprocess.run([sys.executable, "-c", RUN, str(path), out], capture_output=True, text=True)
        ok = run.returncode == 0
        if ok:
            import torch

            outputs[name] = torch.load(out)
        tail = "" if ok else (run.stderr.strip().splitlines() or [""])[-1]
        print(json.dumps({"variant": name, "ran": ok, "error": tail}), flush=True)
        if not ok:
            print(run.stderr[-3000:], flush=True)
    if "release" in outputs:
        for name, result in outputs.items():
            same = all(
                result[shape].equal(outputs["release"][shape]) for shape in result
            )
            print(json.dumps({"variant": name, "bits_equal_release": same}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
