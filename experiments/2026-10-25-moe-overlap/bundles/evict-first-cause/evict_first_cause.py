#!/usr/bin/env python3
"""Why the evict-first GEMM's cp.async helper raises CUDA_ERROR_ILLEGAL_INSTRUCTION.

evict_first_diag.py traced the illegal instruction to the cp.async half of the change,
but cp_async_hint_diag.py ran every hinted cp.async form alone on GB10 with the right
bytes. The failing GEMM (512x512, gathered, coded scales, no K tail) uses the helper only
for the scale codes' 16-byte copy, so the difference is the context: TileLang's passes
know ``tl.ptx_cp_async`` and not an extern call. Each variant compiles that GEMM, saves
its generated CUDA, and runs in its own process (the error kills the context):

- release: TileLang's cp.async (the control);
- helper: the change's helper (createpolicy and the hint in one asm block);
- helper, no hint: the same extern call, a plain cp.async (no createpolicy, no hint);
- helper, policy operand: the hint with the policy word as a C operand, the form that
  ran alone.

Then it prints the generated-source diffs against the release.

Mounts: /b/moe_release.py (the release's moe.py), /b/moe_evict_first.py (the change's).
"""

import difflib
import json
import subprocess
import sys
from pathlib import Path

RELEASE = Path("/b/moe_release.py").read_text()
CHANGE = Path("/b/moe_evict_first.py").read_text()
TMA_HINT = '                eviction_policy="evict_first",\n'
assert CHANGE.count(TMA_HINT) == 1
HELPER = CHANGE.replace(TMA_HINT, "")

HINT16 = """        "{\\n"
        "  .reg .b64 policy;\\n"
        "  createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\\n"
        "  cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, policy;\\n"
        "}\\n" ::"r"(dst), "l"(gmem) : "memory");"""
HINT_N = """        "{\\n"
        "  .reg .b64 policy;\\n"
        "  createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\\n"
        "  cp.async.ca.shared.global.L2::cache_hint [%0], [%1], %2, policy;\\n"
        "}\\n" ::"r"(dst), "l"(gmem), "n"(N) : "memory");"""
assert HELPER.count(HINT16) == 1 and HELPER.count(HINT_N) == 1
PLAIN = HELPER.replace(
    HINT16,
    '        "cp.async.cg.shared.global [%0], [%1], 16;\\n" ::"r"(dst), "l"(gmem) : "memory");',
).replace(
    HINT_N,
    '        "cp.async.ca.shared.global [%0], [%1], %2;\\n" ::"r"(dst), "l"(gmem), "n"(N) : "memory");',
)
OPERAND_POLICY = (
    "  unsigned long long policy;\n"
    '  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));\n'
)
OPERAND = (
    HELPER.replace(
        "  unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));\n",
        "  unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));\n"
        + OPERAND_POLICY,
    )
    .replace(
        HINT16,
        '        "cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\\n" '
        '::"r"(dst), "l"(gmem), "l"(policy) : "memory");',
    )
    .replace(
        HINT_N,
        '        "cp.async.ca.shared.global.L2::cache_hint [%0], [%1], %2, %3;\\n" '
        '::"r"(dst), "l"(gmem), "n"(N), "l"(policy) : "memory");',
    )
)
assert PLAIN != HELPER and OPERAND != HELPER
VARIANTS = {
    "release": RELEASE,
    "helper": HELPER,
    "helper, no hint": PLAIN,
    "helper, policy operand": OPERAND,
}

RUN = r"""
import importlib.util, sys
import torch
path, out, source_out = sys.argv[1], sys.argv[2], sys.argv[3]
package = "vllm.models.deepseek_v4_1.tilelang"
spec = importlib.util.spec_from_file_location(package + ".moe_variant", path)
moe = importlib.util.module_from_spec(spec)
moe.__package__ = package
spec.loader.exec_module(moe)
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang.gemm import pack_scale_codes
from vllm.models.deepseek_v4_1.tilelang.linear import _calls
import tile_kernels
n, k, gather = 512, 512, True
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
x = (torch.randn((rows, k), generator=gen, device="cuda") * 0.5).bfloat16()
xq, sf = tile_kernels.quant.per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True)
gemm = moe.grouped_fp8_fp4_gemm(experts, n, k, topk, gather, 32, block_N, coded_scales=True)
open(source_out, "w").write(gemm.get_kernel_source())
c = torch.zeros((order.shape[0], n), dtype=torch.bfloat16, device="cuda")
gemm(xq, sf.contiguous().view(torch.uint32), tiles, moe.tile_codes(codes, block_N), row_base, order, tile_expert, c)
torch.cuda.synchronize()
torch.save(c[order >= 0].view(torch.int16).cpu(), out)
"""


def main() -> int:
    outputs, sources = {}, {}
    for index, (name, module) in enumerate(VARIANTS.items()):
        path = Path(f"/tmp/moe_cause_{index}.py")
        out, source = f"/tmp/moe_cause_{index}.pt", Path(f"/tmp/moe_cause_{index}.cu")
        path.write_text(module)
        script = Path(f"/tmp/moe_cause_run_{index}.py")
        script.write_text(RUN)
        run = subprocess.run(
            [sys.executable, str(script), str(path), out, str(source)],
            capture_output=True,
            text=True,
        )
        if source.exists():
            sources[name] = source.read_text()
        if run.returncode == 0:
            import torch

            outputs[name] = torch.load(out)
            same = (
                outputs[name].equal(outputs["release"])
                if "release" in outputs
                else None
            )
            outcome = {"ran": True, "bits_equal_release": same}
        else:
            lines = [line for line in run.stderr.splitlines() if "Error" in line]
            outcome = {"ran": False, "error": lines[-1] if lines else run.stderr[-400:]}
        print(json.dumps({"variant": name, **outcome}), flush=True)
    for name in ("helper", "helper, no hint", "helper, policy operand"):
        if name not in sources or "release" not in sources:
            continue
        diff = difflib.unified_diff(
            sources["release"].splitlines(),
            sources[name].splitlines(),
            "release",
            name,
            n=4,
            lineterm="",
        )
        body = [line for line in diff if not line.startswith(("---", "+++"))]
        print(
            f"\n===== generated source, release -> {name}: {len(body)} lines",
            flush=True,
        )
        print("\n".join(body[:400]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
