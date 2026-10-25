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

Each runs at the gate/up shape (512x512, gathered, no K tail) and the TP4 down shape
(384x576, a 64-wide K tail: the first failing test's). Two more variants put the helper
on one copy only (the K tail's 8 bytes or the scale codes' 16), TileLang's cp.async on
the other. Then it prints the down shape's generated-source diffs against the release.

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
TAIL_HELPER = """                        _cp_async_evict_first(
                            T.access_ptr(B_shared[s, n, g * 16], "w", 8),"""
TAIL_RELEASE = """                        T.ptx_cp_async(
                            T.access_ptr(B_shared[s, n, g * 16], "w", 8),"""
CODES_HELPER = """                _cp_async_evict_first(
                    T.access_ptr(SFB_shared[s, tx * 8], "w", 8),
                    T.access_ptr(
                        SFB[(expert * n_blocks + bx) * k_blocks + ko, tx * 8], "r", 8
                    ),
                    16,  # eight 2-byte codes"""
CODES_RELEASE = """                T.ptx_cp_async(
                    T.access_ptr(SFB_shared[s, tx * 8], "w", 8),
                    T.access_ptr(
                        SFB[(expert * n_blocks + bx) * k_blocks + ko, tx * 8], "r", 8
                    ),
                    8,  # elements: eight 2-byte codes, one 16-byte copy"""
assert HELPER.count(TAIL_HELPER) == 1 and HELPER.count(CODES_HELPER) == 1
assert RELEASE.count(TAIL_RELEASE) == 1 and RELEASE.count(CODES_RELEASE) == 1
VARIANTS = {
    "release": RELEASE,
    "helper": HELPER,
    "helper, no hint": PLAIN,
    "helper, policy operand": OPERAND,
    "helper, tail only": HELPER.replace(CODES_HELPER, CODES_RELEASE),
    "helper, codes only": HELPER.replace(TAIL_HELPER, TAIL_RELEASE),
}
SHAPES = ((512, 512, True), (384, 576, False))
DIFFS = ("helper, tail only", "helper, no hint", "helper, codes only")

RUN = r"""
import importlib.util, sys
import torch
path, out, source_out = sys.argv[1], sys.argv[2], sys.argv[3]
n, k, gather = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] == "True"
package = "vllm.models.deepseek_v4_1.tilelang"
spec = importlib.util.spec_from_file_location(package + ".moe_variant", path)
moe = importlib.util.module_from_spec(spec)
moe.__package__ = package
spec.loader.exec_module(moe)
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang.gemm import pack_scale_codes
from vllm.models.deepseek_v4_1.tilelang.linear import _calls
import tile_kernels
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
open(source_out, "w").write(gemm.get_kernel_source())
c = torch.zeros((order.shape[0], n), dtype=torch.bfloat16, device="cuda")
extra = (tails,) if tails is not None else ()
gemm(xq, sf.contiguous().view(torch.uint32), tiles, *extra, moe.tile_codes(codes, block_N), row_base, order, tile_expert, c)
torch.cuda.synchronize()
torch.save(c[order >= 0].view(torch.int16).cpu(), out)
"""


def main() -> int:
    sources = {}
    for n, k, gather in SHAPES:
        shape = f"{n}x{k}"
        outputs = {}
        for index, (name, module) in enumerate(VARIANTS.items()):
            tag = f"{index}_{shape}"
            path, script = (
                Path(f"/tmp/moe_cause_{tag}.py"),
                Path(f"/tmp/moe_cause_run_{tag}.py"),
            )
            out, source = f"/tmp/moe_cause_{tag}.pt", Path(f"/tmp/moe_cause_{tag}.cu")
            path.write_text(module)
            script.write_text(RUN)
            source.unlink(missing_ok=True)
            run = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(path),
                    out,
                    str(source),
                    str(n),
                    str(k),
                    str(gather),
                ],
                capture_output=True,
                text=True,
            )
            if source.exists():
                sources[name, shape] = source.read_text()
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
                outcome = {
                    "ran": False,
                    "error": lines[-1] if lines else run.stderr[-400:],
                }
            print(json.dumps({"shape": shape, "variant": name, **outcome}), flush=True)
    shape = "384x576"
    for name in DIFFS:
        if (name, shape) not in sources or ("release", shape) not in sources:
            continue
        diff = difflib.unified_diff(
            sources["release", shape].splitlines(),
            sources[name, shape].splitlines(),
            "release",
            name,
            n=6,
            lineterm="",
        )
        body = [line for line in diff if not line.startswith(("---", "+++"))]
        print(
            f"\n===== {shape} generated source, release -> {name}: {len(body)} lines",
            flush=True,
        )
        print("\n".join(body[:400]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
