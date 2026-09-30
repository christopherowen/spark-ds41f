#!/usr/bin/env python3
"""Does a block-FP8 linear give the same bits alone and beside other work? (GPU; cluster stopped)

usage: gemm_concurrency_check.py [--reps N] [--alone-only]   (cwd: the B12X checkout)

DeepSeek V4.1 Flash shared-expert projections at TP3 with 32x32 block-FP8
weights: down (6 rows, K 768 -> N 5120) and gate_up (K 5120 -> N 1536), at
the capacities decode plans use (6 and 8). Each runs N times alone, then N
times on a side stream right after a long matmul is queued on another stream,
and N times under CUDA-graph replay beside the same matmul. Every output must
equal the first.
"""
import sys
from contextlib import contextmanager

sys.path.insert(0, ".")

import torch  # noqa: E402

from b12x.gemm import block_fp8_linear as bfl  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from tests.gemm.test_gemm_block_fp8_linear import _make_block_fp8_weight  # noqa: E402

REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 20
ALONE_ONLY = "--alone-only" in sys.argv  # for compute-sanitizer: skip the hog matmuls


@contextmanager
def prepared(source, packed, capacity):
    caps = bfl.Caps(device=source.device, max_tokens=capacity, in_features=source.shape[1],
                    out_features=packed.out_features, source_dtype=source.dtype,
                    output_dtype=source.dtype, block_size=packed.block_size,
                    output_mode="provided")
    plan = bfl.plan(caps)

    def prepare(state):
        spec, = state.scratch.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=source.device)
        output = torch.empty((source.shape[0], packed.out_features, 1),
                             dtype=source.dtype, device=source.device)
        binding = state.bind(scratch=scratch, source=source, packed_weight=packed, output=output)
        return PreparedCall(run=lambda: state.run_binding(binding), output=output,
                            owners=(scratch, binding))

    with PreparationSession(device=source.device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="block-fp8", prepare_call=prepare),))
        session.freeze()
        yield plan


failures = 0
torch.manual_seed(20260930)
hog_a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
hog_b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
side = torch.cuda.Stream()
main = torch.cuda.current_stream()
for name, (n, k) in {"down": (5120, 768), "gate_up": (1536, 5120)}.items():
    weight, scale = _make_block_fp8_weight(n, k, block_size=32)
    packed = bfl.pack_weight(weight, scale, block_size=(32, 32))
    for capacity in (6, 8):
        source = (torch.randn(6, k, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
        with prepared(source, packed, capacity) as plan:
            spec, = plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device="cuda")
            output = torch.empty((6, n, 1), dtype=torch.bfloat16, device="cuda")
            binding = bfl.bind(plan, scratch=scratch, source=source, packed_weight=packed,
                               output=output)
            outs = {"alone": [], "beside": [], "graph": []}
            for _ in range(REPS):
                bfl.run(binding=binding)
                torch.cuda.synchronize()
                outs["alone"].append(output.clone())
            for _ in range(0 if ALONE_ONLY else REPS):
                side.wait_stream(main)
                hog = torch.mm(hog_a, hog_b)  # queued on the main stream
                with torch.cuda.stream(side):
                    bfl.run(binding=binding)
                torch.cuda.synchronize()
                outs["beside"].append(output.clone())
                del hog
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(side):
                with torch.cuda.graph(graph, stream=side):
                    bfl.run(binding=binding)
            for _ in range(0 if ALONE_ONLY else REPS):
                side.wait_stream(main)
                hog = torch.mm(hog_a, hog_b)
                with torch.cuda.stream(side):
                    graph.replay()
                torch.cuda.synchronize()
                outs["graph"].append(output.clone())
                del hog
            first = outs["alone"][0]
            summary = {key: sum(torch.equal(o, first) for o in values) for key, values in outs.items()}
            worst = max((o.float() - first.float()).abs().max().item()
                        for values in outs.values() for o in values)
            print(f"{name} capacity {capacity}: equal to first {summary} of {REPS} each, "
                  f"max diff {worst:.3g}", flush=True)
            failures += any(v != len(outs[key]) for key, v in summary.items())
            del graph
print("FAIL" if failures else "OK", flush=True)
sys.exit(1 if failures else 0)
