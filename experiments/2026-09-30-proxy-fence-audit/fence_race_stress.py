#!/usr/bin/env python3
"""Do other TMA-fed B12X kernels return wrong outputs beside co-resident work? (GPU; cluster stopped)

usage: fence_race_stress.py --target dense|bf16prefill|mhc|varlen [--rounds R]
                            [--calls C] [--sizes 256,1024]   (cwd: the B12X checkout)

The dense GEMM released each TMA stage while its last ldmatrix reads were
still in flight (0007). The r5m SASS shows the same pending shared loads at
the stage release of three more serving kernels, where the compiler scheduled
the consuming MMAs after the release: the BF16 prefill projection (TMA, two
stages), the mHC TF32 prefill projection (TMA, two stages, split-K 4) and the
contiguous varlen attention forward (V stages). This runs one target on a side
stream many times while the main stream runs, in turn: nothing; a
bandwidth-bound copy; the prepared B12X W4A8 routed MoE (DS4.1 TP3, top 6).
Every output is compared on the device with the result computed alone; the
counts of wrong calls are printed per mode. `dense` is gemm_race_stress.py's
shared-expert down projection, the known positive control.
"""
import os
import sys
from contextlib import ExitStack, contextmanager

sys.path.insert(0, os.getcwd())
os.environ.setdefault("B12X_W4A8_TINY_DECODE", "0")
os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")

import torch  # noqa: E402

from b12x.moe import fused_moe  # noqa: E402
from b12x.moe.fused_moe import _impl  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe  # noqa: E402
from tests._reference.helpers import make_tp_moe_fp4_binding  # noqa: E402


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


TARGET = arg("--target", "bf16prefill")
ROUNDS = int(arg("--rounds", "40"))
CALLS = int(arg("--calls", "100"))
SIZES = [int(s) for s in arg("--sizes", "256,1024").split(",")]
MOE_PER_ROUND = 24
E, H, I, TOPK, MOE_TOKENS = 384, 5120, 768, 6, 6
device = torch.device("cuda")
torch.manual_seed(20260930)


@contextmanager
def dense_calls(session):
    """gemm_race_stress.py's block-FP8 down projection (K 768 -> N 5120)."""
    from b12x.gemm import block_fp8_linear as bfl
    from tests.gemm.test_gemm_block_fp8_linear import _make_block_fp8_weight

    K, N = 768, 5120
    weight, scale = _make_block_fp8_weight(N, K, block_size=32)
    packed = bfl.pack_weight(weight, scale, block_size=(32, 32))
    plans, requests = {}, []
    for rows in SIZES:
        caps = bfl.Caps(device=device, max_tokens=rows, in_features=K, out_features=N,
                        source_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
                        block_size=(32, 32), output_mode="provided")
        plan = plans[rows] = bfl.plan(caps)
        source = torch.randn(rows, K, device=device, dtype=torch.bfloat16)

        def prepare(state, source=source):
            spec, = state.scratch.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            out = torch.empty((source.shape[0], N, 1), dtype=torch.bfloat16, device=device)
            binding = state.bind(scratch=scratch, source=source, packed_weight=packed, output=out)
            return PreparedCall(run=lambda: state.run_binding(binding), output=out,
                                owners=(scratch, binding))

        requests.append(plan.request(name=f"bfl-{rows}", prepare_call=prepare))
    session.prepare(tuple(requests))
    calls = {}
    for rows in SIZES:
        source = (torch.randn(rows, K, device=device, dtype=torch.bfloat16) * 0.25).contiguous()
        spec, = plans[rows].scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        out = torch.empty((rows, N, 1), dtype=torch.bfloat16, device=device)
        binding = bfl.bind(plans[rows], scratch=scratch, source=source, packed_weight=packed,
                           output=out)
        calls[f"down rows {rows}"] = (lambda b=binding: bfl.run(binding=b), (out,),
                                      (scratch, source, binding))
    yield calls


@contextmanager
def bf16prefill_calls(session):
    """Serving shapes: K 5120 -> N 384 (BF16 out), 512 and 1024 (FP32 out)."""
    from b12x.gemm.bf16_gemv._prefill import prefill_mm

    calls = {}
    for n, dtype in ((384, torch.bfloat16), (512, torch.float32), (1024, torch.float32)):
        weight = (torch.randn(n, 5120, device=device) / 64).bfloat16().contiguous()
        for rows in SIZES:
            x = torch.randn(rows, 5120, device=device).bfloat16().contiguous()
            out = torch.empty(rows, n, device=device, dtype=dtype)
            calls[f"n {n} rows {rows}"] = (
                lambda x=x, w=weight, o=out: prefill_mm(x, w, o), (out,), (x, weight))
    yield calls


@contextmanager
def mhc_calls(session):
    """Serving geometry: hidden 5120, split_k 80, 64x24x64 tiles, 2 stages, k_splits 4."""
    from b12x._lib.compile_plan import compile_only_launches, load_programs
    from b12x.norm.mhc._kernels import _run_mhc_prefill_tf32_project_launch as launch

    geometry = dict(tile_m=64, tile_n=24, tile_k=64, num_stages=2, num_m_warps=4,
                    num_n_warps=1, k_splits=4, split_fp32_fn=True)
    fn = (torch.randn(24, 4 * 5120, device=device) / 64).contiguous()
    calls = {}
    for rows in SIZES:
        out = (torch.randn(rows, 4, 5120, device=device) / 3).bfloat16().contiguous()
        partials = torch.zeros(rows, 80, 25, device=device)
        with compile_only_launches():
            program = launch(out=out, fn=fn, partials=partials, **geometry)
        load_programs(program)

        def run(out=out, partials=partials, program=program):
            launch(out=out, fn=fn, partials=partials, **geometry, _prepared=program)

        calls[f"rows {rows}"] = (run, (partials,), (out, fn, program))
    yield calls


@contextmanager
def varlen_calls(session):
    """Vision tower shape class: 16 heads of 64, one non-causal sequence."""
    from b12x.attention import varlen
    from b12x.preparation import require_prepared

    tensors, requests = {}, []
    for rows in SIZES:
        q, k, v = (torch.randn(rows, 16, 64, device=device).bfloat16() for _ in range(3))
        cu = torch.tensor([0, rows], dtype=torch.int32, device=device)
        declaration = varlen.plan(q, k, v, cu, max_seqlen_q=rows, max_seqlen_k=rows,
                                  causal=False, window_size=(-1, -1))

        def prepare(state, q=q, k=k, v=v, cu=cu, rows=rows):
            spec, = state.scratch_plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            binding = state.bind(scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                                 max_seqlen_q=rows, max_seqlen_k=rows, causal=False,
                                 window_size=(-1, -1))
            return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))

        tensors[rows] = (declaration, q, k, v, cu)
        requests.append(declaration.request(name=f"varlen-{rows}", prepare_call=prepare))
    session.prepare(tuple(requests))
    calls = {}
    for rows, (declaration, q, k, v, cu) in tensors.items():
        state = require_prepared(declaration, "attention.varlen")
        spec, = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = varlen.bind(declaration, scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                              max_seqlen_q=rows, max_seqlen_k=rows, causal=False,
                              window_size=(-1, -1))
        output, lse = varlen.run(binding)
        calls[f"tokens {rows}"] = (lambda b=binding: varlen.run(b), (output, lse),
                                   (scratch, binding, q, k, v, cu))
    yield calls


TARGETS = {"dense": dense_calls, "bf16prefill": bf16prefill_calls, "mhc": mhc_calls,
           "varlen": varlen_calls}

moe_weights = make_synthetic_mxfp4_moe(E, H, I, seed=7, device=device)
moe_plan = fused_moe.plan_weights(
    source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
                                  w13_layout=fused_moe.W13Layout("w13")),
    activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A8, nonlinearity="silu",
                                        io_dtype=torch.bfloat16),
    geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
)
experts = fused_moe.prepare_weights(plan=moe_plan, weights=fused_moe.PackedWeights(
    w13=moe_weights["w13_fp4"], w2=moe_weights["w2_fp4"], w13_block_scales=moe_weights["w13_mx"],
    w2_block_scales=moe_weights["w2_mx"], w13_global_scales=moe_weights["alphas"],
    w2_global_scales=moe_weights["alphas"], input_scale=moe_weights["input_scale"],
    intermediate_scale=moe_weights["input_scale"], immutable_input_scales=True))
gen = torch.Generator(device=device).manual_seed(606)
moe_x = (torch.randn(MOE_TOKENS, H, generator=gen, device=device) * 2.0).to(torch.bfloat16)
logits = torch.randn(MOE_TOKENS, E, generator=gen, device=device)
top_logits, topk_ids = torch.topk(logits, TOPK, dim=-1)
topk_weights = torch.softmax(top_logits, dim=-1).float().contiguous()
topk_ids = topk_ids.to(torch.int32).contiguous()
moe_out = torch.zeros(MOE_TOKENS, H, dtype=torch.bfloat16, device=device)
hog_src = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
hog_dst = torch.empty_like(hog_src)
side = torch.cuda.Stream()
main = torch.cuda.current_stream()

with ExitStack() as stack:
    session = stack.enter_context(
        PreparationSession(device=device, autotune=False, compile_workers=2))
    calls = stack.enter_context(TARGETS[TARGET](session))
    moe_binding = stack.enter_context(make_tp_moe_fp4_binding(
        a=moe_x, experts=experts, topk_weights=topk_weights, topk_ids=topk_ids,
        output=moe_out, input_scales_static=True, quant_mode="w4a8_mx"))
    failures = 0
    for name, (run, outputs, _owners) in calls.items():
        run()
        torch.cuda.synchronize()
        refs = [o.clone() for o in outputs]
        stable = True
        for _ in range(5):
            run()
            stable &= all(torch.equal(o, r) for o, r in zip(outputs, refs, strict=True))
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(50):
            run()
        start.record()
        for _ in range(500):
            run()
        stop.record()
        torch.cuda.synchronize()
        alone_us = start.elapsed_time(stop) * 1000 / 500
        results = {}
        for mode in ("alone", "copy", "moe"):
            wrong = torch.zeros((), dtype=torch.int64, device=device)
            for _ in range(ROUNDS):
                side.wait_stream(main)
                if mode == "copy":
                    for _ in range(4):
                        hog_dst.copy_(hog_src)
                elif mode == "moe":
                    for _ in range(MOE_PER_ROUND):
                        _impl.b12x_moe_fp4(binding=moe_binding)
                with torch.cuda.stream(side):
                    for _ in range(CALLS):
                        run()
                        bad = torch.zeros((), dtype=torch.bool, device=device)
                        for o, r in zip(outputs, refs, strict=True):
                            bad |= torch.ne(o, r).any()
                        wrong += bad
                main.wait_stream(side)
            torch.cuda.synchronize()
            results[mode] = (int(wrong), ROUNDS * CALLS)
            failures += int(wrong)
        line = "; ".join(f"{m}: wrong {w}/{t}" for m, (w, t) in results.items())
        print(f"{TARGET} {name}: alone-stable {stable}, {alone_us:.2f} us/call alone; {line}",
              flush=True)
    print("FAIL" if failures else "OK", flush=True)
    sys.exit(1 if failures else 0)
