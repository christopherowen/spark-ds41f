"""Compare boundary choices using candidate-38's unchanged serving kernels.

Print raw JSON lines. No timing-derived dispatch decisions or source mutations.
"""
import json
import random
import statistics
from types import SimpleNamespace

import torch
from cuda.bindings import driver as cu

from vllm.models.deepseek_v4_1.tilelang import pdl
from vllm.models.deepseek_v4_1.tilelang.cache_writer import RECORD_BYTES, write_cache
from vllm.models.deepseek_v4_1.tilelang.front import attention_front
from front_variants import front_variant
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    WIDE_DECODE, pack_scale_words, pack_weight_tiles,
)
from vllm.models.deepseek_v4_1.tilelang.linear import (
    _block32_gemm, _block32_kernels, _launch, empty_mxfp8,
)
from vllm.models.deepseek_v4_1.tilelang.rope import rope_
from vllm.models.glm5next.nvidia.tilelang.l2_prefetch import l2_prefetch_kernel

ROWS = (1, 6, 9, 16, 24, 48, 96)
VARIANTS = ("fused", "fused64", "fused128", "fused256", "split_qkv", "split_kvq", "split_parallel", "split_q", "standalone", "projection_only")
MODES = ("l2", "cold", "prefetch")
EPS, PAGE, PAGES = 1e-20, 256, 2


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def byte_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def capture(fn, calls=1):
    side = WARMUP_STREAM
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=CAPTURE_STREAM):
        for _ in range(calls):
            fn()
    return graph


def make_case(n, m, padded):
    gen = torch.Generator(device="cuda").manual_seed(517)
    # Generate maximum rows first so every row has the same input in every batch.
    raw = (torch.randn((96, 1792), generator=gen, device="cuda") * 3).bfloat16()[:m]
    qw = (torch.randn(1280, generator=gen, device="cuda") * .3 + 1).bfloat16()
    kw = (torch.randn(512, generator=gen, device="cuda") * .3 + 1).bfloat16()
    positions = torch.arange(m, device="cuda", dtype=torch.int64) + 100
    positions[0] = 0
    if m > 1: positions[1] = 4095
    inv = 1 / (10000 ** (torch.arange(0, 64, 2, device="cuda").double() / 64))
    angles = torch.arange(4096, device="cuda").double()[:, None] * inv
    table = torch.cat((angles.cos(), angles.sin()), -1).float()
    slots = torch.arange(m, device="cuda", dtype=torch.int64) * 2
    slots[0] = 255
    if m > 1: slots[1] = 256
    slots[-1] = 511
    # Q-B: TP3's final rank has 16 real heads and 8 zero-weight padded heads.
    weight = (torch.randn((n, 1280), generator=gen, device="cuda") * .2).to(torch.float8_e4m3fn)
    if padded:
        weight[8192:].zero_()
    exps = torch.randint(120, 125, (n // 32, 40), generator=gen, device="cuda").to(torch.uint8)
    sf = pack_scale_words(exps)
    packed = pack_weight_tiles(weight)
    kernels = _launch(_block32_kernels(n, 1280))
    layer = SimpleNamespace(
        weight=weight, tilelang_weight=packed, tilelang_weight_sf=sf,
        tilelang_decode_rows=max((64, *WIDE_DECODE.get((n, 1280), {}))),
        tilelang_wide=WIDE_DECODE.get((n, 1280), {}),
        **{f"tilelang_{key}": value for key, value in kernels.items()},
    )
    segs = torch.tensor([packed.data_ptr(), packed.numel(), sf.data_ptr(), sf.numel() * 4],
                        dtype=torch.int64, device="cuda")
    outputs, functions = {}, {}
    for variant in VARIANTS[:-1]:
        auxiliary = AUXILIARY_STREAM
        q = torch.empty((m, 1280), device="cuda", dtype=torch.bfloat16)
        cast = empty_mxfp8(q)
        # Explicit poison in unread row padding: live outputs must ignore it.
        cast.values.view(torch.uint8).fill_(127)
        cast.scales.fill_(255)
        kv = torch.empty((m, 512), device="cuda", dtype=torch.bfloat16)
        cache = torch.full((PAGES, PAGE * RECORD_BYTES["swa"]), 91, device="cuda", dtype=torch.uint8)
        projected = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

        def front(variant=variant, q=q, cast=cast, kv=kv, cache=cache, auxiliary=auxiliary):
            if variant == "fused":
                attention_front(raw, qw, kw, EPS, positions, table, cache, slots,
                                PAGE, q, cast.values, cast.scales, kv)
            elif variant in ("fused64", "fused128", "fused256", "split_qkv", "split_kvq", "split_parallel"):
                args = (raw, qw, kw, EPS, positions, table, cache, slots, PAGE, q, cast.values, cast.scales, kv)
                if variant.startswith("fused"):
                    front_variant(*args, threads=int(variant[5:]))
                elif variant == "split_parallel":
                    auxiliary.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(auxiliary):
                        front_variant(*args, partition=2)
                    front_variant(*args, partition=1)
                else:
                    for partition in ((1,2) if variant == "split_qkv" else (2,1)):
                        front_variant(*args, partition=partition)
            elif variant == "split_q":
                pdl.norm_forward_and_per_token_cast(
                    raw[:, :1280], qw, EPS, "e4m3", 32, round_sf=True,
                    use_packed_ue8m0=True, with_out_bf16=True,
                    out=(cast.values[:m], cast.scales[:m].view(torch.int32)), out_bf16=q,
                )
                pdl.norm_forward(raw[:, 1280:], kw, EPS, out=kv)
                rope_(kv, positions, table, 1, False)
                write_cache(kv, cache, slots, page_size=PAGE, cache_kind="swa")
            else:
                pdl.norm_forward(raw[:, :1280], qw, EPS, out=q)
                pdl.norm_forward(raw[:, 1280:], kw, EPS, out=kv)
                pdl.per_token_cast(q, "e4m3", 32, round_sf=True, use_packed_ue8m0=True,
                                   out=(cast.values[:m], cast.scales[:m]))
                rope_(kv, positions, table, 1, False)
                write_cache(kv, cache, slots, page_size=PAGE, cache_kind="swa")

        def chain(front=front, cast=cast, projected=projected, variant=variant, auxiliary=auxiliary):
            front()
            _block32_gemm(layer, cast.values, cast.scales, projected, m)
            if variant == "split_parallel":
                torch.cuda.current_stream().wait_stream(auxiliary)

        if variant == "fused":
            def projection_only(cast=cast, projected=projected):
                _block32_gemm(layer, cast.values, cast.scales, projected, m)

        functions[variant] = chain
        outputs[variant] = (q, cast.values[:m], cast.scales[:m], kv, cache, projected)
    functions["projection_only"] = projection_only
    outputs["projection_only"] = outputs["fused"]
    return functions, outputs, segs, raw, slots


def main():
    global WARMUP_STREAM, CAPTURE_STREAM, AUXILIARY_STREAM
    WARMUP_STREAM = torch.cuda.Stream()
    CAPTURE_STREAM = torch.cuda.Stream()
    AUXILIARY_STREAM = torch.cuda.Stream()
    torch.manual_seed(517)
    emit(identity={"image_vllm_tree": "c061e5f6931306fa4655ba531428f639c365bf10",
                   "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                   "norm_eps": EPS, "rows": ROWS, "variants": VARIANTS, "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "weights": "seeded synthetic checkpoint shapes",
                   "evict_bytes": 128 << 20, "reset_persisting_l2_before_eviction": True, "prefetch_grid": 2,
                   "prefetch_block": 128, "prefetch_chunk": 4096})
    evict = torch.ones((128 << 20) // 4, device="cuda", dtype=torch.int32)
    prefetch = l2_prefetch_kernel(2, 128, 4096)
    side = torch.cuda.Stream()
    assert len({stream.cuda_stream for stream in (WARMUP_STREAM,CAPTURE_STREAM,AUXILIARY_STREAM,side)}) == 4
    emit(distinct_graph_streams=True)
    ordering = random.Random(517)
    saved = {}
    for n, padded in ((8192, False), (12288, False), (12288, True)):
        for m in reversed(ROWS):
            functions, outputs, segs, raw, slots = make_case(n, m, padded)
            for fn in functions.values():
                fn()
            torch.cuda.synchronize()
            for variant in VARIANTS[1:]:
                assert all(byte_equal(a, b) for a, b in zip(outputs["fused"], outputs[variant])), (n, m, padded, variant)
            projected = outputs["fused"][-1]
            if padded:
                assert torch.count_nonzero(projected[:, 8192:]) == 0
            key = (n, padded)
            if m == 96:
                saved[key] = projected.clone()
            else:
                assert byte_equal(projected, saved[key][:m]), ("batch invariance", n, m, padded)
            # Check skipped slots separately: all timed rows have live cache writes,
            # including the single-row case. Also cover a page crossing and last slot.
            saved_slot = slots[-1].clone()
            slots[-1] = -1
            for out in outputs.values(): out[4].fill_(91)
            for fn in functions.values(): fn()
            for variant in VARIANTS:
                assert torch.all(outputs[variant][4][1, -RECORD_BYTES["swa"]:] == 91)
                assert all(byte_equal(a,b) for a,b in zip(outputs["fused"],outputs[variant]))
            slots[-1] = saved_slot
            for fn in functions.values(): fn()
            clean = tuple(x.clone() for x in outputs["fused"])
            # Live Q and KV norms must preserve the native NaN/Inf response and
            # leave the disjoint latent unchanged; no non-finite model input is
            # made acceptable. Downstream NaN payloads are not an accuracy metric.
            for column, norm_index, unaffected in ((0,0,(3,4)), (1280,3,(0,1,2,5))):
                original = raw[0, column].clone()
                for value in (float("nan"), float("inf"), -float("inf")):
                    raw[0, column] = value
                    for fn in functions.values(): fn()
                    for variant in VARIANTS:
                        normalized = outputs[variant][norm_index]
                        assert torch.isnan(normalized[0]).any(), ("nonfinite",variant,column,value)
                        assert byte_equal(normalized[1:],clean[norm_index][1:])
                        assert byte_equal(normalized,outputs["fused"][norm_index])
                        assert all(byte_equal(outputs[variant][i],clean[i]) for i in unaffected)
                raw[0, column] = original
                for fn in functions.values(): fn()
            del clean
            prefetch(segs)
            graphs = {v: capture(fn) for v, fn in functions.items()}
            warm = {v: capture(fn, 20) for v, fn in functions.items()}
            def race(fn):
                # Capture the stream fork as well: an eager prefetch followed by
                # a Python graph launch could finish before the chain starts.
                def call():
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        prefetch(segs)
                    fn()
                    torch.cuda.current_stream().wait_stream(side)
                return capture(call)
            racing = {v: race(fn) for v, fn in functions.items()}
            reference = {v: tuple(x.clone() for x in out) for v, out in outputs.items()}
            samples = {mode: {v: [] for v in VARIANTS} for mode in MODES}
            for mode in MODES:
                for trial in range(31):
                    order = list(VARIANTS)
                    ordering.shuffle(order)
                    for variant in order:
                        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        if mode != "l2":
                            # GB10 evict-last lines survive a normal scrub. Reset
                            # their priority in this benchmark's current context
                            # before reading the 128 MiB eviction buffer.
                            status, = cu.cuCtxResetPersistingL2Cache()
                            assert status == cu.CUresult.CUDA_SUCCESS, status
                            evict.max()
                        else:
                            # Prime this exact chain immediately before its timed
                            # 20-call graph, independent of the preceding arm.
                            warm[variant].replay()
                        start.record()
                        {"l2": warm, "cold": graphs, "prefetch": racing}[mode][variant].replay()
                        end.record()
                        torch.cuda.current_stream().wait_stream(side)
                        torch.cuda.synchronize()
                        samples[mode][variant].append(start.elapsed_time(end) * 1000 / (20 if mode == "l2" else 1))
                        assert all(byte_equal(a, b) for a, b in zip(outputs[variant], reference[variant]))
            emit(n=n, rows=m, tp3_last_rank=padded, exact=True, batch_invariant=True,
                 finite_contract=True, samples_us=samples,
                 medians_us={mode: {v: statistics.median(xs) for v, xs in arms.items()}
                             for mode, arms in samples.items()},
                 memory_allocated=torch.cuda.memory_allocated(),
                 memory_reserved=torch.cuda.memory_reserved())
            del functions, outputs, graphs, warm, racing, reference
    emit(passed=True, max_memory_allocated=torch.cuda.max_memory_allocated(),
         max_memory_reserved=torch.cuda.max_memory_reserved())


if __name__ == "__main__":
    main()
