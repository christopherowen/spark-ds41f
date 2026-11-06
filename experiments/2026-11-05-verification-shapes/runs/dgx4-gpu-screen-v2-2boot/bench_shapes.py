"""2-boot CUDA graph capture/replay verification & latency benchmark for PR #921 (including Sparse-MLA)."""
import json
import statistics
import time
import torch
from vllm.models.deepseek_v4_1.tilelang import dspark_context
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    mxfp8_gemm,
    mxfp8_gemm_decode,
    mxfp8_gemm_partials,
    split_row_tile,
    splitk_reduce,
)
from vllm.models.deepseek_v4_1.tilelang.scale_pack import pack_scale_words
from vllm.models.deepseek_v4_1.tilelang.sparse_mla import HEAD_DIM, SparseMlaV41Plan
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from tests.kernels.attention.test_deepseek_v41_tilelang import (
    _mla_case,
    _mla_ref,
    _paged,
    mxfp8_values,
)
from tests.kernels.quantization.test_deepseek_v41_tilelang_linear import (
    _block32_weight,
    _gen,
    _mxfp8,
)

PAIRS = [
    {"num_reqs": 3, "native_rows": 18, "wider_padded_rows": 20},
    {"num_reqs": 5, "native_rows": 30, "wider_padded_rows": 32},
    {"num_reqs": 6, "native_rows": 36, "wider_padded_rows": 40},
    {"num_reqs": 7, "native_rows": 42, "wider_padded_rows": 48},
]

def time_graph(graph, warmup=30, rounds=30, inner=20):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples_us = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        start_ev.record()
        for _ in range(inner):
            graph.replay()
        end_ev.record()
        torch.cuda.synchronize()
        samples_us.append((start_ev.elapsed_time(end_ev) * 1000.0) / inner)
    samples_us.sort()
    return {
        "median_us": round(statistics.median(samples_us), 4),
        "mean_us": round(statistics.mean(samples_us), 4),
        "min_us": round(samples_us[0], 4),
        "p95_us": round(samples_us[int(0.95 * (len(samples_us) - 1))], 4),
        "samples": len(samples_us),
    }

def bench_context_inputs(rows, capacity):
    width, allocated = 5120, 132
    gen = torch.Generator(device="cuda").manual_seed(1000 + rows)
    sources = []
    for index in range(3):
        storage = torch.empty(allocated, width + 8 * (index + 1), dtype=torch.bfloat16, device="cuda")
        source = storage[:, :width]
        source[:rows].copy_(torch.randn(rows, width, generator=gen, device="cuda"))
        sources.append(source)
    aux = torch.full((allocated, 3 * width), 7, device="cuda", dtype=torch.bfloat16)
    positions = torch.full((allocated,), 99, device="cuda", dtype=torch.int64)
    slots = torch.full((2, allocated), 77, device="cuda", dtype=torch.int64)

    def run():
        dspark_context.context_inputs(aux, sources, positions, slots, rows, capacity)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.full_like(aux, 7)
    for index, source in enumerate(sources):
        expected[:rows, index * width : (index + 1) * width].copy_(source[:rows])
    expected[rows:capacity].zero_()
    pos = torch.full_like(positions, 99)
    pos[rows:capacity].zero_()
    slot = torch.full_like(slots, 77)
    slot[:, rows:capacity].fill_(PAD_SLOT_ID)
    exact = bool(
        torch.equal(aux.view(torch.int16), expected.view(torch.int16))
        and torch.equal(positions, pos)
        and torch.equal(slots, slot)
    )
    if not exact:
        raise RuntimeError(f"context_inputs mismatch at rows={rows}, capacity={capacity}")
    stats = time_graph(graph)
    stats["exact_match"] = True
    stats["rows"] = rows
    stats["capacity"] = capacity
    return stats

def bench_mxfp8(rows, split=False):
    n, k, shards = 64, 256, 2 if split else 1
    tile = split_row_tile(rows)
    padded = -(-rows // tile) * tile
    weight, exps, _ = _block32_weight(n, k, 41)
    weight_sf = pack_scale_words(exps)
    x = torch.randn((rows, k), generator=_gen(42), device="cuda").bfloat16()
    xq, sf, _ = _mxfp8(x)
    a = torch.empty((padded, k), dtype=xq.dtype, device="cuda")
    scales = torch.empty((padded, sf.shape[1]), dtype=sf.dtype, device="cuda")
    a[:rows].copy_(xq)
    scales[:rows].copy_(sf)
    out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    expected = torch.empty_like(out)
    reference = mxfp8_gemm(n, k, block_M=64, block_N=64, block_K=64, num_stages=2, shards=shards)
    reference(xq, weight, sf.view(torch.uint32), weight_sf, expected)
    if split:
        partials = torch.empty((shards, rows, n), device="cuda")
        kernel = mxfp8_gemm_partials(n, k, shards, block_M=tile, block_N=64, block_K=64, num_stages=2)
        reduce = splitk_reduce(n, shards)
        def run():
            kernel(a, weight, scales.view(torch.uint32), weight_sf, partials)
            reduce(partials, out)
    else:
        kernel = mxfp8_gemm_decode(n, k, block_M=tile, block_N=64, block_K=64, num_stages=2, padded_rows=True)
        def run():
            kernel(a, weight, scales.view(torch.uint32), weight_sf, out)

    run()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    stats = time_graph(graph)
    stats["exact_match"] = True
    stats["rows"] = rows
    stats["tile"] = tile
    stats["padded_rows"] = padded
    stats["split"] = split
    return stats

def setup_sparse_mla_suite(heads=24, max_rows=64):
    case_idx = _mla_case(idx_width=512, heads=heads, rows=max_rows)
    plan_idx = SparseMlaV41Plan(
        heads, 128, 512, 128, swa_page_size=128,
        max_rows=max_rows, decode_max_rows=max_rows, split_max_rows=max_rows,
    )
    swa_pool_idx = _paged(case_idx["swa"], case_idx["swa_page"])
    idx_pool_idx = _paged(case_idx["idx"], case_idx["idx_page"])
    ref_idx = _mla_ref(case_idx)

    case_swa = _mla_case(idx_width=0, heads=heads, rows=max_rows)
    plan_swa = SparseMlaV41Plan(
        heads, 128, 0, 128, swa_page_size=128,
        max_rows=max_rows, decode_max_rows=max_rows, split_max_rows=max_rows,
    )
    swa_pool_swa = _paged(case_swa["swa"], case_swa["swa_page"])
    ref_swa = _mla_ref(case_swa)
    return {
        "heads": heads,
        "idx": (plan_idx, case_idx, swa_pool_idx, idx_pool_idx, ref_idx),
        "swa": (plan_swa, case_swa, swa_pool_swa, None, ref_swa),
    }

def bench_sparse_mla_route(suite, kind, schedule, rows, layers=1):
    plan, case, swa_pool, idx_pool, ref = suite[kind]
    heads = suite["heads"]
    q = case["q"][:rows].contiguous()
    swa_idx = case["swa_idx"][:rows].contiguous()
    swa_len = case["swa_len"][:rows].contiguous()
    positions = case["positions"][:rows].contiguous()
    out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    indexed = {}
    if idx_pool is not None:
        indexed = dict(
            idx_cache=idx_pool,
            idx_indices=case["idx_logical"][:rows].contiguous(),
            idx_lengths=case["idx_len"][:rows].contiguous(),
            idx_page_table=case["table"][:rows].contiguous(),
        )
    scratch = None
    if schedule != "extend":
        scratch = [
            torch.empty(shape, dtype=dtype, device="cuda")
            for shape, dtype in plan.scratch_specs(rows, schedule)
        ]

    def run_once():
        plan.run(
            q, swa_pool, swa_idx, swa_len, case["sinks"], positions, case["rope"], out,
            kernel=schedule, scratch=scratch, **indexed
        )

    def run_stack():
        for _ in range(layers):
            plan.run(
                q, swa_pool, swa_idx, swa_len, case["sinks"], positions, case["rope"], out,
                kernel=schedule, scratch=scratch, **indexed
            )

    run_once()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref[:rows].float(), rtol=0.13, atol=2e-2)
    flat = out.view(rows, -1)
    assert torch.equal(mxfp8_values(flat).bfloat16(), flat)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_stack()
    graph.replay()
    torch.cuda.synchronize()
    stats = time_graph(graph, warmup=20, rounds=30, inner=10 if layers == 1 else 2)
    stats["exact_match"] = True
    stats["rows"] = rows
    stats["layers"] = layers
    stats["schedule"] = schedule
    stats["kind"] = kind
    return stats, out.clone()

def main():
    import sys
    boot = int(sys.argv[1])
    out_path = sys.argv[2]
    mla_suite = setup_sparse_mla_suite(heads=24, max_rows=64)
    results = {
        "boot": boot,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(0),
        "pairs": [],
    }
    for pair in PAIRS:
        nr = pair["native_rows"]
        wr = pair["wider_padded_rows"]
        mla_ext_nat, out_ext_nat = bench_sparse_mla_route(mla_suite, "idx", "extend", nr, layers=1)
        mla_ext_wid, out_ext_wid = bench_sparse_mla_route(mla_suite, "idx", "extend", wr, layers=1)
        mla_dec_nat, out_dec_nat = bench_sparse_mla_route(mla_suite, "idx", "decode", nr, layers=1)
        mla_dec_wid, out_dec_wid = bench_sparse_mla_route(mla_suite, "idx", "decode", wr, layers=1)
        mla_swa_nat, out_swa_nat = bench_sparse_mla_route(mla_suite, "swa", "extend", nr, layers=1)
        mla_swa_wid, out_swa_wid = bench_sparse_mla_route(mla_suite, "swa", "extend", wr, layers=1)
        mla_78l_nat, _ = bench_sparse_mla_route(mla_suite, "idx", "extend", nr, layers=78)
        mla_78l_wid, _ = bench_sparse_mla_route(mla_suite, "idx", "extend", wr, layers=78)

        # Bit-identical live-row invariance across native vs wider padded & extend vs decode
        assert torch.equal(out_ext_nat, out_dec_nat), "extend vs decode mismatch"
        assert torch.equal(out_ext_nat, out_ext_wid[:nr]), "native vs wider extend live-row mismatch"
        assert torch.equal(out_dec_nat, out_dec_wid[:nr]), "native vs wider decode live-row mismatch"
        assert torch.equal(out_swa_nat, out_swa_wid[:nr]), "native vs wider swa live-row mismatch"

        entry = {
            "num_reqs": pair["num_reqs"],
            "native_rows": nr,
            "wider_padded_rows": wr,
            "sparse_mla_idx_extend_native": mla_ext_nat,
            "sparse_mla_idx_extend_wider_padded": mla_ext_wid,
            "sparse_mla_idx_extend_speedup_pct": round((1.0 - mla_ext_nat["median_us"] / mla_ext_wid["median_us"]) * 100.0, 2),
            "sparse_mla_idx_decode_native": mla_dec_nat,
            "sparse_mla_idx_decode_wider_padded": mla_dec_wid,
            "sparse_mla_idx_decode_speedup_pct": round((1.0 - mla_dec_nat["median_us"] / mla_dec_wid["median_us"]) * 100.0, 2),
            "sparse_mla_swa_extend_native": mla_swa_nat,
            "sparse_mla_swa_extend_wider_padded": mla_swa_wid,
            "sparse_mla_swa_extend_speedup_pct": round((1.0 - mla_swa_nat["median_us"] / mla_swa_wid["median_us"]) * 100.0, 2),
            "sparse_mla_78layer_graph_native": mla_78l_nat,
            "sparse_mla_78layer_graph_wider_padded": mla_78l_wid,
            "sparse_mla_78layer_saved_us": round(mla_78l_wid["median_us"] - mla_78l_nat["median_us"], 2),
            "sparse_mla_78layer_speedup_pct": round((1.0 - mla_78l_nat["median_us"] / mla_78l_wid["median_us"]) * 100.0, 2),
            "context_inputs_native": bench_context_inputs(nr, nr),
            "context_inputs_wider_padded": bench_context_inputs(nr, wr),
            "mxfp8_decode_native": bench_mxfp8(nr, split=False),
            "mxfp8_decode_wider_padded": bench_mxfp8(wr, split=False),
            "mxfp8_splitk_native": bench_mxfp8(nr, split=True),
            "mxfp8_splitk_wider_padded": bench_mxfp8(wr, split=True),
        }
        results["pairs"].append(entry)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
