"""2-boot CUDA graph capture/replay verification & latency benchmark for PR #921."""
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
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
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

def time_graph(graph, warmup=30, rounds=30, inner=50):
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

def main():
    import sys
    boot = int(sys.argv[1])
    out_path = sys.argv[2]
    results = {
        "boot": boot,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(0),
        "pairs": [],
    }
    for pair in PAIRS:
        nr = pair["native_rows"]
        wr = pair["wider_padded_rows"]
        entry = {
            "num_reqs": pair["num_reqs"],
            "native_rows": nr,
            "wider_padded_rows": wr,
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
