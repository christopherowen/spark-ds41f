"""Vocabulary heads per call under CUDA graphs, through patch 0036's prepared TileLang heads.

The packed BF16 head (DS4.1's TP4 and TP3 vocabulary shards, K = 5,120) against
B12X's Triton projection with its default configuration; the 256-wide BF16
Markov head against B12X's Triton row kernel (one row) and cuBLAS. The Markov
head cycles eight weight copies (about 130 MB), so no call finds its weights in
L2; the packed head's 248-330 MB never fits.
"""
from types import SimpleNamespace

import torch
from b12x.gemm.bf16_vocab_projection import _kernel as row_kernel
from b12x.gemm.packed_bf16_vocab_projection import _kernel as triton_kernel
from b12x.gemm.packed_bf16_vocab_projection import pack
from b12x.gemm.packed_bf16_vocab_projection._tuning import (
    TUNING,
    PackedBf16VocabProjectionQuery,
    block_m,
)

from vllm.models.deepseek_v4_1.tilelang.vocab import prepare_bf16_head, prepare_packed_head, project_head


def timed(calls, rounds=20):
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / (rounds * len(calls))


count = 0
K = 5120
for n in (32320, 43092):
    weight = (torch.randn(n, K, device="cuda") * 0.02).bfloat16()
    packed = pack(weight.clone())
    del weight
    layer = SimpleNamespace(
        weight=packed.values, weight_exception_rows=packed.exception_rows,
        weight_exception_entries=packed.exception_entries, weight_row_offsets=packed.row_offsets,
        packed_bf16_in_features=K, packed_bf16_exponent_base=packed.exponent_base)
    prepare_packed_head(layer)
    for rows in (1, 6, 16, 32, 48, 64):
        x = (torch.randn(rows, K, device="cuda") * 0.5).bfloat16()
        config = TUNING.default_config(
            PackedBf16VocabProjectionQuery(max_tokens=rows, in_features=K, out_features=n), None)
        tile = block_m(rows)
        out_t = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        grid = ((n + config.block_n - 1) // config.block_n, (rows + tile - 1) // tile, 1)

        def run_triton():
            triton_kernel._projection_kernel[grid](
                x, packed.values, out_t, packed.exception_rows, packed.exception_entries, packed.row_offsets,
                rows, packed.exponent_base - 1, K=K, N=n, BLOCK_N=config.block_n, BLOCK_M=tile,
                STEP_K=config.step_k, NUM_STAGES=config.num_stages, EVICT_FIRST=config.evict_first,
                num_warps=config.num_warps)

        triton_us = timed([run_triton] * 10)
        tilelang_us = timed([lambda: project_head(layer, x)] * 10)
        run_triton()
        gap = (project_head(layer, x).float() - out_t.float()).abs().max().item()
        print(f"packed head N {n} rows {rows:2d}: triton {triton_us:7.1f} us  tilelang {tilelang_us:7.1f} us"
              f"  {100 * (tilelang_us / triton_us - 1):+5.1f}%  {packed.values.numel() / (tilelang_us * 1e3):4.0f} GB/s"
              f"  max |tilelang - triton| {gap:.3g}")
        count += 1
    del packed, layer

COPIES = 8
for n in (32320, 43092):
    layers = [SimpleNamespace(weight=(torch.randn(n, 256, device="cuda") * 0.05).bfloat16()) for _ in range(COPIES)]
    for layer in layers:
        prepare_bf16_head(layer)
    for rows in (1, 8, 16, 48):
        x = torch.randn(rows, 256, device="cuda").bfloat16()
        line = f"BF16 head N {n} K 256 rows {rows:2d}:"
        if rows == 1:
            out_t = torch.empty(1, n, dtype=torch.bfloat16, device="cuda")
            line += " triton row {:6.1f} us".format(timed([
                lambda w=layer.weight: row_kernel._row_kernel[(n, 1, 1)](x, w, out_t, K=256, BLOCK_K=256, N=n,
                                                                       num_warps=8)
                for layer in layers]))
        out_c = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        line += "  cublas {:6.1f} us".format(timed([lambda w=layer.weight: torch.matmul(x, w.T, out=out_c)
                                                    for layer in layers]))
        tilelang_us = timed([lambda layer=layer: project_head(layer, x) for layer in layers])
        reference = x.float() @ layers[-1].weight.float().T
        error = (project_head(layers[-1], x).float() - reference).abs().max().item()
        print(line + f"  tilelang {tilelang_us:6.1f} us  {n * 512 / (tilelang_us * 1e3):4.0f} GB/s"
              f"  max |tilelang - fp32| {error:.3g}")
        count += 1
print(f"{count} passed")
