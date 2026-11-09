"""Same-session A/B of #1009's own kernel sequences, by CUDA-graph replay, CUPTI kernel
time and eager CUDA events.

The two sequences and their inputs are copied from #1009's own bench
(experiments/2026-11-08-sp-prefill-local-attention-front/bench_sp_attention_front.py,
``benchmark_shapes``): rank 0 of TP3 and TP4 at T = 4096, 8192, 16384.

- ``front_1107``: the bench's ``run_baseline``, the front candidate-50 without #1009
  runs (q/kv norms into the gather buffer, head weights copied beside them, the
  gathered buffer split, the q cast, RoPE and the SWA write on all T rows). Every
  tree has these kernels.
- ``front_1009``: the bench's ``run_option_b`` (``sp_attention_front`` on the local
  real rows, ``write_swa_records`` on all T rows). Only a tree with #1009 has it.

As in #1009's bench, neither includes the all-gather (its payload is 3,648 B/row
before and 1,848 B/row after) or ``weights_proj``.

usage: python3 bench_abba.py <out.json>
"""

import json
import sys
import time

import torch

from vllm.models.deepseek_v4_1 import kernels, sp_prefill
from vllm.models.deepseek_v4_1.tilelang import columns
from vllm.models.deepseek_v4_1.tilelang.cache_writer import RECORD_BYTES, write_cache
from vllm.models.deepseek_v4_1.tilelang.front import KV_LATENT, Q_LATENT, ROPE_DIM
from vllm.models.deepseek_v4_1.tilelang.linear import empty_mxfp8
from vllm.models.deepseek_v4_1.tilelang.pdl import norm_forward
from vllm.models.deepseek_v4_1.tilelang.rope import rope_

try:
    from vllm.models.deepseek_v4_1.tilelang.cache_writer import write_swa_records
    from vllm.models.deepseek_v4_1.tilelang.front import sp_attention_front

    HAS_1009 = True
except ImportError:
    HAS_1009 = False

DEVICE = "cuda"
EPS = 1e-6
HEADS = 32
PAGE_SIZE = 64


def _make_cos_sin(max_pos: int = 32768) -> torch.Tensor:
    inv = 1.0 / (10000 ** (torch.arange(0, ROPE_DIM, 2, device=DEVICE).double() / ROPE_DIM))
    angles = torch.arange(max_pos, device=DEVICE).double()[:, None] * inv
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float()


def sequences(world_size, num_tokens, cos_sin):
    torch.manual_seed(world_size * 100000 + num_tokens)
    q_weight = torch.ones(Q_LATENT, dtype=torch.bfloat16, device=DEVICE)
    kv_weight = torch.ones(KV_LATENT, dtype=torch.bfloat16, device=DEVICE)
    widths = (Q_LATENT, KV_LATENT, HEADS)
    sp = sp_prefill.SPRows(num_tokens, world_size, 0)
    local, real = sp.local_rows, sp.real_rows
    qr_kv_local = torch.randn((local, Q_LATENT + KV_LATENT), dtype=torch.bfloat16, device=DEVICE)
    qr_local, kv_local = qr_kv_local.split((Q_LATENT, KV_LATENT), dim=-1)
    w_local = torch.randn((local, HEADS), dtype=torch.bfloat16, device=DEVICE)
    front_local = torch.empty((local, sum(widths)), dtype=torch.bfloat16, device=DEVICE)
    gathered_front = torch.randn((num_tokens, sum(widths)), dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    local_pos = positions[:real]
    num_pages = (num_tokens // PAGE_SIZE) + 4
    slots = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    pages = torch.zeros((num_pages, PAGE_SIZE * RECORD_BYTES["swa"]), dtype=torch.uint8, device=DEVICE)

    def front_1107():
        norm_forward(qr_local, q_weight, EPS, out=front_local[:, :Q_LATENT])
        norm_forward(kv_local, kv_weight, EPS, out=front_local[:, Q_LATENT : Q_LATENT + KV_LATENT])
        columns.copy_columns(w_local, front_local[:, Q_LATENT + KV_LATENT :])
        qr_t, kv_t, _ = columns.split_columns(gathered_front, widths)
        _ = kernels.cast_mxfp8(qr_t)
        rope_(kv_t, positions, cos_sin, 1, False)
        write_cache(kv_t, pages, slots, page_size=PAGE_SIZE, cache_kind="swa")

    seqs = {"front_1107": front_1107}
    if HAS_1009:
        local_qr_b = torch.empty((local, Q_LATENT), dtype=torch.bfloat16, device=DEVICE)
        local_q_cast_b = empty_mxfp8(local_qr_b)
        local_records_b = torch.empty((local, RECORD_BYTES["swa"]), dtype=torch.uint8, device=DEVICE)
        gathered_records_b = torch.zeros((num_tokens, RECORD_BYTES["swa"]), dtype=torch.uint8, device=DEVICE)

        def front_1009():
            sp_attention_front(
                qr_kv_local[:real], q_weight, kv_weight, EPS, local_pos, cos_sin,
                local_q_cast_b.values[:real], local_q_cast_b.scales[:real], local_records_b[:real],
            )  # fmt: skip
            write_swa_records(gathered_records_b, pages, slots, page_size=PAGE_SIZE)

        seqs["front_1009"] = front_1009
    return dict(local_rows=local, real_rows=real, padded_rows=sp.padded_rows), seqs


def time_eager(fn, warmup=40, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return dict(median_us=round(t[len(t) // 2], 3), q25_us=round(t[len(t) // 4], 3),
                q75_us=round(t[(3 * len(t)) // 4], 3), min_us=round(t[0], 3))


def time_graph(fn, reps=10, warmup=10, iters=60):
    """One CUDA graph of ``reps`` back-to-back sequences, replayed; us per sequence."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000.0 / reps for s, e in zip(starts, ends))
    del graph
    return dict(median_us=round(t[len(t) // 2], 3), q25_us=round(t[len(t) // 4], 3),
                q75_us=round(t[(3 * len(t)) // 4], 3), min_us=round(t[0], 3))


def time_kernels(fn, blocks=5, iters=40):
    """CUPTI: the sum of the sequence's kernel durations, per sequence, in ``blocks``
    profiled blocks of ``iters`` sequences; the median block mean and the kernels."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    means = []
    names = None
    for _ in range(blocks):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        evts = [e for e in prof.events() if e.device_type.name == "CUDA" and e.device_time > 0]
        kern = [e for e in evts if "memcpy" not in e.name.lower() and "memset" not in e.name.lower()]
        means.append(sum(e.device_time for e in kern) / iters)
        if names is None:
            per = {}
            for e in kern:
                per[e.name] = per.get(e.name, 0) + 1
            names = {k: v // iters for k, v in per.items()}
    means.sort()
    return dict(median_us=round(means[len(means) // 2], 3), min_us=round(means[0], 3),
                max_us=round(means[-1], 3), kernels_per_sequence=names)


def main():
    out_path = sys.argv[1]
    cos_sin = _make_cos_sin()
    cells = []
    for world_size in (3, 4):
        for num_tokens in (4096, 8192, 16384):
            shape, seqs = sequences(world_size, num_tokens, cos_sin)
            cell = dict(tp_size=world_size, num_tokens=num_tokens, **shape)
            for name, fn in seqs.items():
                cell[name] = dict(
                    graph=time_graph(fn), kernels=time_kernels(fn), eager=time_eager(fn)
                )
            cells.append(cell)
            print(json.dumps({k: v for k, v in cell.items()}), flush=True)
    payload = dict(device=torch.cuda.get_device_name(0), has_1009=HAS_1009,
                   timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), cells=cells)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
