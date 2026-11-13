#!/usr/bin/env python3
"""Image A/B for #998 (candidate-53 own marks) at TP3 (tp3-triangle shard shapes: vocab shard
43,136, DSpark MoE inter 768), the shape it acts on. Same allocation workload as the PR's
bench_carveout_expansion.py measure_allocation_pass, laid out the way each image's code lays it
out: the base image packs embed_tokens/lm_head in place in their carve-out buffers and keeps
markov_w1 and the DSpark MoE slabs in regular memory (its display_carveout has no compact/move;
total_bytes/close are shimmed here for the accounting only); the candidate image runs
display_carveout.compact/allocate/move as the PR's candidate pass does. Then the PR's kernels
on the image's own placement (markov_anchor_embed reqs 1/3/8, dspark_moe M 5/15/40,
vocab_embedding_packed M 1/5/40/4096; 200 timed, median, CUDA events, after at least 30 calls and
100 ms of warm-up), and a SHA-256 of each kernel output for a cross-image torch.equal.
Seeded (torch.manual_seed(998)) so both images build identical tensors."""
import gc, hashlib, json, sys
import torch
from vllm.v1.worker import display_carveout as dc
from vllm.models.deepseek_v4_1.tilelang.vocab import pack_bf16_weight
from vllm.models.deepseek_v4_1.tilelang.embedding import declare_packed, vocab_embedding_packed
from vllm.models.deepseek_v4_1.tilelang.moe import MoeExpertsV41Plan, workspace_bytes
from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8
from vllm.v1.worker.gpu.sample import tilelang_sampling as tl_dspark

label, out_path = sys.argv[1], sys.argv[2]
CAND = hasattr(dc, "compact") and hasattr(dc, "move")
total_bytes = getattr(dc, "total_bytes", lambda: sum(c.nbytes for c in dc._carveouts))
def mem_avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"): return int(l.split()[1]) * 1024
import time
def bench_cuda(fn, warmup=30, iters=200):
    # PR's timer, with the warm-up extended to at least 100 ms of back-to-back calls
    t0 = time.perf_counter(); n = 0
    while n < warmup or time.perf_counter() - t0 < 0.1:
        fn(); n += 1
        if n % 10 == 0: torch.cuda.synchronize()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]; e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    t = sorted(a.elapsed_time(b) * 1000.0 for a, b in zip(s, e))
    return round(t[len(t) // 2], 2)
digest = lambda *ts: hashlib.sha256(b"".join(t.contiguous().view(torch.uint8).cpu().numpy().tobytes() for t in ts)).hexdigest()[:16]
dev = torch.device("cuda:0")
torch.manual_seed(998)
hidden, vocab_size, markov_rank, E, topk, max_rows = 5120, 129280, 256, 128, 3, 40
vocab_shard, inter = 43136, 768
plan = MoeExpertsV41Plan(E, topk, hidden, inter, swiglu_limit=7.0, out_dtype=torch.bfloat16, idx_dtypes=(torch.int32,), max_rows=max_rows)
plan.declare_kernels()
tl_dspark.markov_anchor_embed(torch.zeros((5,), dtype=torch.int64, device=dev), torch.zeros((1,), dtype=torch.int64, device=dev),
                              torch.zeros((16, markov_rank), dtype=torch.bfloat16, device=dev),
                              torch.zeros((1, markov_rank), dtype=torch.bfloat16, device=dev),
                              torch.zeros((1, markov_rank), dtype=torch.bfloat16, device=dev))
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
dc._REQUIRED = True
torch.cuda.reset_peak_memory_stats()
avail0, free0 = mem_avail(), torch.cuda.mem_get_info()[0]
alloc0, res0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
embed_raw = dc.allocate((vocab_shard, hidden), torch.bfloat16, dev)
lm_raw = dc.allocate((vocab_shard, hidden), torch.bfloat16, dev)
embed_raw.copy_(torch.randn(hidden, dtype=torch.bfloat16, device=dev).expand(vocab_shard, hidden))
lm_raw.copy_(torch.randn(hidden, dtype=torch.bfloat16, device=dev).expand(vocab_shard, hidden))
embed_packed = pack_bf16_weight(embed_raw, in_place=True); del embed_raw
embed_vals = dc.compact(embed_packed.values) if CAND else embed_packed.values
lm_packed = pack_bf16_weight(lm_raw, in_place=True); del lm_raw
lm_vals = dc.compact(lm_packed.values) if CAND else lm_packed.values
markov_w1 = dc.allocate((vocab_size, markov_rank), torch.bfloat16, dev) if CAND else None
if markov_w1 is None:
    markov_w1 = torch.empty((vocab_size, markov_rank), dtype=torch.bfloat16, device=dev)
markov_w1.copy_(torch.randn(markov_rank, dtype=torch.bfloat16, device=dev).expand(vocab_size, markov_rank))
layers = []
for _ in range(3):
    w13 = torch.randint(0, 255, (E, 2 * inter, hidden // 2), dtype=torch.uint8, device=dev)
    s13 = torch.full((E, 2 * inter, hidden // 32), 127, dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 255, (E, hidden, inter // 2), dtype=torch.uint8, device=dev)
    s2 = torch.full((E, hidden, inter // 32), 127, dtype=torch.uint8, device=dev)
    p13 = list(plan.prepare_weight(w13, s13)); del w13, s13
    if CAND: p13[0] = dc.move(p13[0])
    p2 = list(plan.prepare_weight(w2, s2)); del w2, s2
    if CAND: p2[0] = dc.move(p2[0])
    layers.append((tuple(p13), tuple(p2)))
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
held = [t for t in (embed_vals, lm_vals, markov_w1) + tuple(x for l in layers for x in (l[0][0], l[1][0]))
        if any(dc._holds(c, t) for c in dc._carveouts)]
mib = float(1 << 20)
mem = {"carveout_buffers_mib": [c.nbytes >> 20 for c in dc._carveouts], "carveout_total_mib": total_bytes() >> 20,
       "carveout_weight_mib": round(sum(t.numel() * t.element_size() for t in held) / mib, 3),
       "uma_mem_available_used_mib": round((avail0 - mem_avail()) / mib, 2),
       "uma_cuda_free_used_mib": round((free0 - torch.cuda.mem_get_info()[0]) / mib, 2),
       "cuda_allocated_mib": round((torch.cuda.memory_allocated() - alloc0) / mib, 3),
       "cuda_reserved_mib": round((torch.cuda.memory_reserved() - res0) / mib, 3),
       "cuda_peak_reserved_mib": round((torch.cuda.max_memory_reserved() - res0) / mib, 3),
       "placement": {"embed_tokens": "carveout" if any(dc._holds(c, embed_vals) for c in dc._carveouts) else "regular",
                     "markov_w1": "carveout" if any(dc._holds(c, markov_w1) for c in dc._carveouts) else "regular",
                     "dspark_moe_slabs_in_carveout": sum(1 for l in layers for x in (l[0][0], l[1][0]) if any(dc._holds(c, x) for c in dc._carveouts))}}
kern = {}
for reqs in (1, 3, 8):
    ids = torch.randint(0, vocab_size, (reqs * 5,), dtype=torch.int64, device=dev)
    idx = torch.arange(reqs, dtype=torch.int64, device=dev) * 5
    step = torch.empty((reqs, markov_rank), dtype=torch.bfloat16, device=dev)
    conf = torch.zeros((reqs, markov_rank), dtype=torch.bfloat16, device=dev)
    t = bench_cuda(lambda: tl_dspark.markov_anchor_embed(ids, idx, markov_w1, step, conf))
    kern[f"markov_anchor_embed_reqs_{reqs}"] = {"us": t, "digest": digest(step, conf)}
ws = torch.empty((workspace_bytes(plan.workspace_specs(max_rows, torch.int32)),), dtype=torch.uint8, device=dev)
w0 = layers[0]
for M in (5, 15, 40):
    x = cast_mxfp8(torch.randn((M, hidden), dtype=torch.bfloat16, device=dev))
    ti = (torch.arange(M * topk, dtype=torch.int32, device=dev).reshape(M, topk) * 7) % E
    tw = torch.full((M, topk), 1.0 / topk, dtype=torch.float32, device=dev)
    o = torch.empty((M, hidden), dtype=torch.bfloat16, device=dev)
    t = bench_cuda(lambda: plan.run(x, ti, tw, w0, o, ws, None))
    kern[f"dspark_moe_M_{M}"] = {"us": t, "digest": digest(o)}
declare_packed(hidden, embed_packed.exponent_base)
for M in (1, 5, 40, 4096):
    ids = torch.randint(0, vocab_size, (M,), dtype=torch.int32, device=dev)
    o = torch.empty((M, hidden), dtype=torch.bfloat16, device=dev)
    t = bench_cuda(lambda: vocab_embedding_packed(ids, embed_vals, embed_packed.exception_entries, embed_packed.row_offsets,
                                                  hidden, embed_packed.exponent_base, o, 0, vocab_shard, 0, vocab_shard, vocab_shard))
    kern[f"vocab_embedding_packed_M_{M}"] = {"us": t, "digest": digest(o)}
res = {"label": label, "candidate_api": CAND, "gpu": torch.cuda.get_device_name(0), "memory": mem, "kernels": kern}
open(out_path, "w").write(json.dumps(res, indent=1) + "\n")
print(json.dumps(res))
