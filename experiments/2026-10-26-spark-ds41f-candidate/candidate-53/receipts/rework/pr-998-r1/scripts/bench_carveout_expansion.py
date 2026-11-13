import gc
import json
import sys
import torch

def mem_avail_bytes() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0

def bench_cuda(fn, warmup=30, iters=200):
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
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return round(times[len(times) // 2], 2)

def measure_allocation_pass(
    device: torch.device,
    plan,
    vocab_size: int,
    vocab_shard: int,
    hidden: int,
    markov_rank: int,
    num_experts: int,
    moe_inter: int,
    use_candidate: bool,
):
    from vllm.v1.worker import display_carveout
    from vllm.models.deepseek_v4_1.tilelang.vocab import pack_bf16_weight

    for c in list(display_carveout._carveouts):
        c.close()
    display_carveout._carveouts.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    display_carveout._REQUIRED = True
    torch.cuda.reset_peak_memory_stats()
    avail_0 = mem_avail_bytes()
    free_0 = torch.cuda.mem_get_info()[0]
    alloc_0 = torch.cuda.memory_allocated()
    res_0 = torch.cuda.memory_reserved()

    # 1. Target create_model + load_weights: embed_tokens and lm_head allocated in carve-out
    embed_raw = display_carveout.allocate((vocab_shard, hidden), torch.bfloat16, device)
    lm_raw = display_carveout.allocate((vocab_shard, hidden), torch.bfloat16, device)
    embed_raw.copy_(torch.randn(hidden, dtype=torch.bfloat16, device=device).expand(vocab_shard, hidden))
    lm_raw.copy_(torch.randn(hidden, dtype=torch.bfloat16, device=device).expand(vocab_shard, hidden))

    # 2. Target process_weights_after_loading: pack in-place and compact
    embed_packed = pack_bf16_weight(embed_raw, in_place=True)
    del embed_raw
    embed_vals = display_carveout.compact(embed_packed.values) if use_candidate else embed_packed.values
    lm_packed = pack_bf16_weight(lm_raw, in_place=True)
    del lm_raw
    lm_vals = display_carveout.compact(lm_packed.values) if use_candidate else lm_packed.values

    # 3. DSpark markov_w1 (63.125 MiB)
    if use_candidate:
        markov_w1 = display_carveout.allocate((vocab_size, markov_rank), torch.bfloat16, device)
    else:
        markov_w1 = torch.empty((vocab_size, markov_rank), dtype=torch.bfloat16, device=device)
    markov_w1.copy_(torch.randn(markov_rank, dtype=torch.bfloat16, device=device).expand(vocab_size, markov_rank))

    # 4. 3 DSpark MoE layers (mtp.0, mtp.1, mtp.2)
    moe_layers = []
    for _ in range(3):
        w13 = torch.randint(0, 255, (num_experts, 2 * moe_inter, hidden // 2), dtype=torch.uint8, device=device)
        s13 = torch.full((num_experts, 2 * moe_inter, hidden // 32), 127, dtype=torch.uint8, device=device)
        w2 = torch.randint(0, 255, (num_experts, hidden, moe_inter // 2), dtype=torch.uint8, device=device)
        s2 = torch.full((num_experts, hidden, moe_inter // 32), 127, dtype=torch.uint8, device=device)
        t13, tail13, codes13, byte13, narrow13, base13 = plan.prepare_weight(w13, s13)
        del w13, s13
        if use_candidate:
            t13 = display_carveout.move(t13)
        t2, tail2, codes2, byte2, narrow2, base2 = plan.prepare_weight(w2, s2)
        del w2, s2
        if use_candidate:
            t2 = display_carveout.move(t2)
        moe_layers.append((t13, tail13, codes13, byte13, narrow13, base13, t2, tail2, codes2, byte2, narrow2, base2))
        del t13, tail13, codes13, byte13, narrow13, base13, t2, tail2, codes2, byte2, narrow2, base2

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    carveout_bytes = display_carveout.total_bytes()
    carveout_buffers = [c.nbytes for c in display_carveout._carveouts]
    held_tensors = [embed_vals, lm_vals]
    if use_candidate and any(display_carveout._holds(c, markov_w1) for c in display_carveout._carveouts):
        held_tensors.append(markov_w1)
    for layer_t in moe_layers:
        for t in (layer_t[0], layer_t[6]):
            if any(display_carveout._holds(c, t) for c in display_carveout._carveouts):
                held_tensors.append(t)
    weight_bytes_in_carveout = sum(t.numel() * t.element_size() for t in held_tensors)
    mem_avail_used = avail_0 - mem_avail_bytes()
    cuda_free_used = free_0 - torch.cuda.mem_get_info()[0]
    alloc_used = torch.cuda.memory_allocated() - alloc_0
    res_used = torch.cuda.memory_reserved() - res_0
    peak_res = torch.cuda.max_memory_reserved() - res_0

    stats = {
        "carveout_buffers_mib": [b >> 20 for b in carveout_buffers],
        "carveout_total_mib": carveout_bytes >> 20,
        "carveout_weight_mib": round(weight_bytes_in_carveout / (1 << 20), 3),
        "uma_mem_available_used_mib": round(mem_avail_used / (1 << 20), 2),
        "uma_cuda_free_used_mib": round(cuda_free_used / (1 << 20), 2),
        "cuda_allocated_mib": round(alloc_used / (1 << 20), 3),
        "cuda_reserved_mib": round(res_used / (1 << 20), 3),
        "cuda_peak_reserved_mib": round(peak_res / (1 << 20), 3),
        "_raw_carveout_bytes": carveout_bytes,
        "_raw_weight_bytes": weight_bytes_in_carveout,
        "_raw_mem_avail": mem_avail_used,
        "_raw_cuda_free": cuda_free_used,
        "_raw_alloc": alloc_used,
        "_raw_res": res_used,
        "_raw_peak_res": peak_res,
    }
    if not use_candidate:
        del embed_packed, embed_vals, lm_packed, lm_vals, markov_w1, moe_layers
        for c in list(display_carveout._carveouts):
            c.close()
        display_carveout._carveouts.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return stats, None
    return stats, (embed_packed, embed_vals, lm_packed, lm_vals, markov_w1, moe_layers)

def run_shape(shape_name: str, tp_size: int, vocab_shard: int, moe_inter: int):
    from vllm.v1.worker import display_carveout
    from vllm.models.deepseek_v4_1.tilelang.embedding import (
        declare_packed,
        vocab_embedding_packed,
    )
    from vllm.models.deepseek_v4_1.tilelang.moe import MoeExpertsV41Plan, workspace_bytes
    from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8
    from vllm.v1.worker.gpu.sample import tilelang_sampling as tl_dspark

    device = torch.device("cuda:0")
    hidden = 5120
    vocab_size = 129280
    markov_rank = 256
    num_experts = 128
    topk = 3
    max_rows = 40 if tp_size == 3 else 60

    plan = MoeExpertsV41Plan(
        num_experts, topk, hidden, moe_inter,
        swiglu_limit=7.0, out_dtype=torch.bfloat16,
        idx_dtypes=(torch.int32,), max_rows=max_rows,
    )
    # Warm up plan/packing kernels once before measuring memory so JIT state is settled
    plan.declare_kernels()
    tl_dspark.markov_anchor_embed(
        torch.zeros((5,), dtype=torch.int64, device=device),
        torch.zeros((1,), dtype=torch.int64, device=device),
        torch.zeros((16, markov_rank), dtype=torch.bfloat16, device=device),
        torch.zeros((1, markov_rank), dtype=torch.bfloat16, device=device),
        torch.zeros((1, markov_rank), dtype=torch.bfloat16, device=device),
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    base_stats, _ = measure_allocation_pass(
        device, plan, vocab_size, vocab_shard, hidden, markov_rank, num_experts, moe_inter, False
    )
    cand_stats, live = measure_allocation_pass(
        device, plan, vocab_size, vocab_shard, hidden, markov_rank, num_experts, moe_inter, True
    )
    cand_embed_packed, cand_embed_vals, cand_lm_packed, cand_lm_vals, cand_markov_w1, cand_moe_layers = live

    kernels = {}
    # 1. markov_anchor_embed (markov_w1 in carve-out vs regular)
    reg_markov_w1 = cand_markov_w1.clone()
    for reqs in (1, 3, 8):
        ids = torch.randint(0, vocab_size, (reqs * 5,), dtype=torch.int64, device=device)
        idx = torch.arange(reqs, dtype=torch.int64, device=device) * 5
        step_reg = torch.empty((reqs, markov_rank), dtype=torch.bfloat16, device=device)
        conf_reg = torch.zeros((reqs, markov_rank), dtype=torch.bfloat16, device=device)
        step_co = torch.empty_like(step_reg)
        conf_co = torch.zeros_like(conf_reg)
        t_reg = bench_cuda(lambda: tl_dspark.markov_anchor_embed(ids, idx, reg_markov_w1, step_reg, conf_reg))
        t_co = bench_cuda(lambda: tl_dspark.markov_anchor_embed(ids, idx, cand_markov_w1, step_co, conf_co))
        eq = bool(torch.equal(step_reg, step_co) and torch.equal(conf_reg, conf_co))
        kernels[f"markov_anchor_embed_reqs_{reqs}"] = {
            "regular_us": t_reg,
            "carveout_us": t_co,
            "delta_us": round(t_co - t_reg, 2),
            "ratio": round(t_co / t_reg, 4),
            "bit_identical": eq,
        }
    del reg_markov_w1

    # 2. DSpark MoE (mtp.0 in carve-out vs regular clone)
    t13_co, tail13, codes13, byte13, narrow13, base13, t2_co, tail2, codes2, byte2, narrow2, base2 = cand_moe_layers[0]
    t13_reg = t13_co.clone()
    t2_reg = t2_co.clone()
    weights_reg = ((t13_reg, tail13, codes13, byte13, narrow13, base13), (t2_reg, tail2, codes2, byte2, narrow2, base2))
    weights_co = ((t13_co, tail13, codes13, byte13, narrow13, base13), (t2_co, tail2, codes2, byte2, narrow2, base2))
    ws_nbytes = workspace_bytes(plan.workspace_specs(max_rows, torch.int32))
    ws = torch.empty((ws_nbytes,), dtype=torch.uint8, device=device)
    for M in (5, 15, 40):
        x = cast_mxfp8(torch.randn((M, hidden), dtype=torch.bfloat16, device=device))
        topk_ids = (torch.arange(M * topk, dtype=torch.int32, device=device).reshape(M, topk) * 7) % num_experts
        topk_weights = torch.full((M, topk), 1.0 / topk, dtype=torch.float32, device=device)
        out_reg = torch.empty((M, hidden), dtype=torch.bfloat16, device=device)
        out_co = torch.empty((M, hidden), dtype=torch.bfloat16, device=device)
        t_reg = bench_cuda(lambda: plan.run(x, topk_ids, topk_weights, weights_reg, out_reg, ws, None))
        t_co = bench_cuda(lambda: plan.run(x, topk_ids, topk_weights, weights_co, out_co, ws, None))
        eq = bool(torch.equal(out_reg, out_co))
        kernels[f"dspark_moe_M_{M}"] = {
            "regular_us": t_reg,
            "carveout_us": t_co,
            "delta_us": round(t_co - t_reg, 2),
            "ratio": round(t_co / t_reg, 4),
            "bit_identical": eq,
        }
    del t13_reg, t2_reg, ws

    # 3. Compacted vocab_embedding_packed (embed_tokens in compacted carve-out vs regular)
    declare_packed(hidden, cand_embed_packed.exponent_base)
    reg_embed_vals = cand_embed_vals.clone()
    for M in (1, 5, 40, 4096):
        ids = torch.randint(0, vocab_size, (M,), dtype=torch.int32, device=device)
        out_reg = torch.empty((M, hidden), dtype=torch.bfloat16, device=device)
        out_co = torch.empty((M, hidden), dtype=torch.bfloat16, device=device)
        t_reg = bench_cuda(lambda: vocab_embedding_packed(
            ids, reg_embed_vals, cand_embed_packed.exception_entries,
            cand_embed_packed.row_offsets, hidden, cand_embed_packed.exponent_base,
            out_reg, 0, vocab_shard, 0, vocab_shard, vocab_shard
        ))
        t_co = bench_cuda(lambda: vocab_embedding_packed(
            ids, cand_embed_vals, cand_embed_packed.exception_entries,
            cand_embed_packed.row_offsets, hidden, cand_embed_packed.exponent_base,
            out_co, 0, vocab_shard, 0, vocab_shard, vocab_shard
        ))
        eq = bool(torch.equal(out_reg, out_co))
        kernels[f"vocab_embedding_packed_M_{M}"] = {
            "regular_us": t_reg,
            "carveout_us": t_co,
            "delta_us": round(t_co - t_reg, 2),
            "ratio": round(t_co / t_reg, 4),
            "bit_identical": eq,
        }
    del reg_embed_vals

    del live, cand_embed_packed, cand_embed_vals, cand_lm_packed, cand_lm_vals, cand_markov_w1, cand_moe_layers
    for c in list(display_carveout._carveouts):
        c.close()
    display_carveout._carveouts.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    savings = {
        "additional_carveout_buffers_mib": (cand_stats["_raw_carveout_bytes"] - base_stats["_raw_carveout_bytes"]) >> 20,
        "additional_carveout_weight_mib": round((cand_stats["_raw_weight_bytes"] - base_stats["_raw_weight_bytes"]) / (1 << 20), 3),
        "uma_mem_available_saved_mib": round((base_stats["_raw_mem_avail"] - cand_stats["_raw_mem_avail"]) / (1 << 20), 2),
        "uma_cuda_free_saved_mib": round((base_stats["_raw_cuda_free"] - cand_stats["_raw_cuda_free"]) / (1 << 20), 2),
        "cuda_allocated_saved_mib": round((base_stats["_raw_alloc"] - cand_stats["_raw_alloc"]) / (1 << 20), 3),
        "cuda_reserved_saved_mib": round((base_stats["_raw_res"] - cand_stats["_raw_res"]) / (1 << 20), 3),
        "cuda_peak_reserved_saved_mib": round((base_stats["_raw_peak_res"] - cand_stats["_raw_peak_res"]) / (1 << 20), 3),
    }
    for d in (base_stats, cand_stats):
        for k in list(d):
            if k.startswith("_raw_"):
                del d[k]

    return {
        "shape": shape_name,
        "tp_size": tp_size,
        "baseline": base_stats,
        "candidate": cand_stats,
        "savings": savings,
        "kernels": kernels,
    }

if __name__ == "__main__":
    boot_id = int(sys.argv[1])
    out = {
        "boot": boot_id,
        "tp3": run_shape("tp3-triangle", 3, 43136, 768),
        "tp4": run_shape("tp4-ring", 4, 32320, 576),
    }
    print(json.dumps(out, indent=2))
