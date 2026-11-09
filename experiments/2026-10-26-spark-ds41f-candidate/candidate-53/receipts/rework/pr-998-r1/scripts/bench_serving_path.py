"""#998's own mark on the path serving takes at TP3 (sp8: one GB10, TP3 shard shapes).

Builds, in serving order and through the image's own layer code, what the carve-out
holds at TP3 with SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1 and --linear-backend tilelang:
  1. the target's embed_tokens shard (VocabParallelEmbedding + TileLangEmbeddingMethod:
     create_weights -> load -> process_weights_after_loading: 12-bit pack, and with
     #998 compact),
  2. the target's lm_head shard (PackedBf16LmHeadMethod, TileLang backend: the 1.25x
     carve-out buffer whose tail holds the 8-bit screening table; pack, screen),
  3. DSpark's replicated markov_w1 (DSparkMarkovHead.__init__, the image's code),
  4. DSpark's three draft MoE layers (TileLangExperts.process_weights_after_loading
     with a real MoeExpertsV41Plan, router max_rows=40: decode-only),
then confirm_streamed_weights as the model runner calls it, and measures memory
(carve-out buffers, torch allocated/reserved, host MemAvailable) and lookup times,
and hashes every output (bit-exactness across images).

usage: bench_serving_path.py LABEL BOOT OUT.json
"""
import gc
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

LABEL, BOOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
assert os.environ.get("SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS") == "1"
DEV = torch.device("cuda:0")
HIDDEN, VOCAB, SHARD, RANK = 5120, 129280, 43136, 256
NUM_EXPERTS, TOPK, INTER, MAX_ROWS = 128, 3, 768, 40  # DSpark draft MoE at TP3


def mem_available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def settle():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def sha(t: torch.Tensor) -> str:
    t = t.detach().contiguous()
    return hashlib.sha256(t.view(torch.uint8).reshape(-1).cpu().numpy().tobytes()).hexdigest()[:16]


def bench(fn, warmup=30, iters=200):
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


def fill_head_like(dst: torch.Tensor, seed: int):
    """Head-like BF16 rows (randn * 0.02) with out-of-window values, in row chunks."""
    rows = dst.shape[0]
    for start in range(0, rows, 4096):
        end = min(start + 4096, rows)
        g = torch.Generator(device=DEV).manual_seed(seed * 100003 + start)
        chunk = (torch.randn(end - start, dst.shape[1], generator=g, device=DEV) * 0.02).bfloat16()
        flat = chunk.view(-1)
        picks = torch.randint(0, flat.numel(), (16,), generator=g, device=DEV)
        flat[picks[:8]] = 300.0
        flat[picks[8:12]] = -1.5e-7
        flat[picks[12:]] = 0.0
        dst[start:end].copy_(chunk)
        del chunk, flat, picks


from vllm.v1.worker import display_carveout as dc  # noqa: E402
from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding  # noqa: E402
from vllm.model_executor.layers.quantization.online.packed_bf16 import PackedBf16LmHeadMethod  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.embedding import (  # noqa: E402
    TileLangEmbeddingMethod,
    vocab_embedding_packed,
)
from vllm.models.deepseek_v4_1.tilelang.vocab import project_head, project_head_greedy_pairs  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.moe import workspace_bytes  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8  # noqa: E402
from vllm.model_executor.layers.fused_moe import tilelang as tl_moe  # noqa: E402
from vllm.model_executor.models import qwen3_dspark  # noqa: E402
from vllm.v1.worker.gpu.sample import tilelang_sampling as tl_dspark  # noqa: E402

assert dc.required(), "carve-out must be required (TP3 recipe environment)"
has_998 = hasattr(dc, "move")


class _HeadLayer(torch.nn.Module):
    pass


# The head layer stands in for ParallelLMHead (a VocabParallelEmbedding) in the
# carve-out's confirmation, as the vocab release test does.
dc.VocabParallelEmbedding = (VocabParallelEmbedding, _HeadLayer)

result = {"label": LABEL, "boot": BOOT, "has_998_api": has_998,
          "shapes": {"hidden": HIDDEN, "vocab": VOCAB, "shard_rows": SHARD, "markov_rank": RANK,
                     "experts": NUM_EXPERTS, "topk": TOPK, "intermediate": INTER, "max_rows": MAX_ROWS}}

config = VllmConfig(kernel_config=KernelConfig(linear_backend="tilelang"))
with set_current_vllm_config(config):
    settle()
    torch.cuda.reset_peak_memory_stats()
    avail0, free0 = mem_available(), torch.cuda.mem_get_info()[0]
    alloc0, res0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    t0 = time.perf_counter()

    # 1. Target embed_tokens shard.
    emb = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    torch.nn.Module.__init__(emb)
    emb_method = TileLangEmbeddingMethod()
    with torch.device(DEV):
        emb_method.create_weights(emb, HIDDEN, [SHARD], HIDDEN, SHARD, torch.bfloat16)
    # 2. Target lm_head shard (created with the model, before loading).
    head = _HeadLayer()
    head_method = PackedBf16LmHeadMethod()
    with torch.device(DEV):
        head_method.create_weights(head, HIDDEN, [SHARD], HIDDEN, SHARD, torch.bfloat16)
    # Loading streams the checkpoint in.
    fill_head_like(emb.weight.data, 1)
    fill_head_like(head.weight.data, 2)
    settle()
    emb_method.process_weights_after_loading(emb)
    head_method.process_weights_after_loading(head)
    head.tilelang_vocab_dtype = torch.float32  # VLLM_DS41_HEAD_DTYPE=float32
    settle()
    after_target = [c.nbytes >> 20 for c in dc._carveouts]

    # 3. DSpark markov head (the image's own constructor; markov_w2 is a TP-sharded
    # 21 MiB head below the carve-out threshold, stubbed out here).
    real_head_cls = qwen3_dspark.ParallelLMHead
    qwen3_dspark.ParallelLMHead = lambda *a, **k: torch.nn.Module()
    torch.set_default_dtype(torch.bfloat16)  # the model dtype, as the loader sets it
    with torch.device(DEV):
        markov = qwen3_dspark.DSparkMarkovHead(VOCAB, VOCAB, RANK, prefix="model.markov_head")
    torch.set_default_dtype(torch.float32)
    qwen3_dspark.ParallelLMHead = real_head_cls
    g = torch.Generator(device=DEV).manual_seed(3)
    markov.markov_w1.weight.data.copy_(
        (torch.randn(VOCAB, RANK, generator=g, device=DEV) * 0.02).bfloat16())
    settle()

    # 4. Three decode-only DSpark MoE layers.
    experts_layers = []
    for index in range(3):
        layer = torch.nn.Module()
        g = torch.Generator(device=DEV).manual_seed(10 + index)
        layer.register_parameter("w13_weight", torch.nn.Parameter(torch.randint(
            0, 256, (NUM_EXPERTS, 2 * INTER, HIDDEN // 2), generator=g, device=DEV,
            dtype=torch.uint8), requires_grad=False))
        layer.register_parameter("w13_weight_scale", torch.nn.Parameter(torch.randint(
            118, 128, (NUM_EXPERTS, 2 * INTER, HIDDEN // 32), generator=g, device=DEV,
            dtype=torch.uint8), requires_grad=False))
        layer.register_parameter("w2_weight", torch.nn.Parameter(torch.randint(
            0, 256, (NUM_EXPERTS, HIDDEN, INTER // 2), generator=g, device=DEV,
            dtype=torch.uint8), requires_grad=False))
        layer.register_parameter("w2_weight_scale", torch.nn.Parameter(torch.randint(
            118, 128, (NUM_EXPERTS, HIDDEN, INTER // 32), generator=g, device=DEV,
            dtype=torch.uint8), requires_grad=False))
        layer.apply_router_weight_on_input = False
        layer.router = SimpleNamespace(max_rows=MAX_ROWS, indices_dtype=torch.int32)
        experts = tl_moe.TileLangExperts.__new__(tl_moe.TileLangExperts)
        experts.quant_config = SimpleNamespace(
            gemm1_clamp_limit=None, _w1=SimpleNamespace(scale=None), _w2=SimpleNamespace(scale=None))
        experts.moe_config = SimpleNamespace(
            experts_per_token=TOPK, swiglu_limit=7.0, in_dtype=torch.bfloat16)
        experts.fold_shared = False
        experts.process_weights_after_loading(layer)
        settle()
        experts_layers.append((layer, experts))

    build_s = time.perf_counter() - t0
    settle()
    speculator = torch.nn.Module()
    speculator.markov_head = markov
    for index, (layer, _) in enumerate(experts_layers):
        speculator.add_module(f"layer{index}", layer)
    model = torch.nn.Module()
    model.embed_tokens = emb
    model.lm_head = head
    if has_998:
        dc.confirm_streamed_weights(model, DEV, speculator)
    else:
        dc.confirm_streamed_weights(model, DEV)
    streamed = [name for name, _ in (dc.streamed_weights(model, speculator) if has_998
                                     else dc.streamed_weights(model))]

    def in_carveout(t):
        return any(dc._holds(c, t) for c in dc._carveouts)

    placement = {
        "embed_tokens.weight": in_carveout(emb.weight),
        "lm_head.weight": in_carveout(head.weight),
        "lm_head.screen_wq": in_carveout(head._tilelang_vocab_screen[0]),
        "markov_w1.weight": in_carveout(markov.markov_w1.weight),
    }
    for index, (layer, _) in enumerate(experts_layers):
        placement[f"layer{index}.w13_weight"] = in_carveout(layer.w13_weight)
        placement[f"layer{index}.w2_weight"] = in_carveout(layer.w2_weight)
    regular_weight_mib = 0.0
    for name, t in [("emb", emb.weight), ("head", head.weight), ("markov", markov.markov_w1.weight)] + [
            (f"l{i}{w}", getattr(l, w)) for i, (l, _) in enumerate(experts_layers)
            for w in ("w13_weight", "w2_weight")]:
        if not in_carveout(t):
            regular_weight_mib += t.numel() * t.element_size() / 2**20
    result["memory"] = {
        "carveout_buffers_mib": [c.nbytes >> 20 for c in dc._carveouts],
        "carveout_buffers_after_target_mib": after_target,
        "carveout_total_mib": sum(c.nbytes for c in dc._carveouts) >> 20,
        "placement_in_carveout": placement,
        "carveout_weights_regular_memory_mib": round(regular_weight_mib, 3),
        "cuda_allocated_mib": round((torch.cuda.memory_allocated() - alloc0) / 2**20, 3),
        "cuda_reserved_mib": round((torch.cuda.memory_reserved() - res0) / 2**20, 3),
        "cuda_peak_reserved_mib": round((torch.cuda.max_memory_reserved() - res0) / 2**20, 3),
        "host_mem_available_used_mib": round((avail0 - mem_available()) / 2**20, 2),
        "cuda_free_used_mib": round((free0 - torch.cuda.mem_get_info()[0]) / 2**20, 2),
        "build_seconds": round(build_s, 2),
        "streamed_weights": streamed,
    }

    hashes, times = {}, {}
    hashes["embed_tokens.weight"] = sha(emb.weight)
    hashes["embed_tokens.exceptions"] = sha(emb.weight_exception_entries)
    hashes["lm_head.weight"] = sha(head.weight)
    for i, t in enumerate(head._tilelang_vocab_screen):
        hashes[f"lm_head.screen[{i}]"] = sha(t)
    hashes["markov_w1.weight"] = sha(markov.markov_w1.weight)
    for index, (layer, experts) in enumerate(experts_layers):
        hashes[f"layer{index}.w13_weight"] = sha(layer.w13_weight)
        hashes[f"layer{index}.w2_weight"] = sha(layer.w2_weight)

    # Embedding lookups (the target's embed_tokens).
    emb_method.declare_layer_kernels(emb)
    for M in (1, 5, 40, 4096):
        g = torch.Generator(device=DEV).manual_seed(100 + M)
        ids = torch.randint(0, SHARD, (M,), generator=g, device=DEV, dtype=torch.int32)
        out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=DEV)
        fn = lambda: vocab_embedding_packed(  # noqa: E731
            ids, emb.weight, emb.weight_exception_entries, emb.weight_row_offsets,
            HIDDEN, emb.packed_bf16_exponent_base, out, 0, SHARD, 0, SHARD, SHARD)
        fn()
        hashes[f"embedding_M{M}"] = sha(out)
        times[f"embedding_M{M}"] = bench(fn)

    # LM head: FP32 logits and the certified greedy pairs (the greedy decode path).
    head_method.declare_layer_kernels(head)
    for M in (1, 5, 40):
        g = torch.Generator(device=DEV).manual_seed(200 + M)
        x = (torch.randn(M, HIDDEN, generator=g, device=DEV) * 0.5).bfloat16()
        logits = project_head(head, x)
        hashes[f"lm_head_logits_M{M}"] = sha(logits)
        pairs = project_head_greedy_pairs(head, x, vocab=SHARD, vocab_start=0)
        hashes[f"lm_head_greedy_pairs_M{M}"] = sha(pairs)
        times[f"lm_head_logits_M{M}"] = bench(lambda: project_head(head, x))
        times[f"lm_head_greedy_pairs_M{M}"] = bench(
            lambda: project_head_greedy_pairs(head, x, vocab=SHARD, vocab_start=0))

    # Markov anchor embedding (DSpark drafting).
    w1 = markov.markov_w1.weight
    for reqs in (1, 3, 8):
        g = torch.Generator(device=DEV).manual_seed(300 + reqs)
        ids = torch.randint(0, VOCAB, (reqs * 5,), generator=g, device=DEV, dtype=torch.int64)
        idx = torch.arange(reqs, dtype=torch.int64, device=DEV) * 5
        step = torch.empty((reqs, RANK), dtype=torch.bfloat16, device=DEV)
        conf = torch.zeros((reqs, RANK), dtype=torch.bfloat16, device=DEV)
        fn = lambda: tl_dspark.markov_anchor_embed(ids, idx, w1, step, conf)  # noqa: E731
        fn()
        hashes[f"markov_anchor_embed_reqs{reqs}"] = sha(torch.cat([step, conf]))
        times[f"markov_anchor_embed_reqs{reqs}"] = bench(fn)

    # DSpark MoE layers (draft steps: M = 5 rows a request).
    for index, (layer, experts) in enumerate(experts_layers):
        experts.declare_kernels()
        plan = experts._plan
        tail13, tail2 = experts._tails
        base13, base2 = experts._bases
        byte13, byte2 = experts._codes_2bit
        narrow13, narrow2 = experts._narrow
        weights = ((layer.w13_weight, tail13, layer.w13_weight_scale, byte13, narrow13, base13),
                   (layer.w2_weight, tail2, layer.w2_weight_scale, byte2, narrow2, base2))
        ws = torch.empty((workspace_bytes(plan.workspace_specs(MAX_ROWS, torch.int32)),),
                         dtype=torch.uint8, device=DEV)
        for M in (5, 15, 40):
            g = torch.Generator(device=DEV).manual_seed(400 + M)
            x = cast_mxfp8((torch.randn((M, HIDDEN), generator=g, device=DEV) * 0.5).bfloat16())
            topk_ids = torch.stack([torch.randperm(NUM_EXPERTS, generator=g, device=DEV)[:TOPK]
                                    for _ in range(M)]).to(torch.int32).contiguous()
            topk_weights = torch.softmax(torch.randn((M, TOPK), generator=g, device=DEV,
                                                     dtype=torch.float32), dim=-1).contiguous()
            out = torch.empty((M, HIDDEN), dtype=plan.out_dtype, device=DEV)
            fn = lambda: plan.run(x, topk_ids, topk_weights, weights, out, ws, None)  # noqa: E731
            fn()
            torch.cuda.synchronize()
            hashes[f"dspark_moe_layer{index}_M{M}"] = sha(out)
            times[f"dspark_moe_layer{index}_M{M}"] = bench(fn)
        del ws

    result["hashes"] = hashes
    result["times_us"] = times

with open(OUT, "w") as f:
    json.dump(result, f, indent=1)
print(json.dumps(result["memory"], indent=1))
print(json.dumps(times, indent=1))
