"""In-process controls (greedy pairs compared on their two written columns) for the two rows where the image A/B put the corrected #998
beyond the interval (embedding at M=4096, FP32 head logits at M=40).

One process, TP3 shard shapes, carve-out required, TileLang backend, the image's own
layer code (as bench_serving_path.py):
  A. embedding lookup, interleaved over rounds, from
       - the table the image serves (with #998: compacted into a tight 320 MiB buffer),
       - the same packed bytes in the prefix of an unpacked-size 432 MiB carve-out
         buffer (the base's layout), and
       - a regular-memory copy;
  B. FP32 head logits and greedy pairs at M=1, 5, 40, interleaved over rounds, with
     960 MiB of other carve-out buffers mapped (the draft experts' share with #998)
     and with them closed (the base's carve-out).
Every output is compared with torch.equal across the variants.

usage: control_inprocess.py LABEL OUT.json
"""
import gc
import json
import os
import sys

import torch

LABEL, OUT = sys.argv[1], sys.argv[2]
assert os.environ.get("SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS") == "1"
DEV = torch.device("cuda:0")
HIDDEN, SHARD = 5120, 43136
ROUNDS = 6


def bench(fn, warmup=20, iters=200):
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


def fill_head_like(dst, seed):
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


from vllm.v1.worker import display_carveout as dc  # noqa: E402
from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding  # noqa: E402
from vllm.model_executor.layers.quantization.online.packed_bf16 import PackedBf16LmHeadMethod  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.embedding import (  # noqa: E402
    TileLangEmbeddingMethod,
    vocab_embedding_packed,
)
from vllm.models.deepseek_v4_1.tilelang.vocab import project_head, project_head_greedy_pairs  # noqa: E402


class _HeadLayer(torch.nn.Module):
    pass


def close_carveout(c):
    """_Carveout.close() where the image has it (#998); else the same driver calls."""
    if hasattr(c, "close"):
        c.close()
        return
    import contextlib
    import ctypes
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuMemFree_v2(c.pointer)
    cuda.cuDestroyExternalMemory(c.external)
    with contextlib.suppress(OSError):
        os.close(c.dmabuf_fd)


def va(t):
    return hex(t.data_ptr())


torch.empty(1, device=DEV)  # a CUDA context before the carve-out's driver calls
torch.cuda.synchronize()
result = {"label": LABEL, "has_998_api": hasattr(dc, "move")}
config = VllmConfig(kernel_config=KernelConfig(linear_backend="tilelang"))
with set_current_vllm_config(config):
    emb = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    torch.nn.Module.__init__(emb)
    emb_method = TileLangEmbeddingMethod()
    with torch.device(DEV):
        emb_method.create_weights(emb, HIDDEN, [SHARD], HIDDEN, SHARD, torch.bfloat16)
    head = _HeadLayer()
    head_method = PackedBf16LmHeadMethod()
    with torch.device(DEV):
        head_method.create_weights(head, HIDDEN, [SHARD], HIDDEN, SHARD, torch.bfloat16)
    fill_head_like(emb.weight.data, 1)
    fill_head_like(head.weight.data, 2)
    torch.cuda.synchronize()
    emb_method.process_weights_after_loading(emb)
    head_method.process_weights_after_loading(head)
    head.tilelang_vocab_dtype = torch.float32
    emb_method.declare_layer_kernels(emb)
    head_method.declare_layer_kernels(head)
    torch.cuda.synchronize()
    served = emb.weight.data
    # The base's layout: the packed rows in the prefix of an unpacked-size buffer.
    unpacked_buf = dc.allocate((SHARD, HIDDEN), torch.bfloat16, DEV)
    prefix = unpacked_buf.view(torch.uint8).view(-1)[: served.numel() * served.element_size()]
    prefix = prefix.view(served.dtype).view(served.shape)
    prefix.copy_(served)
    regular = served.clone()
    torch.cuda.synchronize()
    result["embedding_tables"] = {
        "served": {"va": va(served), "buffer_mib": next(c.nbytes >> 20 for c in dc._carveouts
                                                        if dc._holds(c, served))},
        "unpacked_prefix": {"va": va(prefix), "buffer_mib": next(c.nbytes >> 20 for c in dc._carveouts
                                                                 if dc._holds(c, prefix))},
        "regular": {"va": va(regular)},
    }

    emb_rows = {}
    for M in (40, 4096):
        g = torch.Generator(device=DEV).manual_seed(100 + M)
        ids = torch.randint(0, SHARD, (M,), generator=g, device=DEV, dtype=torch.int32)
        outs = {}
        fns = {}
        for name, table in (("served", served), ("unpacked_prefix", prefix), ("regular", regular)):
            out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=DEV)
            outs[name] = out
            fns[name] = (lambda table=table, out=out: vocab_embedding_packed(
                ids, table, emb.weight_exception_entries, emb.weight_row_offsets, HIDDEN,
                emb.packed_bf16_exponent_base, out, 0, SHARD, 0, SHARD, SHARD))
            fns[name]()
        torch.cuda.synchronize()
        equal = all(torch.equal(outs["served"], o) for o in outs.values())
        rounds = {name: [] for name in fns}
        for _ in range(ROUNDS):
            for name, fn in fns.items():
                rounds[name].append(bench(fn))
        emb_rows[f"M{M}"] = {"equal": equal, "rounds_us": rounds,
                             "median_us": {n: sorted(v)[len(v) // 2] for n, v in rounds.items()}}
    result["embedding"] = emb_rows
    # Release the control buffer (the carve-out cannot hold it and the 960 MiB below).
    control = next(c for c in dc._carveouts if dc._holds(c, unpacked_buf))
    del unpacked_buf, prefix, regular, fns, outs
    gc.collect()
    torch.cuda.synchronize()
    close_carveout(control)
    dc._carveouts.remove(control)

    # B. Head with and without 960 MiB of other carve-out buffers mapped.
    extra = []
    head_rows = {}
    xs = {}
    for M in (1, 5, 40):
        g = torch.Generator(device=DEV).manual_seed(200 + M)
        xs[M] = (torch.randn(M, HIDDEN, generator=g, device=DEV) * 0.5).bfloat16()
    reference = {M: (project_head(head, xs[M]).clone(),
                     project_head_greedy_pairs(head, xs[M], vocab=SHARD, vocab_start=0)[:, :2].clone())
                 for M in xs}
    rounds = {f"{kind}_M{M}_{state}": [] for kind in ("logits", "greedy_pairs") for M in xs
              for state in ("extra_mapped", "extra_closed")}
    equal = True
    for _ in range(ROUNDS):
        for state in ("extra_mapped", "extra_closed"):
            if state == "extra_mapped":
                for mib in (480, 240, 240):
                    t = dc.allocate((mib << 20,), torch.uint8, DEV)
                    t.fill_(0x5A)
                    extra.append(t)
                torch.cuda.synchronize()
            for M, x in xs.items():
                rounds[f"logits_M{M}_{state}"].append(bench(lambda x=x: project_head(head, x)))
                rounds[f"greedy_pairs_M{M}_{state}"].append(bench(
                    lambda x=x: project_head_greedy_pairs(head, x, vocab=SHARD, vocab_start=0)))
                equal = equal and torch.equal(project_head(head, x), reference[M][0])
                equal = equal and torch.equal(
                    project_head_greedy_pairs(head, x, vocab=SHARD, vocab_start=0)[:, :2], reference[M][1])
            if state == "extra_mapped":
                ptrs = {t.data_ptr() for t in extra}
                for c in [c for c in dc._carveouts if c.pointer.value in ptrs]:
                    close_carveout(c)
                    dc._carveouts.remove(c)
                extra.clear()
                torch.cuda.synchronize()
    head_rows["equal"] = equal
    head_rows["rounds_us"] = rounds
    head_rows["median_us"] = {n: sorted(v)[len(v) // 2] for n, v in rounds.items()}
    result["head"] = head_rows

with open(OUT, "w") as f:
    json.dump(result, f, indent=1)
print(json.dumps({"embedding": {k: v["median_us"] for k, v in result["embedding"].items()},
                  "embedding_equal": {k: v["equal"] for k, v in result["embedding"].items()},
                  "head": result["head"]["median_us"], "head_equal": result["head"]["equal"],
                  "tables": result["embedding_tables"]}, indent=1))
