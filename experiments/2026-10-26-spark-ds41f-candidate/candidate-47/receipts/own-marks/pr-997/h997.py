#!/usr/bin/env python3
"""Own-mark harness for PR #997 (ours, not the PR's): project_head_greedy_pairs on the
real DeepSeek-V4.1-Flash-1017 head shard, built by the PR bench's make_layer, timed
eagerly (the bench's back-to-back loop) and by CUDA-graph replay, with the layer's
memory footprint and output hashes for cross-image bit-exactness.

usage: h997.py LABEL   (writes /out/h997-LABEL.json); SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS
selects the carve-out as in the PR bench.
"""

import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time

import torch
from safetensors import safe_open

spec = importlib.util.spec_from_file_location("b997", "/tmp/bench_two_boots.py")
b997 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b997)

from vllm.models.deepseek_v4_1.tilelang.vocab import project_head, project_head_greedy_pairs  # noqa: E402
from vllm.v1.worker import display_carveout  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.greedy_verification import vocab_parallel_pairs  # noqa: E402

LABEL = sys.argv[1]
DEV = torch.device("cuda")
M_LIST = [1, 2, 4, 6, 8, 16, 32, 48]
WINDOWS = 30
REPS = 20


def summ(xs):
    xs = list(xs)
    n = len(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    return {"median": round(statistics.median(xs), 2), "mean": round(statistics.fmean(xs), 2),
            "ci95": round(1.96 * sd / math.sqrt(n), 2), "n": n}


def meminfo_kib(key="MemAvailable"):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return -1


def carveout_bytes():
    return sum(getattr(c, "nbytes", 0) for c in display_carveout._carveouts)


def main():
    use_carveout = os.environ.get("SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS", "0") == "1"
    out = {"label": LABEL, "carveout": use_carveout, "device": torch.cuda.get_device_name(0),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shapes": {}}
    with safe_open("/weights/head.safetensors", framework="pt", device="cuda") as f:
        full_w = f.get_tensor("head.weight")
    for shape_name, n, vocab, vocab_start in [("tp4-ring", 32320, 32320, 0), ("tp3-triangle", 43136, 43094, 0)]:
        w = torch.zeros((n, 5120), dtype=torch.bfloat16, device=DEV)
        w[:vocab].copy_(full_w[vocab_start: vocab_start + vocab])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem0 = {"cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
                "memavailable_kib": meminfo_kib(), "carveout_bytes": carveout_bytes()}
        layer = b997.make_layer(w, use_carveout=use_carveout)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem1 = {"cuda_allocated": torch.cuda.memory_allocated(), "cuda_reserved": torch.cuda.memory_reserved(),
                "memavailable_kib": meminfo_kib(), "carveout_bytes": carveout_bytes()}
        res = {"n": n, "vocab": vocab, "memory_before_layer": mem0, "memory_after_layer": mem1,
               "layer_cuda_allocated_MiB": round((mem1["cuda_allocated"] - mem0["cuda_allocated"]) / 2**20, 2),
               # make_layer clears display_carveout._carveouts first, so after it the
               # list holds this layer's carve-outs only
               "layer_carveout_MiB": round(mem1["carveout_bytes"] / 2**20, 2),
               "memavailable_drop_MiB": round((mem0["memavailable_kib"] - mem1["memavailable_kib"]) / 1024, 1),
               "layer_tensors_MiB": {}, "bits": {}, "timing": {}}
        for name in dir(layer):
            try:
                t = getattr(layer, name)
            except Exception:
                continue
            ts = t if isinstance(t, (tuple, list)) else (t,)
            nb = sum(x.numel() * x.element_size() for x in ts if isinstance(x, torch.Tensor) and x.is_cuda)
            if nb >= 1 << 20:
                res["layer_tensors_MiB"][name] = round(nb / 2**20, 2)
        # warm every row tile, largest first so the scratch is sized once
        for m in sorted(M_LIST, reverse=True) + M_LIST:
            xw = torch.randn((m, 5120), dtype=torch.bfloat16, device=DEV)
            project_head_greedy_pairs(layer, xw, vocab=vocab, vocab_start=vocab_start, scratch_pairs=True)
        torch.cuda.synchronize()
        # bit-exactness: random rows and near-tie blends, against the image's full logits
        g = torch.Generator(device=DEV).manual_seed(20261109)
        mism, rows = 0, 0
        for m in M_LIST:
            hs = []
            for trial in range(4):
                if trial < 3:
                    x = (torch.randn((m, 5120), generator=g, device=DEV) / (5120 ** 0.5)).bfloat16()
                else:
                    i1 = torch.randint(0, vocab, (m,), generator=g, device=DEV)
                    i2 = torch.randint(0, vocab, (m,), generator=g, device=DEV)
                    w1 = torch.nn.functional.normalize(w[i1].float(), dim=1)
                    w2 = torch.nn.functional.normalize(w[i2].float(), dim=1)
                    x = (0.5 * (w1 + w2) + torch.randn((m, 5120), generator=g, device=DEV) * 1e-3).bfloat16()
                actual = project_head_greedy_pairs(layer, x, vocab=vocab, vocab_start=vocab_start, scratch_pairs=False)
                expected = vocab_parallel_pairs(project_head(layer, x), vocab, vocab_start, tp_size=1)
                a = actual[:, :2].contiguous().view(torch.int32)
                if not torch.equal(a, expected[:, :2].contiguous().view(torch.int32)):
                    mism += 1
                rows += m
                hs.append(hashlib.sha256(a.cpu().numpy().tobytes()).hexdigest()[:16])
            res["bits"][str(m)] = hs
        res["mismatched_batches"] = mism
        res["rows_checked"] = rows
        # timing: eager back-to-back loop and graph replay, interleaved window by window
        gb = torch.Generator(device=DEV).manual_seed(777)
        for m in M_LIST:
            x = (torch.randn((m, 5120), generator=gb, device=DEV) / (5120 ** 0.5)).bfloat16()

            def call():
                project_head_greedy_pairs(layer, x, vocab=vocab, vocab_start=vocab_start, scratch_pairs=True)

            for _ in range(10):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(REPS):
                    call()
            graph.replay()
            torch.cuda.synchronize()
            eager, replay = [], []
            for _ in range(WINDOWS):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(REPS):
                    call()
                e.record()
                torch.cuda.synchronize()
                eager.append(s.elapsed_time(e) * 1e3 / REPS)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                graph.replay()
                e.record()
                torch.cuda.synchronize()
                replay.append(s.elapsed_time(e) * 1e3 / REPS)
            scratch = getattr(layer, "_tilelang_vocab_greedy_scratch", None)
            active = int(scratch[8].item()) if scratch is not None and len(scratch) >= 9 else None
            res["timing"][str(m)] = {"eager_us": summ(eager), "graph_us": summ(replay), "active_cols": active}
            print(shape_name, m, res["timing"][str(m)]["eager_us"]["median"], res["timing"][str(m)]["graph_us"]["median"], active, flush=True)
            del graph
        out["shapes"][shape_name] = res
        print(shape_name, "memory", json.dumps({k: res[k] for k in ("layer_cuda_allocated_MiB", "layer_carveout_MiB", "memavailable_drop_MiB")}),
              "mismatched_batches", mism, flush=True)
        del layer, w
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"/out/h997-{LABEL}.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", f"/out/h997-{LABEL}.json")


if __name__ == "__main__":
    main()
