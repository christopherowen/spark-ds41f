"""The engine grid's decode steps at a TP3 rank's shapes, on one GB10.

One process, TP group of 1, vLLM's own DeepSeek V4.1 serving code (the V2 model runner,
TileLang attention/linear/MoE, DSpark with 5 drafts and adaptive verification, CUDA graphs
FULL_AND_PIECEWISE on the TP3 recipe's capture ladder 1..48), dummy weights at one TP3
rank's shapes (``iket_step.TP_SHAPES[3]``: 24 heads, 3 WO groups, MoE intermediate 768;
the vocabulary at the TP3 head shard, 43,136 rows). The layer stack is 12 target layers
in the checkpoint's mix (2 SWA, 5 CSA, 5 HCA; two compressor owners, four indexer
sources) plus DSpark's three draft layers; no Engram layer (a dummy step reads no
Engram rows, so the reader cannot act in the grid).

The TP all-reduces do not exist with one rank, but the L2 prefetch windows that overlap
them do (FFN before the attention all-reduce, NEXT before the MoE all-reduce). Each is
followed on the main stream by a spin kernel of ``--allreduce-us`` microseconds, a
stand-in for the one-shot all-reduce the window overlaps in serving (no memory traffic:
the "pure wait" model of experiments/2026-11-08-l2-prefetch-pacing).

The grid is ``GPUModelRunner.engine_grid`` (what servebench's engine step calls through
the dev-mode RPC): each shape's dummy decode step, warm-up runs, then ``--replays``
rounds alternating the shape order, one settle run before each timed run. Each sample has
the whole step (events around the dummy run), the target forward and the drafter.
"""

import argparse
import functools
import hashlib
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/h")

TEXT12 = {
    "num_hidden_layers": 12,
    # 12 target layers in the checkpoint's mix, then the drafter's three (ratio 0).
    "compress_ratios": [0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0],
    "kv_source_layer_ids": [2, 7],
    "index_source_layer_ids": [2, 7, 9, 11],
    "candidate_source_layer_id": 7,
    "engram_layer_ids": [],
    "dspark_target_layer_ids": [9, 10, 11],
}
TP3_HEAD = {"vocab_size": 43136, "dspark_noise_token_id": 43000}
SHAPES = [  # (requests, rows) at 4K context: the TP3 screen's grid cells
    (1, 2), (4, 8), (8, 16), (1, 3), (4, 12), (8, 24), (1, 6), (2, 12), (4, 24), (8, 48),
]


def blob(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify(expect_path, root):
    expected = json.load(open(expect_path))
    bad = {p: (h, blob(os.path.join(root, p))) for p, h in expected.items()
           if blob(os.path.join(root, p)) != h}
    return len(expected), bad


def describe(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    sd = statistics.stdev(s) if n > 1 else 0.0
    return {"n": n, "mean": round(mean, 4), "median": round(s[n // 2], 4), "sd": round(sd, 4),
            "ci95": round(1.96 * sd / n ** 0.5, 4), "min": round(s[0], 4), "max": round(s[-1], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--boot", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", required=True, help="JSON {repo path: git blob} of the arm's files")
    ap.add_argument("--replays", type=int, default=60)
    ap.add_argument("--allreduce-us", type=float, default=30.0)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--full-vocab", action="store_true", help="keep the checkpoint's 129,280 rows")
    ap.add_argument("--shapes", default="", help="RxT,... (default: the TP3 screen's ten 4K cells)")
    args = ap.parse_args()
    shapes = ([tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
              if args.shapes else SHAPES)

    root = "/opt/spark-ds41f/candidate/vllm"
    nfiles, bad = verify(args.expect, root)
    print(f"[tp3grid] arm {args.arm} boot {args.boot}: {nfiles} files checked, {len(bad)} differ",
          flush=True)
    if bad:
        print(json.dumps(bad, indent=1), flush=True)
        raise SystemExit("overlay does not match the arm's tree")

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch

    torch.cuda.init()
    import iket_step as ik

    # Spin calibration: torch.cuda._sleep is clock64-based and graph-capturable.
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    cyc = 2_000_000
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(5):
        a.record()
        torch.cuda._sleep(cyc)
        b.record()
        torch.cuda.synchronize()
        samples.append(a.elapsed_time(b) * 1000.0)
    cycles_per_us = cyc / statistics.median(samples)
    spin_cycles = int(args.allreduce_us * cycles_per_us)
    print(f"[tp3grid] spin: {cycles_per_us:.0f} cycles/us, {spin_cycles} cycles for "
          f"{args.allreduce_us} us", flush=True)

    from vllm.models.deepseek_v4_1 import l2_prefetch as ds_l2

    orig_issue = ds_l2.issue
    counts = {"ffn": 0, "next": 0, "other": 0}

    def issue(plan, num_tokens):
        orig_issue(plan, num_tokens)
        names = " ".join(getattr(plan, "names", None) or [])
        if "gate.weight" in names:
            kind = "ffn"
        elif "wqa_wkv" in names or "engram.wkv" in names:
            kind = "next"
        else:
            counts["other"] += 1
            return
        counts[kind] += 1
        if spin_cycles > 0 and ds_l2.ENABLED and plan is not None and plan.segs is not None:
            torch.cuda._sleep(spin_cycles)

    ds_l2.issue = issue

    from vllm import LLM

    overrides = {**TEXT12, **ik.TP_SHAPES[3], **({} if args.full_vocab else TP3_HEAD)}
    cleanup = ik.same_dummy_weights(0)
    started = time.monotonic()
    max_seqs, drafts = 8, 5
    llm = LLM(
        model=ik.MODEL,
        tokenizer_mode="deepseek_v41",
        load_format="dummy",
        dtype="bfloat16",
        seed=0,
        tensor_parallel_size=1,
        block_size=256,
        max_model_len=65536,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=6144,
        long_prefill_token_threshold=256,
        max_parallel_prefills=1,
        async_scheduling=True,
        kv_cache_memory_bytes=2 << 30,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        attention_backend="TILELANG",
        linear_backend="tilelang",
        moe_backend="tilelang",
        speculative_config={
            "method": "dspark", "num_speculative_tokens": drafts,
            "draft_tensor_parallel_size": 1, "attention_backend": "TILELANG",
            "draft_sample_method": "greedy", "rejection_sample_method": "block",
            "enable_adaptive_verification": True, "adaptive_verification_row_cost": 0.035,
        },
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"],
            "cudagraph_capture_sizes": ik.capture_ladder(max_seqs * (drafts + 1)),
        },
        generation_config="vllm",
        hf_overrides=functools.partial(ik.shrink, overrides=overrides),
        jit_monitor_mode="off",
        limit_mm_per_prompt={"image": 0},
        mm_processor_cache_gb=0,
    )
    cleanup()
    boot_s = time.monotonic() - started
    print(f"[tp3grid] engine ready in {boot_s:.0f} s; issue calls during startup {counts}",
          flush=True)

    spec = {"shapes": [{"num_reqs": r, "num_tokens": t, "context_len": args.context}
                       for r, t in shapes],
            "replays": args.replays, "warmup": 3, "settle": 1}
    t0 = time.monotonic()
    reply = llm.collective_rpc("engine_grid", args=(json.dumps(spec),))[0]
    grid_s = time.monotonic() - t0
    rows = []
    for (r, t), shape in zip(shapes, reply["shapes"]):
        modes = shape.get("modes") or []
        rows.append({
            "label": f"decode {r}x{t} rows @{args.context // 1024}K",
            "num_reqs": r, "num_tokens": t,
            "step_ms": describe(shape["step_ms"]),
            "forward_ms": describe(shape["forward_ms"]),
            "drafter_ms": describe(shape["drafter_ms"]),
            "modes": sorted({json.dumps(m) for m in modes}),
            "samples": {k: [round(v, 4) for v in shape[k]] for k in ("step_ms", "forward_ms", "drafter_ms")},
        })
        f = rows[-1]["forward_ms"]
        print(f"[tp3grid] {rows[-1]['label']:24s} step {rows[-1]['step_ms']['mean']:7.3f} "
              f"forward {f['mean'] if f else float('nan'):7.3f} modes {rows[-1]['modes']}", flush=True)
    result = {
        "arm": args.arm, "boot": args.boot, "boot_s": round(boot_s, 1), "grid_s": round(grid_s, 1),
        "files_checked": nfiles, "allreduce_us": args.allreduce_us,
        "cycles_per_us": round(cycles_per_us, 1), "spin_cycles": spin_cycles,
        "issue_counts_after_grid": counts, "overrides": overrides, "spec": spec,
        "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("VLLM_", "SPARK_", "CUDA_DEVICE"))},
        "device": torch.cuda.get_device_name(), "rows": rows,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"[tp3grid] wrote {args.out}", flush=True)
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
