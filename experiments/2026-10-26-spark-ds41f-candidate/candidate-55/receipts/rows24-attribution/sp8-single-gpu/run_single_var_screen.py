import functools, gc, json, os, sys, time
from pathlib import Path

sys.path.insert(0, "/b")
import iket_step

SHAPES = [
    ("1x2@4K",   {"num_reqs": 1,  "num_tokens": 2,  "context_len": 4096}),
    ("1x3@4K",   {"num_reqs": 1,  "num_tokens": 3,  "context_len": 4096}),
    ("1x6@4K",   {"num_reqs": 1,  "num_tokens": 6,  "context_len": 4096}),
    ("4x8@4K",   {"num_reqs": 4,  "num_tokens": 8,  "context_len": 4096}),
    ("4x12@4K",  {"num_reqs": 4,  "num_tokens": 12, "context_len": 4096}),
    ("8x16@4K",  {"num_reqs": 8,  "num_tokens": 16, "context_len": 4096}),
    ("8x24@4K",  {"num_reqs": 8,  "num_tokens": 24, "context_len": 4096}),
    ("16x32@4K", {"num_reqs": 16, "num_tokens": 32, "context_len": 4096}),
    ("16x48@4K", {"num_reqs": 16, "num_tokens": 48, "context_len": 4096}),
    ("12x72@4K", {"num_reqs": 12, "num_tokens": 72, "context_len": 4096}),
    ("16x96@4K", {"num_reqs": 16, "num_tokens": 96, "context_len": 4096}),
]

def main():
    label = sys.argv[1]
    out_file = Path(sys.argv[2])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import torch
    from vllm import LLM
    overrides = {**iket_step.TEXT_OVERRIDES, **iket_step.TP_SHAPES[4]}
    cleanup_dummy = iket_step.same_dummy_weights(0)
    t0 = time.monotonic()
    llm = LLM(
        model=iket_step.MODEL,
        tokenizer_mode="deepseek_v41",
        load_format="dummy",
        dtype="bfloat16",
        seed=0,
        tensor_parallel_size=1,
        block_size=256,
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=512,
        long_prefill_token_threshold=256,
        max_parallel_prefills=1,
        async_scheduling=True,
        kv_cache_memory_bytes=2 << 30,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        enforce_eager=False,
        attention_backend="TILELANG",
        linear_backend="tilelang",
        moe_backend="tilelang",
        enable_flashinfer_autotune=False,
        speculative_config={
            "method": "dspark",
            "num_speculative_tokens": 5,
            "draft_tensor_parallel_size": 1,
            "attention_backend": "TILELANG",
            "draft_sample_method": "greedy",
            "rejection_sample_method": "block",
            "enable_adaptive_verification": True,
            "adaptive_verification_row_cost": 0.035,
        },
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "custom_ops": ["all"],
            "cudagraph_capture_sizes": [2, 3, 6, 8, 12, 16, 20, 24, 32, 48, 72, 80, 96],
        },
        generation_config="vllm",
        hf_overrides=functools.partial(iket_step.shrink, overrides=overrides),
        jit_monitor_mode="off",
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
    )
    cleanup_dummy()
    print(f"[{label}] LLM ready in {time.monotonic() - t0:.1f}s", flush=True)

    replays = 60
    results = {"label": label, "replays": replays, "shapes": {}}
    for name, shape in SHAPES:
        warm_spec = {"shapes": [shape], "replays": 10, "warmup": 5, "settle": 2}
        llm.collective_rpc("engine_grid", args=(json.dumps(warm_spec),))
        torch.cuda.synchronize()

        meas_spec = {"shapes": [shape], "replays": replays, "warmup": 0, "settle": 0}
        reply = llm.collective_rpc("engine_grid", args=(json.dumps(meas_spec),))[0]["shapes"][0]
        torch.cuda.synchronize()
        fwd_sorted = sorted(reply["forward_ms"])
        drf_sorted = sorted(reply["drafter_ms"])
        stp_sorted = sorted(reply["step_ms"])
        f_med = fwd_sorted[replays // 2]
        d_med = drf_sorted[replays // 2]
        s_med = stp_sorted[replays // 2]
        results["shapes"][name] = {
            "fwd_med": round(f_med, 4),
            "fwd_mean": round(sum(reply["forward_ms"]) / replays, 4),
            "drf_med": round(d_med, 4),
            "stp_med": round(s_med, 4),
        }
        print(f"  [{label:14s}] {name:10s}: fwd_med={f_med:.3f} ms  drf_med={d_med:.3f} ms  step_med={s_med:.3f} ms", flush=True)

    out_file.write_text(json.dumps(results, indent=2) + "\n")
    llm.llm_engine.engine_core.shutdown()

if __name__ == "__main__":
    main()
