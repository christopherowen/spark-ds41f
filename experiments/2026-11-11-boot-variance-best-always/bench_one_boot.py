import ctypes
import functools
import json
import math
import os
import statistics as st
import subprocess
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch

LIBC = ctypes.CDLL(None)
def cpu_now():
    return int(LIBC.sched_getcpu())

def gpu_telemetry():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=clocks.current.graphics,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"], text=True
    ).strip().split(",")
    vals = [float(x.strip()) if "N/A" not in x else -1.0 for x in out]
    return {"sm_mhz": vals[0], "temp_c": vals[1], "power_w": vals[2]}

MODEL = ("/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
         "1bae0e1088993b9b82de08f59de5be5a34e42170")
OVERRIDES = {
    "num_hidden_layers": 16,
    "compress_ratios": [0, 0] + [2] * 6 + [1] * 5 + [0, 0, 0],
    "kv_source_layer_ids": [2, 5, 8, 11],
    "index_source_layer_ids": [2, 5, 8, 11, 12],
    "candidate_source_layer_id": 8,
    "engram_layer_ids": [],
    "dspark_target_layer_ids": [13, 14, 15],
    "num_attention_heads": 16,
    "o_groups": 2,
    "moe_intermediate_size": 576,
}
LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96]
SHAPES = [
    (1, 2, 4096), (4, 8, 4096), (8, 16, 4096), (16, 32, 4096),
    (1, 3, 4096), (4, 12, 4096), (8, 24, 4096), (16, 48, 4096),
    (1, 6, 4096), (16, 96, 4096),
    (1, 3, 32768), (8, 24, 32768),
]

def shrink(config, overrides):
    text = config.get_text_config()
    for k, v in overrides.items():
        setattr(text, k, v)
    return config

def patch_compressor_tp4():
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    import vllm.models.deepseek_v4_1.compressor as comp
    orig_init = comp.DeepseekCompressor.__init__
    @functools.wraps(orig_init)
    def init(self, *a, **kw):
        saved = [(m, getattr(m, "get_tensor_model_parallel_world_size"), getattr(m, "get_tensor_model_parallel_rank")) for m in (linear, parameter)]
        for m in (linear, parameter):
            m.get_tensor_model_parallel_world_size = lambda: 4
            m.get_tensor_model_parallel_rank = lambda: 0
        try:
            orig_init(self, *a, **kw)
        finally:
            for m, ws, rk in saved:
                m.get_tensor_model_parallel_world_size = ws
                m.get_tensor_model_parallel_rank = rk
    comp.DeepseekCompressor.__init__ = init
    comp.tensor_model_parallel_all_gather = lambda t, dim=0: t.repeat(4, 1)

def main():
    out_path = sys.argv[1]
    arm_label = sys.argv[2]
    replays = int(sys.argv[3]) if len(sys.argv) > 3 else 80

    torch.cuda.init()
    def gpu_warmup():
        a = torch.randn((4096, 4096), dtype=torch.bfloat16, device="cuda")
        for _ in range(25):
            _ = a @ a
        torch.cuda.synchronize()
    patch_compressor_tp4()
    from vllm import LLM

    t0 = time.monotonic()
    llm = LLM(
        model=MODEL,
        tokenizer_mode="deepseek_v41",
        load_format="dummy",
        dtype="bfloat16",
        seed=0,
        tensor_parallel_size=1,
        block_size=256,
        max_model_len=65536,
        max_num_seqs=16,
        max_num_batched_tokens=512,
        long_prefill_token_threshold=256,
        max_parallel_prefills=1,
        async_scheduling=True,
        kv_cache_memory_bytes=2 << 30,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        attention_backend="TILELANG",
        linear_backend="tilelang",
        moe_backend="tilelang",
        speculative_config={
            "method": "dspark", "num_speculative_tokens": 5,
            "draft_tensor_parallel_size": 1, "attention_backend": "TILELANG",
            "draft_sample_method": "greedy", "rejection_sample_method": "block",
            "enable_adaptive_verification": True, "adaptive_verification_row_cost": 0.035,
        },
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"],
                            "cudagraph_capture_sizes": LADDER},
        generation_config="vllm",
        hf_overrides=functools.partial(shrink, overrides=OVERRIDES),
        jit_monitor_mode="off",
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
    )
    boot_s = round(time.monotonic() - t0, 2)
    aff_after_boot = sorted(os.sched_getaffinity(0))
    cpu_pre = cpu_now()
    gpu_warmup()
    tel_pre = gpu_telemetry()

    spec = {"shapes": [{"num_reqs": r, "context_len": c, "num_tokens": t} for r, t, c in SHAPES],
            "replays": replays, "warmup": 3, "settle": 1}
    reply = llm.collective_rpc("engine_grid", args=(json.dumps(spec),))[0]
    tel_post = gpu_telemetry()
    cpu_post = cpu_now()

    rows = []
    for (r, t, c), row in zip(SHAPES, reply["shapes"]):
        rows.append({
            "label": f"decode {r}x{t} rows @{c//1024}K",
            "shape": {"num_reqs": r, "num_tokens": t, "context_len": c},
            "samples": [round(v, 4) for v in row["step_ms"]],
            "forward_ms": [round(v, 4) for v in row["forward_ms"]],
            "drafter_ms": [round(v, 4) for v in row["drafter_ms"]],
        })

    def gm_metric(fn, key="samples"):
        return round(math.exp(sum(math.log(fn(x[key])) for x in rows) / len(rows)), 4)

    def drop_slow25(s):
        xs = sorted(s)
        return st.fmean(xs[:len(xs) - len(xs)//4])

    def trim_mad(s):
        mid = st.median(s)
        mad = st.median(abs(v - mid) for v in s) * 1.4826
        kept = [v for v in s if v <= mid + 5.0 * mad] if mad else s
        return st.fmean(kept)

    summary = {
        "arm": arm_label,
        "boot_s": boot_s,
        "affinity": aff_after_boot,
        "cpu_pre": cpu_pre,
        "cpu_post": cpu_post,
        "telemetry_pre": tel_pre,
        "telemetry_post": tel_post,
        "step_drop25_gm_ms": gm_metric(drop_slow25, "samples"),
        "step_trim_mad_gm_ms": gm_metric(trim_mad, "samples"),
        "step_p10_gm_ms": gm_metric(lambda s: sorted(s)[len(s)//10], "samples"),
        "fwd_drop25_gm_ms": gm_metric(drop_slow25, "forward_ms"),
        "drf_drop25_gm_ms": gm_metric(drop_slow25, "drafter_ms"),
        "host_med_mean_ms": round(st.fmean(st.median(x["samples"]) - st.median(x["forward_ms"]) - st.median(x["drafter_ms"]) for x in rows), 4),
    }
    print("BOOT_SUMMARY " + json.dumps(summary), flush=True)
    with open(out_path, "w") as fh:
        json.dump({"summary": summary, "engine": rows}, fh)
    os._exit(0)

if __name__ == "__main__":
    main()
