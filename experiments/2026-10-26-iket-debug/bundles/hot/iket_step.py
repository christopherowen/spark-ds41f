"""DS4.1 decode steps end to end on one GPU, for IKET traces.

usage: iket_step.py off|on --out DIR [--iket DIR] [--ctas N] [--model DIR] [--drafts N]
                    [--tokens N]

vLLM's engine runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0) on DS4.1-Flash-1017's
config with dummy weights, at TP1, with the production recipe's kernels and graphs: the
TileLang attention, linear and MoE backends, DSpark with ``--drafts`` greedy drafts and
block verification, full and piecewise CUDA graphs, the packed BF16 head in FP32. One GPU
holds six of the 40 layers: two SWA layers (ratio 0), a CSA layer that is a KV and index
source and one that reuses it (ratio 2), the HCA candidate source and a layer that
re-indexes its candidates (ratio 1); the drafter's three layers follow. The Engram layers
are left out (``engram_layer_ids`` empty), and at TP1 nothing all-reduces.

A warm request runs first. Then a TileLang marker kernel (``iket_step_marker``) brackets
the measured request, a prefill and ``--tokens`` decode steps, so a trace's launches
between the two markers are that request's. ``on`` enables the IKET mode before anything
compiles; the worker's ``init_device`` finds it on (``VLLM_DS41_IKET``).

Writes ``result.json`` to ``--out``: the measured request's token ids, the wall time of
each request, the vLLM tree's IKET state and the config overrides.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL = ("/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
         "1bae0e1088993b9b82de08f59de5be5a34e42170")
TEXT_OVERRIDES = {
    "num_hidden_layers": 6,
    # Six target layers, then the drafter's three (ratio 0, as in the checkpoint).
    "compress_ratios": [0, 0, 2, 2, 1, 1, 0, 0, 0],
    "kv_source_layer_ids": [2, 4],
    "index_source_layer_ids": [2, 4],
    "candidate_source_layer_id": 4,
    "engram_layer_ids": [],
    "dspark_target_layer_ids": [3, 4, 5],
}
PROMPTS = [
    "Explain in detail how a sparse mixture-of-experts layer routes each token to its experts, "
    "and why the routing has to be balanced across the experts during training.",
    "Write a short story about a lighthouse keeper who discovers that the light has been "
    "signalling to ships that do not exist, and what she decides to do about it.",
]


def shrink(config):
    """The six-layer config, for the target and (a callable override reaches it too)
    the DSpark drafter."""
    text = config.get_text_config()
    for key, value in TEXT_OVERRIDES.items():
        setattr(text, key, value)
    return config


def same_dummy_weights():
    """vLLM's dummy loader seeds floating tensors but leaves integer ones (the packed FP4
    experts and their UE8M0 scales) as allocated, different in every process. Zero them,
    as it does on ROCm, so that every process loads the same weights and runs' tokens
    can be compared bit for bit."""
    import torch
    from vllm.model_executor.model_loader import dummy_loader, weight_utils

    seeded = weight_utils.initialize_single_dummy_weight

    def initialize(param, *args, **kwargs):
        if (param.device.type != "meta" and not torch.is_floating_point(param)
                and getattr(param, "dummy_weight_value", None) is None):
            param.zero_()
            return
        seeded(param, *args, **kwargs)

    dummy_loader.initialize_single_dummy_weight = initialize
    weight_utils.initialize_single_dummy_weight = initialize


def marker_kernel():
    import tilelang
    import tilelang.language as T

    @T.prim_func
    def iket_step_marker(A: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            A[0] = A[0] + 1

    return tilelang.compile(iket_step_marker, target="cuda")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("off", "on"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--iket")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--drafts", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--ctas", type=int, default=0, help="VLLM_DS41_IKET_CTAS")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch

    torch.cuda.init()
    if args.mode == "on":
        # The worker enables the mode again from these; the same arguments agree.
        os.environ["VLLM_DS41_IKET"] = args.iket
        os.environ["VLLM_DS41_IKET_CTAS"] = str(args.ctas)
        from vllm.models.deepseek_v4_1.tilelang import iket

        iket.enable(args.iket, args.ctas)
    marker = marker_kernel()
    count = torch.zeros(1, dtype=torch.int32, device="cuda")

    from vllm import LLM, SamplingParams

    same_dummy_weights()
    started = time.monotonic()
    llm = LLM(
        model=args.model,
        tokenizer_mode="deepseek_v41",
        load_format="dummy",
        dtype="bfloat16",
        seed=0,
        tensor_parallel_size=1,
        block_size=256,
        max_model_len=8192,
        # Two: the dead-row warmup launches for one request and for two.
        max_num_seqs=2,
        # A small profile run keeps the startup's share of a trace small; prefill
        # chunks are aligned, as CED's nonfinal-row skip requires (production: 8,192),
        # with room in the step for the decode rows beside one.
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=512,
        max_parallel_prefills=1,
        async_scheduling=True,
        kv_cache_memory_bytes=2 << 30,
        enable_prefix_caching=False,
        attention_backend="TILELANG",
        linear_backend="tilelang",
        moe_backend="tilelang",
        speculative_config={
            "method": "dspark", "num_speculative_tokens": args.drafts,
            "draft_tensor_parallel_size": 1, "attention_backend": "TILELANG",
            "draft_sample_method": "greedy", "rejection_sample_method": "block",
            "enable_adaptive_verification": True, "adaptive_verification_row_cost": 0.035,
        },
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"],
            "cudagraph_capture_sizes": [1, 2, 4, 6, 8, 12],
        },
        generation_config="vllm",
        hf_overrides=shrink,
        jit_monitor_mode="off",
        # One image, not none: a stubbed vision tower (limit 0) still declares its
        # TileLang kernels, and binding them fails on its meta weights.
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
    )
    boot = time.monotonic() - started
    print(f"engine ready in {boot:.0f} s", flush=True)
    greedy = SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True)
    timings = []
    started = time.monotonic()
    llm.generate([PROMPTS[0]], SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))
    timings.append(time.monotonic() - started)
    marker(count)
    torch.cuda.synchronize()
    started = time.monotonic()
    measured = llm.generate([PROMPTS[1]], greedy)[0]
    torch.cuda.synchronize()
    timings.append(time.monotonic() - started)
    marker(count)
    torch.cuda.synchronize()

    from vllm import envs

    result = {
        "mode": args.mode,
        "boot_s": round(boot, 1),
        "request_s": [round(t, 3) for t in timings],
        "prompt_tokens": len(measured.prompt_token_ids),
        "token_ids": list(measured.outputs[0].token_ids),
        "VLLM_DS41_IKET": envs.VLLM_DS41_IKET,
        "VLLM_DS41_IKET_CTAS": envs.VLLM_DS41_IKET_CTAS,
        "iket_modules": sorted(m for m in sys.modules if m.endswith(".iket")),
        "text_overrides": TEXT_OVERRIDES,
        "drafts": args.drafts,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
