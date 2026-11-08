"""Two-boot hardware benchmark and 1,000+ input bit-exactness verification
for certified two-pass target lm_head vocabulary screening on GB10 (sm_121a).
"""

import json
import os
import sys
import torch
from safetensors import safe_open


def make_layer(weight: torch.Tensor, use_carveout: bool):
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.online.packed_bf16 import (
        PackedBf16LmHeadMethod,
    )
    from vllm.v1.worker import display_carveout

    class _HeadLayer(torch.nn.Module):
        pass

    n, k = weight.shape
    display_carveout._carveouts.clear()
    display_carveout._integrity = None
    orig_vpe = display_carveout.VocabParallelEmbedding
    orig_req = display_carveout._REQUIRED
    try:
        display_carveout.VocabParallelEmbedding = _HeadLayer
        display_carveout._REQUIRED = use_carveout
        config = VllmConfig(kernel_config=KernelConfig(linear_backend="tilelang"))
        with set_current_vllm_config(config):
            method = PackedBf16LmHeadMethod()
            layer = _HeadLayer()
            with torch.device("cuda"):
                method.create_weights(layer, k, [n], k, n, torch.bfloat16)
            layer.weight.data.copy_(weight)
            method.process_weights_after_loading(layer)
            method.declare_layer_kernels(layer)
            return layer
    finally:
        display_carveout.VocabParallelEmbedding = orig_vpe
        display_carveout._REQUIRED = orig_req


def bench_fn(fn, warmup: int = 30, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / iters


def main():
    mode = sys.argv[1]  # "baseline" or "candidate"
    boot = int(sys.argv[2])
    use_carveout = os.environ.get("SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS", "0") == "1"

    from vllm.models.deepseek_v4_1.tilelang.vocab import (
        project_head,
        project_head_greedy_pairs,
    )
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import vocab_parallel_pairs

    with safe_open("/weights/head.safetensors", framework="pt", device="cuda") as f:
        full_w = f.get_tensor("head.weight")

    shapes = [
        ("tp4-ring", 32320, 32320, 0),
        ("tp3-triangle", 43136, 43094, 0),
    ]
    m_list = [1, 2, 4, 6, 8, 16, 32, 48]
    results = {
        "mode": mode,
        "boot": boot,
        "carveout": use_carveout,
        "shapes": {},
    }

    for shape_name, n, vocab, vocab_start in shapes:
        w = torch.zeros((n, 5120), dtype=torch.bfloat16, device="cuda")
        w[:vocab].copy_(full_w[vocab_start : vocab_start + vocab])

        layer = make_layer(w, use_carveout=use_carveout)

        # Warmup all row tiles
        for m in m_list:
            x_warm = torch.randn((m, 5120), dtype=torch.bfloat16, device="cuda")
            project_head_greedy_pairs(
                layer, x_warm, vocab=vocab, vocab_start=vocab_start, scratch_pairs=True
            )
        torch.cuda.synchronize()

        # Accuracy check on 1,000+ inputs (125 batches across M + 20 near-tie adversarial batches)
        mismatches = 0
        total_rows_checked = 0
        if boot == 1:
            gen = torch.Generator(device="cuda").manual_seed(20261108)
            for trial in range(130):
                m = m_list[trial % len(m_list)]
                if trial < 110:
                    x_test = (
                        torch.randn((m, 5120), generator=gen, device="cuda")
                        / (5120**0.5)
                    ).bfloat16()
                else:
                    # Near-tie adversarial inputs: blend two vocabulary rows with tiny perturbation
                    idx1 = torch.randint(0, vocab, (m,), generator=gen, device="cuda")
                    idx2 = torch.randint(0, vocab, (m,), generator=gen, device="cuda")
                    w1 = torch.nn.functional.normalize(w[idx1].float(), dim=1)
                    w2 = torch.nn.functional.normalize(w[idx2].float(), dim=1)
                    noise = (
                        torch.randn((m, 5120), generator=gen, device="cuda") * 1e-3
                    )
                    x_test = (0.5 * (w1 + w2) + noise).bfloat16()

                logits = project_head(layer, x_test)
                expected = vocab_parallel_pairs(logits, vocab, vocab_start, tp_size=1)
                actual = project_head_greedy_pairs(
                    layer,
                    x_test,
                    vocab=vocab,
                    vocab_start=vocab_start,
                    scratch_pairs=False,
                )
                if not torch.equal(
                    actual[:, :2].view(torch.int32), expected[:, :2].view(torch.int32)
                ):
                    mismatches += 1
                total_rows_checked += m

        latencies = {}
        active_cols_summary = {}
        gen_bench = torch.Generator(device="cuda").manual_seed(777)
        for m in m_list:
            x = (
                torch.randn((m, 5120), generator=gen_bench, device="cuda")
                / (5120**0.5)
            ).bfloat16()
            us = bench_fn(
                lambda: project_head_greedy_pairs(
                    layer, x, vocab=vocab, vocab_start=vocab_start, scratch_pairs=True
                )
            )
            latencies[str(m)] = round(us, 2)
            scratch = getattr(layer, "_tilelang_vocab_greedy_scratch", None)
            if scratch is not None and len(scratch) >= 9:
                active_cols_summary[str(m)] = int(scratch[8].item())

        results["shapes"][shape_name] = {
            "n": n,
            "vocab": vocab,
            "mismatches": mismatches,
            "total_rows_checked": total_rows_checked,
            "latencies_us": latencies,
            "active_cols": active_cols_summary,
        }
        del layer
        del w
        torch.cuda.empty_cache()

    print("BENCH_JSON:" + json.dumps(results))


if __name__ == "__main__":
    main()
