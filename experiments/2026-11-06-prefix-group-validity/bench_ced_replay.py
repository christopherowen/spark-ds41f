"""Native SM121 TileLang 3-layer CED + SparseMLA replay verification & latency benchmark for PR #931."""
import json
import statistics
import sys
import torch

from tests.kernels.attention.test_deepseek_v41_ced_replay import (
    CED_WINDOW,
    DEVICE,
    HEAD_DIM,
    aligned_chunks,
    decoder_stack,
    model_state_stager,
    replayed,
)


def time_fn(fn, warmup=10, rounds=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples_ms = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        samples_ms.append(start_ev.elapsed_time(end_ev))
    samples_ms.sort()
    return {
        "median_ms": round(statistics.median(samples_ms), 4),
        "mean_ms": round(statistics.mean(samples_ms), 4),
        "min_ms": round(samples_ms[0], 4),
        "p95_ms": round(samples_ms[int(0.95 * (len(samples_ms) - 1))], 4),
        "samples": len(samples_ms),
    }


def main():
    arm = sys.argv[1]
    out_path = sys.argv[2]
    results = {
        "arm": arm,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "ordinary_chunking": {},
        "preemption_recovery": {},
    }

    # 1. Ordinary prompt/prefix-hit chunking latency & bit-identity
    prompt_end = 1024 + 128
    floor = prompt_end - CED_WINDOW
    whole = decoder_stack(prompt_end, [(0, prompt_end)], model_state_stager(0))
    ref_rows = replayed(whole, floor, prompt_end)
    for threshold, prefix in ((512, 0), (384, 0), (512, 448)):
        chunks = aligned_chunks(threshold, prefix, prompt_end)
        def run_ordinary(c=chunks, p=prefix):
            return decoder_stack(prompt_end, c, model_state_stager(p))
        out = run_ordinary()
        rows = replayed(out, floor, prompt_end)
        exact = bool(torch.equal(rows, ref_rows))
        stats = time_fn(run_ordinary)
        stats["exact_match"] = exact
        stats["chunks"] = chunks
        results["ordinary_chunking"][f"chunk_{threshold}_prefix_{prefix}"] = stats

    # 2. Preempted decode recovery latency & accuracy across generated_rows and chunk sizes
    original_end = 1024
    hit = (original_end - CED_WINDOW) // 256 * 256
    for generated_rows in (129, 257):
        total = original_end + generated_rows
        positions = torch.arange(total, device=DEVICE, dtype=torch.float32)[:, None]
        columns = torch.arange(HEAD_DIM, device=DEVICE, dtype=torch.float32)[None, :]
        encoder = ((positions % 17) - 8) / 512 + ((columns % 11) - 5) / 1024
        uninterrupted = decoder_stack(
            total,
            [(0, original_end), *((p, 1) for p in range(original_end, total))],
            model_state_stager(0, original_prompt_end=original_end, prefill_end=original_end),
            encoder=encoder,
        )
        expected = uninterrupted[total - 1]
        for resumed_chunk in (64, 512):
            chunks = aligned_chunks(resumed_chunk, hit, total)
            def run_recovery(t=total, c=chunks, e=encoder):
                return decoder_stack(
                    t,
                    c,
                    model_state_stager(hit, original_prompt_end=original_end, prefill_end=t),
                    encoder=e,
                )
            resumed = run_recovery()
            actual = resumed[total - 1]
            max_abs_diff = float((actual - expected).abs().max().item())
            exact = bool(torch.equal(actual, expected))
            stats = time_fn(run_recovery)
            stats.update(
                generated_rows=generated_rows,
                resumed_chunk=resumed_chunk,
                replayed_rows=total - hit,
                exact_match=exact,
                max_abs_diff=max_abs_diff,
                finite=bool(torch.isfinite(actual).all()),
            )
            results["preemption_recovery"][f"gen_{generated_rows}_chunk_{resumed_chunk}"] = stats

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
