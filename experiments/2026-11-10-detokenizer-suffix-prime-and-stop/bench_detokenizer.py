#!/usr/bin/env python3
"""Benchmark FastIncrementalDetokenizer prompt-suffix priming across 2 container boots."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import (
    FastIncrementalDetokenizer,
    prompt_suffix_to_prime,
)

DEFAULT_MODEL_PATH = (
    "/home/chris/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V4.1-Flash-1017/"
    "snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
)


def _make_request(
    prompt_token_ids: list[int],
    stop: list[str] | None = None,
    include_stop_str_in_output: bool = False,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="bench-req",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(
            stop=stop,
            include_stop_str_in_output=include_stop_str_in_output,
        ),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _baseline_init_and_steps(
    hf_tok,
    prompt_ids: list[int],
    gen_ids: list[int],
) -> tuple[float, str]:
    req = _make_request(prompt_ids)
    with patch(
        "vllm.v1.engine.detokenizer.prompt_suffix_to_prime",
        side_effect=lambda _tk, ids, _skip: ids,
    ):
        t0 = time.perf_counter()
        detok = FastIncrementalDetokenizer(hf_tok, req)
        detok.update(gen_ids, stop_terminated=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, detok.output_text


def _candidate_init_and_steps(
    hf_tok,
    prompt_ids: list[int],
    gen_ids: list[int],
) -> tuple[float, str, int]:
    req = _make_request(prompt_ids)
    t0 = time.perf_counter()
    detok = FastIncrementalDetokenizer(hf_tok, req)
    detok.update(gen_ids, stop_terminated=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    primed = prompt_suffix_to_prime(hf_tok._tokenizer, prompt_ids, True)
    return elapsed_ms, detok.output_text, len(primed or [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--boot-label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    hf_tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tk = hf_tok._tokenizer

    base_block = (
        "def analyze_trace(step_id: int, payload: dict[str, float]) -> str:\n"
        "    # DeepSeek V4.1 Flash Verification Trace — 🚀 測試 — पुरुषाणां\n"
        "    return f\"{step_id}: {payload.get(\x27latency_ms\x27, 0.0):.3f} ms\"\n"
    )
    block_ids = tk.encode(base_block, add_special_tokens=False).ids
    gen_sample_ids = tk.encode(
        "    result = analyze_trace(42, {\"latency_ms\": 1.234})\n",
        add_special_tokens=False,
    ).ids[:8]

    max_len = 524288
    repeats = (max_len // len(block_ids)) + 2
    full_pool = (block_ids * repeats)[:max_len]

    prompt_lengths = [4096, 16384, 65536, 131072, 262144, 524288]
    records: list[dict] = []

    for n_tokens in prompt_lengths:
        prompt_ids = full_pool[:n_tokens]
        trials = 15 if n_tokens <= 131072 else 8
        _baseline_init_and_steps(hf_tok, prompt_ids, gen_sample_ids)
        _candidate_init_and_steps(hf_tok, prompt_ids, gen_sample_ids)

        base_times: list[float] = []
        cand_times: list[float] = []
        primed_len = 0
        for _ in range(trials):
            b_ms, b_out = _baseline_init_and_steps(hf_tok, prompt_ids, gen_sample_ids)
            c_ms, c_out, primed_len = _candidate_init_and_steps(
                hf_tok, prompt_ids, gen_sample_ids
            )
            assert b_out == c_out, f"Output mismatch at {n_tokens}: {b_out!r} != {c_out!r}"
            base_times.append(b_ms)
            cand_times.append(c_ms)

        b_med = statistics.median(base_times)
        c_med = statistics.median(cand_times)
        rec = {
            "boot": args.boot_label,
            "workload": "prompt_suffix_prime_first_step",
            "prompt_tokens": n_tokens,
            "primed_suffix_tokens": primed_len,
            "trials": trials,
            "baseline_median_ms": round(b_med, 4),
            "candidate_median_ms": round(c_med, 4),
            "speedup_x": round(b_med / c_med, 2),
            "saved_ms_c1": round(b_med - c_med, 4),
            "saved_ms_c8": round((b_med - c_med) * 8.0, 4),
            "bit_exact": True,
        }
        print(json.dumps(rec))
        records.append(rec)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
