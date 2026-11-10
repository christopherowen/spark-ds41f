# Experiment: Bounded UTF-8-Aligned Prompt-Suffix Priming in `FastIncrementalDetokenizer`

- **Date**: 2026-11-10
- **Base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`, vLLM patch head `2756478ba1fd70aa4b700e7e467c46960c9888fb`)
- **Candidate patch**: `vllm/0762-perf-detokenizer-suffix-prime-and-stop.patch` (patch head `2a93ee083809b329ff283a22f3c422c6a98a09d0`, tree `993678ca84ce66efbd27756a813d82de23b90755`)
- **Causal variable**: Prime `tokenizers.decoders.DecodeStream` in `FastIncrementalDetokenizer.__init__` (`vllm/v1/engine/detokenizer.py`) with a bounded, UTF-8-aligned prompt token suffix (`prompt_suffix_to_prime()`, 16..67 tokens) instead of passing the entire `request.prompt_token_ids` list.

## Problem & Mechanism

In `vllm/v1/engine/detokenizer.py`, `FastIncrementalDetokenizer.__init__` previously initialized HuggingFace `tokenizers.decoders.DecodeStream` with `ids=request.prompt_token_ids`. Unlike `SlowIncrementalDetokenizer` (which slices only the last 7 prompt tokens via `INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5`), `DecodeStream` copies the full prompt token vector and decodes all `ids` on the first generated token to compute its prefix string. On GB10 Grace CPU with the `DeepSeek-V4.1-Flash-1017` tokenizer, this incurs $O(\text{len(prompt)})$ CPU string allocation and decoding on the `EngineCore` output thread (`11.6..11.7 ms` at `64K`, `26.3..27.1 ms` at `128K`, `55.8..56.2 ms` at `262K`, and `111.8..113.9 ms` at `524K` tokens per request, blocking up to `894..911 ms` across an 8-request long-context batch), even though BPE and byte-fallback merge context only depends on the trailing tokens of the prompt.

`prompt_suffix_to_prime(tokenizer, prompt_token_ids, skip_special_tokens)`:
1. Returns `prompt_token_ids` unchanged when `len(prompt_token_ids) <= 16` (`PROMPT_SUFFIX_PRIME_WINDOW = 16`).
2. Checks window sizes `w in (16, 32, 64)` (`PROMPT_SUFFIX_MAX_WINDOW = 64`) with left-boundary alignment `adjust in range(4)` (`cand_w = w + adjust`):
   - Decodes `suffix = prompt_token_ids[-cand_w:]` with `skip_special_tokens`.
   - If `decoded.endswith("\ufffd")`, the prompt itself ends mid-UTF-8 codepoint; falls back immediately to `prompt_token_ids` so `DecodeStream` preserves its full-prefix recovery behavior.
   - If `decoded and not decoded.startswith("\ufffd")`, returns `suffix`.
3. Falls back to `prompt_token_ids` if no bounded window up to `64 + 3` tokens produces a non-empty, `U+FFFD`-free prefix.

## Screened & Rejected Secondary Idea (Recorded Run)

Before isolating `prompt_suffix_to_prime()` as the single causal variable, we also screened evaluating `check_stop_strings()` per token inside `BaseIncrementalDetokenizer.update()` when `len(new_token_ids) > 1` (`receipts/bench-detokenizer-per-token-stop-rejected-boot1.jsonl`, `receipts/bench-detokenizer-per-token-stop-rejected-boot2.jsonl`). Because `1107`'s `check_stop_strings()` (`vllm/v1/engine/detokenizer.py:334-362`) already selects the earliest-completing stop string (`best_end = sys.maxsize`, `if end < best_end:`) in a single pass at the end of `update()`, checking per token increased 6-token stop-truncation latency from `3.480..3.488 µs` to `5.088..5.104 µs` (`+45.87%..+46.65%`) and would run `K` times per step on all non-terminating speculative steps. Per-token stop checking in `BaseIncrementalDetokenizer.update()` was therefore rejected and dropped from `0762`.

## Exact Verification & Benchmark Commands

```bash
VLLM_SRC=/home/chris/projects/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-09ab6cb9/src/vllm
EXP_DIR=/home/chris/projects/spark-ds41f/experiments/2026-11-10-detokenizer-suffix-prime-and-stop

for boot in boot1 boot2; do
  flock /tmp/spark-lab-gpu.lock docker run --rm --gpus all --entrypoint bash \
    -v "$VLLM_SRC/vllm/v1/engine/detokenizer.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/engine/detokenizer.py:ro" \
    -v "$VLLM_SRC/tests/detokenizer:/opt/spark-ds41f/candidate/vllm/tests/detokenizer:ro" \
    -v "$EXP_DIR:/workspace/exp" \
    -v "/home/chris/.cache:/home/chris/.cache:ro" \
    spark-ds41f:1107 -lc "
      set -e
      pytest --noconftest -o addopts=\"\" \
        /opt/spark-ds41f/candidate/vllm/tests/detokenizer/test_check_stop_strings.py \
        /opt/spark-ds41f/candidate/vllm/tests/detokenizer/test_stop_string_while_stop_model_terminates.py \
        | tee /workspace/exp/receipts/pytest-${boot}.txt
      python3 /workspace/exp/bench_detokenizer.py \
        --boot-label ${boot} \
        --output /workspace/exp/receipts/bench-detokenizer-${boot}.jsonl
    "
done
```

## Two-Boot Results (`receipts/bench-detokenizer-boot1.jsonl`, `receipts/bench-detokenizer-boot2.jsonl`)

Workload: `DeepSeek-V4.1-Flash-1017` tokenizer (`AutoTokenizer.from_pretrained`), mixed code/UTF-8/CJK/Devanagari/emoji prompt of `4,096..524,288` tokens + first 8 generated tokens via `FastIncrementalDetokenizer`.

| Prompt tokens | Primed suffix tokens | Baseline `1107` (`boot1` / `boot2`) | Candidate `0762` (`boot1` / `boot2`) | Speedup (`boot1` / `boot2`) | Saved at `c1` (`boot1` / `boot2`) | Saved at `c8` (`boot1` / `boot2`) | Bit-exact output |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| `4,096` | `16` | `0.7216 ms` / `0.7143 ms` | `0.0132 ms` / `0.0129 ms` | `54.53x` / `55.46x` | `0.7083 ms` / `0.7014 ms` | `5.67 ms` / `5.61 ms` | `True` |
| `16,384` | `16` | `2.7825 ms` / `2.7394 ms` | `0.0153 ms` / `0.0149 ms` | `181.53x` / `183.31x` | `2.7671 ms` / `2.7244 ms` | `22.14 ms` / `21.80 ms` | `True` |
| `65,536` | `16` | `11.6671 ms` / `11.6105 ms` | `0.0232 ms` / `0.0257 ms` | `502.20x` / `451.84x` | `11.6439 ms` / `11.5848 ms` | `93.15 ms` / `92.68 ms` | `True` |
| `131,072` | `16` | `26.3368 ms` / `27.0703 ms` | `0.0275 ms` / `0.0301 ms` | `958.12x` / `898.48x` | `26.3093 ms` / `27.0402 ms` | `210.47 ms` / `216.32 ms` | `True` |
| `262,144` | `16` | `55.7602 ms` / `56.1643 ms` | `0.0309 ms` / `0.0322 ms` | `1,806.61x` / `1,745.97x` | `55.7293 ms` / `56.1322 ms` | `445.83 ms` / `449.06 ms` | `True` |
| `524,288` | `16` | `111.7805 ms` / `113.8830 ms` | `0.0352 ms` / `0.0352 ms` | `3,176.93x` / `3,236.78x` | `111.7453 ms` / `113.8478 ms` | `893.96 ms` / `910.78 ms` | `True` |

Unit tests (`receipts/pytest-boot1.txt`, `receipts/pytest-boot2.txt`): `12 passed` on both boots (`tests/detokenizer/test_check_stop_strings.py`, `tests/detokenizer/test_stop_string_while_stop_model_terminates.py`).

## Memory Observations & Conclusion

- **Memory**: Avoids copying up to `524,288` prompt token IDs into Rust `DecodeStream::ids` and avoids allocating a decoded multi-megabyte full-prompt string on the first generated token of each request (`~2..4 MiB` transient heap saved per `524K` request). Zero GPU/KV memory impact.
- **Conclusion**: `prompt_suffix_to_prime()` reduces `FastIncrementalDetokenizer` initialization and first-token detokenization cost from $O(\text{len(prompt)})$ (`11.6 ms` at `64K` up to `113.9 ms` at `524K`) to a flat `0.013..0.035 ms` (`54x..3,237x` speedup) while preserving 100% step-for-step bit-exact output.
