# Structured-Output Grammar on `/v1/completions`, Size-Classed Prefill Lanes, and Disk Engram Row-Cache Evaluation

## Summary

This experiment evaluates three runtime and scheduler areas on top of `spark-ds41f-1107` (`spark-ds41f:1107`, node `sp6`):

1. **Structured-output grammar enforcement on `/v1/completions` under `--reasoning-parser deepseek_v41` (`0762`)**:
   - Fixes a bug in `StructuredOutputManager._get_constraint_start` where `/v1/completions` and raw `LLM.generate()` requests with `structured_outputs` had their grammar bitmask and speculative draft-token trimming bypassed when `--reasoning-parser deepseek_v41` (or `deepseek_v3`) was enabled and `enable_in_reasoning=False`.
   - Also aligns `DeepseekV41ModelState.prepare_attn` with `ModelState.prepare_attn` and `UBatchRunner` by accepting `ubatch_idx: int = 0`.
2. **Size-classed prefill lanes, multi-lane livelock guards, and aligned-chunk lane rotation (`0763`)**:
   - Fixes an aligned-chunk multi-lane livelock in `Scheduler.schedule()` where enabling `--max-parallel-prefills > 1` with `align_prefill_chunks=True` split the step token budget across lanes (`token_budget // active_lanes < aligned_chunk`), causing `num_new_tokens < aligned_chunk` to skip every selected long prefill without releasing its lane.
   - Adds size-classed prefill lanes (`--prefill-long-threshold` and `--max-long-prefills`, defaulting to `1` when `prefill_long_threshold > 0`) and multi-lane livelock guards (excluding `WAITING_FOR_FSM`, `WAITING_FOR_REMOTE_KVS`, and `WAITING_FOR_STREAMING_REQ` from lane selection, releasing lanes on KV allocation failure or Mamba zero-token split, keeping lane shares above the Mamba block quantum, and allowing preemption in the post-waiting interleave pass when `not num_scheduled_tokens`).
   - Compares multi-lane size-classed scheduling against the promoted single-lane `--prefill-yield short` scheduler (`0545` + `0751`) under concurrent 64K, 128K, and 256K long prefills with both cold and prefix-cached short interactive arrivals.
3. **Disk Engram row-cache metrics and advisory prefetch evaluation**:
   - Evaluates whether a Python-thread advisory row-prefetch helper is needed for disk-backed Engram embedding layers, confirming that the native C/TileLang `EngramDiskRows` engine (`0648`, `0650`, `0659`–`0663`, `0670`, `0750`) already provides zero-GIL ahead-of-step C hashing (`HostEngramHash`), `O_DIRECT` + `io_uring` slot DMA, and a 4-way set-associative user-space row cache with built-in hit/miss/read telemetry (`EngramDiskRows.stats()`).

## Base & Candidate Identity

- **Repository base commit**: `8e5ea208f3d22bc69df91ace94d6975932c858a8` (`main`)
- **Promoted baseline**: `2026-11-07-spark-ds41f-1107` (`spark-ds41f:1107`, digest `sha256:3f32a962ce04`)
- **vLLM base revision**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`
- **vLLM 1107 patch head (641 patches)**: `2756478b60b90700b9918b457aeb38657da637b5` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **vLLM candidate patch head (643 patches, `+0762`, `+0763`)**: `94d7d913036b141033fbd94aa586bbebb5d7c45b` (tree `ab95e712feae4036debe05f52ce0b78fab973439`)
- **Evaluation node**: `sp6` (`spark-0014`, NVIDIA GB10)

## Causal Variables Tested

1. **`0762-fix-structured-output-enforce-completions-grammar-un.patch`**:
   - In `vllm/v1/structured_output/__init__.py` (`StructuredOutputManager._get_constraint_start`), return `0` immediately when `structured_req.reasoning_ended is None and not structured_req.reasoning_parser_kwargs and structured_req.reasoner is None`.
   - Why this happens: `DeepSeekV41ParserReasoningAdapter` (`DeepSeekV4Parser`) defaults `thinking = True` (`initial_state = ParserState.REASONING`, `wait_for_reasoning = True`) when `chat_template_kwargs` is absent or empty. While `/v1/chat/completions` and `/v1/responses` pass `reasoning_parser_kwargs={"chat_template_kwargs": ...}` (and `/v1/chat/completions` sets `reasoning_ended`), `/v1/completions` and raw `LLM.generate()` leave `reasoning_ended=None`, `reasoning_parser_kwargs=None`, and `reasoner=None`. Previously, `_get_constraint_start` lazily instantiated a request-local `reasoner` with `{}` kwargs, saw `reasoner.is_reasoning_end(prompt_token_ids) == False`, latched `structured_req.reasoning_ended = False`, and returned `len(spec_tokens) + 1` on every step—leaving all bitmask rows unconstrained (`-1`) and keeping invalid speculative draft tokens.
   - In `vllm/models/deepseek_v4_1/nvidia/model_state.py`, add `ubatch_idx: int = 0` to `DeepseekV41ModelState.prepare_attn`.
2. **`0763-feat-sched-size-classed-prefill-lanes-and-aligned-ch.patch`**:
   - In `vllm/v1/core/sched/scheduler.py`, exempt aligned chunks (`aligned_chunk is not None` in the running pass and `self.scheduler_config.align_prefill_chunks` in the waiting pass) from per-step lane-share budget splitting so parallel prefill lanes rotate whole aligned chunks across steps instead of livelocking.
   - In `vllm/config/scheduler.py`, `vllm/engine/arg_utils.py`, `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/sched/prefill_interleave.py`, `vllm/v1/core/sched/scheduler.py`, and `vllm/v1/request.py`, add `--prefill-long-threshold`, `--max-long-prefills`, prefix-cache-aware remaining-token classification (`qos_cached_tokens_hint`), 8-step starvation aging (`LANE_AGING_STEPS = 8`), and multi-lane livelock guards.

## Exact Commands

### 1. 2-Boot Unit Test Verification (`196 passed` per boot)

```bash
for BOOT in 1 2; do
  docker run --rm --gpus=all \
    -e HF_HUB_OFFLINE=1 \
    -v /tmp/hf-test-cache:/root/.cache/huggingface:ro \
    -v /tmp/work-lil-vllm-prs/.work/upstreams/vllm:/mnt/vllm-src:ro \
    --entrypoint bash spark-ds41f:1107 -c '
set -e
cd /opt/spark-ds41f/candidate/vllm
for f in \
  vllm/v1/structured_output/__init__.py \
  vllm/models/deepseek_v4_1/nvidia/model_state.py \
  vllm/config/scheduler.py \
  vllm/engine/arg_utils.py \
  vllm/v1/core/kv_cache_manager.py \
  vllm/v1/core/sched/prefill_interleave.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/request.py \
  tests/v1/structured_output/test_structured_output_manager.py \
  tests/v1/core/utils.py \
  tests/v1/core/test_prefill_lane_livelock.py \
  tests/v1/core/test_prefill_qos_lanes.py \
  tests/v1/core/test_aligned_prefill_lanes.py
do
  cp /mnt/vllm-src/$f $f
done
PYTHONPATH=/tmp:$PYTHONPATH pytest -o addopts="" -p tblib_stub -q \
  tests/v1/structured_output/test_structured_output_manager.py \
  tests/v1/spec_decode/test_mtp_structured_output.py \
  tests/v1/core/test_prefill_lane_livelock.py \
  tests/v1/core/test_prefill_qos_lanes.py \
  tests/v1/core/test_aligned_prefill_lanes.py \
  tests/v1/core/test_prefill_yield.py \
  tests/v1/core/test_aligned_prefill_chunks.py \
  tests/v1/core/test_prefill_compute_share_scheduler.py
'
done
```

### 2. 2-Boot GPU, Scheduler, and Engram Benchmark Suite

```bash
for BOOT in boot1 boot2; do
  docker run --rm --gpus=all \
    --security-opt=seccomp=unconfined \
    --cap-add=IPC_LOCK \
    --ulimit=memlock=-1:-1 \
    -e HF_HUB_OFFLINE=1 \
    -v /tmp/hf-test-cache:/root/.cache/huggingface:ro \
    -v /home/chris/.cache/huggingface/hub:/mnt/ds41-hub:ro \
    -v /tmp:/mnt/host-tmp:rw \
    -v /tmp/work-lil-vllm-prs/.work/upstreams/vllm:/mnt/vllm-src:ro \
    --entrypoint bash spark-ds41f:1107 -c "
set -e
cd /opt/spark-ds41f/candidate/vllm
for f in \
  vllm/v1/structured_output/__init__.py \
  vllm/models/deepseek_v4_1/nvidia/model_state.py \
  vllm/config/scheduler.py \
  vllm/engine/arg_utils.py \
  vllm/v1/core/kv_cache_manager.py \
  vllm/v1/core/sched/prefill_interleave.py \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/request.py \
  tests/v1/core/utils.py
do
  cp /mnt/vllm-src/\$f \$f
done
python3 /mnt/host-tmp/bench_suite.py $BOOT
"
done
```

## Results

### Part 1: Structured-Output Grammar Enforcement Under `deepseek_v41` Reasoning Parser (2 Boots)

Workload: `DeepSeek-V4.1-Flash-1017` tokenizer (`vocab_size = 129,280`), JSON schema `{"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": false}`, `num_speculative_tokens = 5` with draft window `[open_brace_id, invalid_thinking_id, quote_id, open_brace_id, open_brace_id]` (and `[invalid_id, think_end_id, open_brace_id, invalid_id, quote_id]` for mid-draft `</think>`), plus GPU `xgrammar.apply_token_bitmask_inplace` verification on `(rows, 129280)` FP32 logits where `logits[:, invalid_id] = 20.0` and `logits[:, open_brace_id] = 10.0`.

| Mode | `_get_constraint_start` | Drafts -> Trimmed | Bitmask Row Pattern (`U`=unconstrained, `C`=constrained) | `reasoner` Instantiated | Invalid Token Masked (`-inf`) & `{` Sampled | CPU Step (`trim + bitmask`) Boot 1 / Boot 2 ($\mu\text{s}$) | GPU Bitmask Kernel Boot 1 / Boot 2 ($\mu\text{s}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `/v1/completions` (unpatched `1107`) | `6` | `5 -> 5` | `UUUUUU` | `True` | `False` / `False` | `54,941.64` / `53,847.65` (first step with parser init); `14.11` / `14.14` (cached) | `12.45` / `12.13` |
| `/v1/completions` (patched `0762`) | `0` | `5 -> 2` | `CCU` | `False` | `True` / `True` | `10.74` / `10.63` | `12.42` / `12.58` |
| `/v1/chat/completions` (thinking active) | `6` | `5 -> 5` | `UUUUUU` | `True` | `False` / `False` | `14.11` / `14.14` | `12.25` / `11.67` |
| `/v1/chat/completions` (`</think>` at draft idx 1) | `2` | `5 -> 4` | `UUCCU` | `True` | `False` (row 0 in thinking) | `14.67` / `14.60` | `12.55` / `11.59` |

- **Finding**: Unpatched `1107` silently disables structured-output grammar enforcement on `/v1/completions` when `--reasoning-parser deepseek_v41` is active (`constraint_start = 6`, all bitmask rows `UUUUUU`, invalid token `"thinking"` sampled instead of `"{"`) and incurs a ~54 ms host stall on the first step to instantiate a request-local `DeepSeekV41ParserReasoningAdapter`. Patch `0762` restores `constraint_start = 0` from token 0 (`CCU` bitmask, invalid drafts trimmed, `"{"` sampled) and skips unnecessary reasoner construction on `/v1/completions`.

### Part 2: Concurrent Long Prefill (64K / 128K / 256K) + Short Interactive Arrival Scheduler Benchmark (2 Boots)

Workload: Promoted `tp4-ring` step geometry (`max_num_batched_tokens = 16,384`, `max_num_seqs = 48`, `num_speculative_tokens = 1` -> `96` decode rows, `long_prefill_token_threshold = 16,288`, `block_size = 64`, `enable_prefix_caching = True`).
- 8 active decode streams (`decode_0..decode_7`) are running in steady-state decode.
- `long1` (`64K = 65,536`, `128K = 131,072`, or `256K = 262,144` tokens) is admitted and completes its first `16,288`-token chunk.
- While `long1` is mid-prefill, three additional long prefills (`long2`, `long3`, `long4` of the same length) arrive in `waiting`, followed by:
  - `short_cold`: a `1,000`-token cold interactive request;
  - `short_cached`: a `65,000`-token request whose first `64,000` tokens hit the prefix cache (`1,000` uncached tokens remaining).

| Context Length | Scheduler Arm | `short_cold` Wait / Finish Steps | `short_cold` TTFT (s) | `short_cached` Wait / Finish Steps | `short_cached` TTFT (s) | `long1` Chunks (Aligned?) | `long1` Done Step / TTFT (s) | Scheduler CPU Boot 1 / Boot 2 ($\mu\text{s}$/step) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64K** | 1-lane FCFS (`--max-parallel-prefills 1 --prefill-yield off`) | `19 / 19` | `8.122` | `19 / 19` | `8.122` | `5` (`True`) | `4 / 1.629` | `417.9 / 404.1` |
| **64K** | **1-lane Prefill Yield (`--max-parallel-prefills 1 --prefill-yield short`, promoted `1107`)** | **`1 / 1`** | **`0.192`** | **`1 / 1`** | **`0.192`** | **`5` (`True`)** | **`12 / 2.242`** | **`264.4 / 272.2`** |
| **64K** | 4-lane Aligned Unpatched (`--max-parallel-prefills 4, align=True`) | Livelock ($\infty$) | Livelock | Livelock ($\infty$) | Livelock | Stalls after chunk 1 | Livelock | — |
| **64K** | 4-lane Aligned Size-Blind (`--max-parallel-prefills 4, align=True`, patched `0763`) | `19 / 19` | `8.026` | `19 / 19` | `8.026` | `5` (`True`) | `12 / 5.510` | `459.6 / 457.3` |
| **64K** | 4-lane Aligned Size-Classed QoS (`--max-parallel-prefills 4 --prefill-long-threshold 16288 --max-long-prefills 1, align=True`, patched `0763`) | `19 / 19` | `8.122` | `19 / 19` | `8.122` | `5` (`True`) | `4 / 1.629` | `749.6 / 737.3` |
| **64K** | 4-lane Unaligned Size-Blind (`--max-parallel-prefills 4, align=False`) | `2 / 2` | `0.843` | `1 / 1` | `0.409` | `14` (`False`) | `14 / 6.720` | `350.2 / 345.0` |
| **64K** | 4-lane Unaligned Size-Classed QoS (`--max-parallel-prefills 4 --prefill-long-threshold 16288 --max-long-prefills 1, align=False`, patched `0763`) | `1 / 1` | `0.274` | `1 / 1` | `0.274` | `7` (`False`) | `6 / 2.592` | `413.6 / 445.6` |
| **128K** | 1-lane FCFS (`--max-parallel-prefills 1 --prefill-yield off`) | `35 / 35` | `16.962` | `35 / 35` | `16.962` | `9` (`True`) | `8 / 3.705` | `412.7 / 413.8` |
| **128K** | **1-lane Prefill Yield (`--max-parallel-prefills 1 --prefill-yield short`, promoted `1107`)** | **`1 / 1`** | **`0.192`** | **`1 / 1`** | **`0.192`** | **`9` (`True`)** | **`24 / 4.773`** | **`453.8 / 456.3`** |
| **128K** | 4-lane Aligned Size-Blind (`--max-parallel-prefills 4, align=True`, patched `0763`) | `35 / 35` | `16.512` | `35 / 35` | `16.512` | `9` (`True`) | `24 / 11.720` | `732.0 / 714.7` |
| **128K** | 4-lane Aligned Size-Classed QoS (`--max-parallel-prefills 4 --prefill-long-threshold 16288 --max-long-prefills 1, align=True`, patched `0763`) | `35 / 35` | `16.962` | `35 / 35` | `16.962` | `9` (`True`) | `8 / 3.705` | `633.2 / 631.6` |
| **128K** | 4-lane Unaligned Size-Blind (`--max-parallel-prefills 4, align=False`) | `2 / 2` | `0.843` | `1 / 1` | `0.409` | `30` (`False`) | `30 / 15.007` | `333.9 / 326.3` |
| **128K** | 4-lane Unaligned Size-Classed QoS (`--max-parallel-prefills 4 --prefill-long-threshold 16288 --max-long-prefills 1, align=False`, patched `0763`) | `1 / 1` | `0.274` | `1 / 1` | `0.274` | `11` (`False`) | `10 / 4.713` | `367.9 / 366.2` |
| **256K** | 1-lane FCFS (`--max-parallel-prefills 1 --prefill-yield off`) | `67 / 67` | `37.327` | `67 / 67` | `37.327` | `17` (`True`) | `16 / 8.243` | `409.0 / 407.6` |
| **256K** | **1-lane Prefill Yield (`--max-parallel-prefills 1 --prefill-yield short`, promoted `1107`)** | **`1 / 1`** | **`0.192`** | **`1 / 1`** | **`0.192`** | **`17` (`True`)** | **`47 / 10.162`** | **`537.9 / 532.4`** |
| **256K** | 4-lane Aligned Size-Blind (`--max-parallel-prefills 4, align=True`, patched `0763`) | `67 / 67` | `35.572` | `67 / 67` | `35.572` | `17` (`True`) | `38 / 19.524` | `830.2 / 821.8` |
| **256K** | 4-lane Aligned Size-Classed QoS (`--max-parallel-prefills 4 --prefill-long-threshold 16288 --max-long-prefills 1, align=True`, patched `0763`) | `67 / 67` | `37.327` | `67 / 67` | `37.327` | `17` (`True`) | `16 / 8.243` | `658.5 / 649.2` |
| **256K** | 4-lane Unaligned Size-Blind (`--max-parallel-prefills 4, align=False`) | `2 / 2` | `0.843` | `1 / 1` | `0.409` | `62` (`False`) | `62 / 33.116` | `571.6 / 568.3` |
| **256K** | 4-lane Unaligned Size-Classed QoS (`--max-parallel-prefills 4 --prefill-long-threshold 16288 --max-long-prefills 1, align=False`, patched `0763`) | `1 / 1` | `0.274` | `1 / 1` | `0.274` | `19` (`False`) | `18 / 9.339` | `343.0 / 332.4` |

- **Findings on Scheduler Behavior**:
  1. **How promoted `1107` (`--max-parallel-prefills 1 --prefill-yield short`, `0545` + `0751`) handles short arrivals during a long prefill**:
     - `0545` does **not** merely yield to already-running decode requests. `_short_prefill_waiting` (`scheduler.py:1192-1208`) scans up to 16 waiting requests, queries `self.kv_cache_manager.get_computed_blocks(request)` to account for prefix-cache hits, and sets `yield_long = True` whenever any waiting request finishes its remaining uncached prefill in one aligned chunk (`<= 16,288` tokens).
     - When `yield_long` is active, the running long prefill (`long1`) defers its step (`deferred_long.append(request)`) and any waiting long prefills (`long2..long4`) at the head of `waiting` are skipped (`step_skipped_waiting.prepend_request(request)`). Both `short_cold` (1,000 tokens) and `short_cached` (65,000 tokens with 64,000 prefix-cached) are admitted and finish in **1 step (`0.192 s` TTFT across 64K, 128K, and 256K—a `42x` to `194x` reduction vs FCFS)** while `long1` retains **100% aligned `16,288`-token CED chunk boundaries** (`5`, `9`, and `17` chunks).
  2. **Why `0763` is needed when `--max-parallel-prefills > 1` is configured**:
     - Unpatched `1107` livelocks when `align_prefill_chunks=True` and `max_parallel_prefills > 1` as soon as two long prefills occupy lanes (`16,368 // 2 = 8,184 < 16,288`). Exempting aligned chunks from per-step lane-share splitting allows parallel lanes to rotate whole aligned chunks without livelock.
     - Under unaligned chunking (`align_prefill_chunks=False`), size-classed lanes (`--prefill-long-threshold 16288 --max-long-prefills 1`) cut `long1`'s chunk count from `14 / 30 / 62` down to `7 / 11 / 19` and reduce `short_cold` admission from `2` steps (`0.843 s`) to `1` step (`0.274 s`).
     - Under aligned chunking (`align_prefill_chunks=True`, DeepSeek V4.1), because each aligned chunk (`16,288` tokens) consumes the full step prefill budget and `schedule_running_requests` executes before `self.waiting`, reserving a lane for the running `long1` on every step with `--prefill-yield off` leaves only `80` tokens (`< 1,000`) when `self.waiting` is reached (`aligned_chunk is not None and num_new_tokens < aligned_chunk -> break`). Thus `--max-parallel-prefills 1 --prefill-yield short` remains the superior runtime configuration for DeepSeek V4.1 serving.

### Part 3: Disk Engram Row-Cache Metrics & Advisory Prefetch Evaluation (2 Boots)

Workload: `EngramDiskRows` (`vllm/models/deepseek_v4_1/engram_rows/rows.py` + `_reader.c`) over a 4,096-row packed table (`264` bytes/row, `15` rows per 4 KiB page) with a 1,024-row user-space 4-way set-associative cache (`cache_rows = 1024`), 16 tokens $\times$ 24 hash columns (`384` row lookups/step), comparing the ahead-of-step `early` host-staging path (`early=hot_ids_np`) against the GPU-stream decode path (`early=None`) over 200 timed steps after warmup.

| Metric | Boot 1 | Boot 2 |
| :--- | :---: | :---: |
| Ahead-of-step `early` prefill `stage + gather` ($\mu\text{s}$/step, 384 lookups) | `86.89` | `94.11` |
| GPU-stream decode `stage + gather` ($\mu\text{s}$/step, 384 lookups) | `143.78` | `154.17` |
| Total lookups / Early lookups | `157,440` / `80,640` | `157,440` / `80,640` |
| User-space C row-cache hits / misses | `157,440` / `384` (`99.76%` hit rate) | `157,440` / `384` (`99.76%` hit rate) |
| NVMe `O_DIRECT` read calls / unique 4 KiB blocks / bytes | `2` / `26` / `106,496 B` | `2` / `26` / `106,496 B` |
| Total C worker NVMe read time (`read_seconds` / `early_read_seconds`) | `0.695 ms` / `0.499 ms` | `0.511 ms` / `0.633 ms` |

- **Finding**: `1107` retired the legacy `b12x` Engram path (`0648`, `0650`) in favor of `EngramDiskRows` (`_reader.c` + `O_DIRECT` + `io_uring` + 4-way set-associative C cache + `HostEngramHash` ahead-of-step C staging in `model_state.py:_early_engram_ids`). A Python background-thread advisory prefetch helper (`event.synchronize()` + `torch.unique(torch.cat(...))` under the GIL targeting `embed.disk_table`) is both inapplicable (targets retired `b12x` attributes) and an architectural regression relative to the zero-GIL C worker in `EngramDiskRows`.

## Memory & Resource Observations

- Patch `0762` allocates zero additional host or device memory; on `/v1/completions` requests with `structured_outputs`, it avoids allocating a redundant per-request `DeepSeekV41ParserReasoningAdapter` instance.
- Patch `0763` adds one `Optional[int]` field (`qos_cached_tokens_hint = None`) per `Request` and zero GPU memory overhead.

## Conclusion

1. **Adopt `0762`**: Fixes silent bypass of structured-output grammar constraints and draft-token trimming on `/v1/completions` under `--reasoning-parser deepseek_v41`, and aligns `DeepseekV41ModelState.prepare_attn` (`ubatch_idx: int = 0`).
2. **Adopt `0763` (scheduler library fix, keeping promoted `--max-parallel-prefills 1 --prefill-yield short` default)**: Eliminates aligned-chunk multi-lane livelocks and adds size-classed prefill lanes (`--prefill-long-threshold`, `--max-long-prefills`) and multi-lane livelock guards when `--max-parallel-prefills > 1` is used, while keeping `--max-parallel-prefills 1 --prefill-yield short` as the promoted serving configuration (`0.192 s` short-request TTFT and 100% aligned `16,288`-token chunks).
3. **Reject Python-thread Engram row prefetch**: Superseded by `1107`'s native C/TileLang `EngramDiskRows` (`O_DIRECT` + `io_uring` + user-space 4-way cache + ahead-of-step `HostEngramHash` staging).
