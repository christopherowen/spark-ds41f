# Periodic SSE Keep-Alive Heartbeats and Shielded Streaming Abort Cleanup (`2026-11-08-sse-keepalive-heartbeat`)

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18b6ca6521b602f09cb3ba326774166d1c` (Candidate `1107`, `experiments/2026-10-26-spark-ds41f-candidate/1107`)
- **Base vLLM commit**: `320021e3c4dbb202d911ab70e2cd924c75420488` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, `0001..0761`)
- **Candidate vLLM commit**: `a0d3351bb5f460b956d23e5e3897bc78660d4718` (tree `fe0b71627865b60990e6bf81704d40f6224edecf`, `0001..0762`, `patchset_sha256` `ef575f1753e21040eb5a20607db5801442537ee951ebb07af200c75e6c39b268`)
- **Intended causal variable**: Enable periodic SSE comment heartbeats (`: keep-alive\n\n`) during long queued or cold-prefill waits across all OpenAI/Anthropic streaming routes (`VLLM_SSE_KEEPALIVE_INTERVAL_S`, default `15.0s`), replace per-token `asyncio.wait` timer allocation with an eager-start task + rolling `loop.call_at` timer, and shield streaming disconnect abort cleanup through Starlette's `StreamingResponse` task group (`0762-feat-serve-sse-keepalive-heartbeat-and-abort-cleanup.patch`).

## 1. Technical Audit & Motivation

At `200K–1M` context lengths or when multiple requests queue behind an active long prefill, time-to-first-token (TTFT) can span `15–175s`. Inspecting `vllm/entrypoints/serve/utils/sse_keep_alive.py`, `vllm/entrypoints/serve/utils/api_utils.py` (`0725`/`0726`), `vllm/v1/engine/async_llm.py`, and the four streaming API routers in Candidate `1107` uncovered four gaps:

1. **Disabled by default and missing on `/v1/responses` and `/v1/messages`**:
   - `sse_keep_alive_interval` defaulted to `0` (`None` unless `--sse-keep-alive-interval` was explicitly passed on the CLI), had no environment-variable override (`VLLM_SSE_KEEPALIVE_INTERVAL_S`), and was only wired to `/v1/chat/completions` and `/v1/completions` (omitting `/v1/responses` and `/v1/messages`).
2. **Per-token `asyncio.wait` timer churn during normal decode streaming**:
   - `_keep_alive_stream` executed `done, _ = await asyncio.wait({next_task}, timeout=interval)` on every yielded chunk. Under `uvloop`, every token allocated a 1-element `set`, a waiter `Future`, and a `loop.call_later(interval, ...)` `TimerHandle` (immediately cancelled when `next_task` completed in `~16 ms`), and forced an event-loop task switch even when `next_task` resolved synchronously (`q.get_nowait()`, role+content chunk pairs, usage+`[DONE]` pairs).
3. **Broken async abort cleanup when a streaming client disconnects (`0725`/`0726` gap for `stream=True`)**:
   - `@with_cancellation` monitors `raw_request.is_disconnected()` only until the route handler returns `StreamingResponse(...)`. Because the streaming generator has not started yet when `StreamingResponse` is returned, `@with_cancellation` exits before prefill begins.
   - Starlette's `StreamingResponse.__call__` then runs `stream_response(send)` and `listen_for_disconnect(receive)` inside its own `anyio` `create_collapsing_task_group()`. When `listen_for_disconnect` returns on client disconnect, AnyIO's `CancelScope._deliver_cancellation` schedules `task.cancel()` on every event-loop turn (`loop.call_soon`).
   - Neither `_keep_alive_stream`'s `finally:` block nor `AsyncLLM.abort()` shielded their cleanup with `anyio.CancelScope(shield=True)` + a cancellation-resilient wait loop. As a result, `await self.abort(q.request_id, internal=True)` inside `AsyncLLM.generate()` was cancelled mid-await on the first `await` inside `engine_core.abort_requests_async(...)` (`0 / 32` completed aborts reproduced on `sp8`), after `output_processor.abort_requests()` had already removed the frontend state—leaving the `EngineCore` prefill running on the GPUs.
4. **Disconnect during `await send(...)` left the streaming generator unclosed**:
   - If the client disconnected while `StreamingResponse.stream_response` was suspended inside `await send(...)` (transmitting either a `: keep-alive\n\n` frame or a data chunk) rather than inside `body_iterator.__anext__()`, Starlette exited `async for chunk in self.body_iterator:` without calling `await self.body_iterator.aclose()`, leaking `next_task` and decrementing `server_load_metrics` while the engine request remained active.

## 2. Implementation (`0762-feat-serve-sse-keepalive-heartbeat-and-abort-cleanup.patch`)

1. **`vllm/entrypoints/serve/utils/sse_keep_alive.py`**:
   - Added `DEFAULT_SSE_KEEP_ALIVE_INTERVAL_S = 15.0` and `get_sse_keep_alive_interval(raw_request=None, default=15.0)` resolving `VLLM_SSE_KEEPALIVE_INTERVAL_S` (`VLLM_SSE_KEEP_ALIVE_INTERVAL`), `raw_request.app.state.args.sse_keep_alive_interval` (`> 0` custom interval, `< 0` disabled, `0` or `None` -> `15.0s` default).
   - Replaced per-chunk `asyncio.wait` in `_keep_alive_stream` with `asyncio.Task(anext(generator), loop=loop, eager_start=True)` and a single rolling `loop.call_at` timer (`_arm_timer`) that only re-arms when a wait actually suspends past `last_activity + interval`. Synchronous yields return immediately on the `next_task.done()` fast path with zero `Future` or `TimerHandle` allocation.
   - Added `consumer_task.add_done_callback(_on_consumer_done)` and `_ensure_cleanup()` shielded with `anyio.CancelScope(shield=True)` + `_shield_wait()` so cancellation during either prefill wait or `await send(...)` cancels `next_task` once, waits for its async abort cleanup to finish, and closes `generator`.
2. **`vllm/entrypoints/serve/utils/api_utils.py`**:
   - Added `finalize_response_load(request, response)` to `load_aware_call` so `StreamingResponse.body_iterator.aclose()` is awaited inside `anyio.CancelScope(shield=True)` before `decrement_server_load(request)` runs.
3. **`vllm/v1/engine/async_llm.py`**:
   - Shielded `await self.engine_core.abort_requests_async(all_request_ids)` inside `AsyncLLM.abort()` with `anyio.CancelScope(shield=True)` and a cancellation-resilient `asyncio.wait` loop so AnyIO's level-triggered `CancelScope` cannot interrupt ZMQ engine abort delivery after `output_processor.abort_requests` has removed local state.
4. **OpenAI & Anthropic Streaming Routers**:
   - Wired `with_sse_keep_alive(..., get_sse_keep_alive_interval(raw_request))` across `vllm/entrypoints/openai/chat_completion/api_router.py`, `vllm/entrypoints/openai/completion/api_router.py`, `vllm/entrypoints/openai/responses/api_router.py`, and `vllm/entrypoints/anthropic/api_router.py`.
5. **Release-Gate Unit & Integration Tests (`tests/entrypoints/serve/utils/test_api_cancellation.py`)**:
   - Added 20 test cases covering interval resolution, long-prefill `: keep-alive\n\n` cadence and OpenAI SDK SSE parser compatibility, streaming disconnects during prefill and `send()`, repeated cancellation during streaming abort cleanup, `AsyncLLM.generate()` + `EngineCore.abort_requests_async` streaming disconnect end-to-end, and router-level SSE keep-alive wrapping across all 4 endpoints.

## 3. Exact Commands & Workload Identity

Executed on `sp8` (`spark-ds41f:1107`, Python `3.12.3`, `uvloop`) across two independent container boots (`boot1` and `boot2`):

```bash
for boot in 1 2; do
  docker run --rm --entrypoint "" -v /tmp/vllm-sse-work:/tmp/vllm-sse-work spark-ds41f:1107 bash -lc '
    set -euo pipefail
    for f in \
      vllm/entrypoints/serve/utils/sse_keep_alive.py \
      vllm/entrypoints/serve/utils/api_utils.py \
      vllm/v1/engine/async_llm.py \
      vllm/entrypoints/openai/chat_completion/api_router.py \
      vllm/entrypoints/openai/completion/api_router.py \
      vllm/entrypoints/openai/responses/api_router.py \
      vllm/entrypoints/anthropic/api_router.py \
      tests/entrypoints/serve/utils/test_api_cancellation.py; do
      cp /tmp/vllm-sse-work/$f /opt/spark-ds41f/candidate/vllm/$f
    done
    cd /opt/spark-ds41f/candidate/vllm
    pytest --noconftest -v \
      tests/entrypoints/serve/utils/test_api_cancellation.py \
      tests/v1/engine/test_request_submission_cleanup.py \
      tests/v1/engine/test_utility_call_cleanup.py
  ' | tee receipts/pytest-boot${boot}.txt

  docker run --rm --entrypoint "" \
    -v /tmp/vllm-sse-work:/tmp/vllm-sse-work \
    -v /tmp/bench_sse_keepalive.py:/tmp/bench_sse_keepalive.py:ro \
    spark-ds41f:1107 bash -lc '
    set -euo pipefail
    for f in \
      vllm/entrypoints/serve/utils/sse_keep_alive.py \
      vllm/entrypoints/serve/utils/api_utils.py \
      vllm/v1/engine/async_llm.py \
      vllm/entrypoints/openai/chat_completion/api_router.py \
      vllm/entrypoints/openai/completion/api_router.py \
      vllm/entrypoints/openai/responses/api_router.py \
      vllm/entrypoints/anthropic/api_router.py; do
      cp /tmp/vllm-sse-work/$f /opt/spark-ds41f/candidate/vllm/$f
    done
    python3 /tmp/bench_sse_keepalive.py
  ' | tee receipts/bench-boot${boot}.json
done
```

## 4. Two-Boot Verification & Benchmark Results (`sp8`, `spark-ds41f:1107`)

### 4.1 Release-Gate Cancellation & Cleanup Test Suite (`boot1` & `boot2`)

- `receipts/pytest-boot1.txt`: **`71 passed in 5.83s`**
- `receipts/pytest-boot2.txt`: **`71 passed in 6.01s`**

### 4.2 Streaming Decode Wrapper & End-to-End ASGI Pipeline Throughput (`uvloop`)

| Workload (`257` chunks/stream) | Arm | Boot 1 `µs/chunk` | Boot 2 `µs/chunk` | Boot 1 `chunks/s` | Boot 2 `chunks/s` | Mean Stream Latency (Boot 1 / Boot 2) | vs `baseline_old_15s` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Micro stream `c1` (`51,400` chunks) | `disabled_0s` | `0.1050` | `0.1066` | `9,519,819` | `9,376,992` | `0.0228 / 0.0228 ms` | — |
| Micro stream `c1` (`51,400` chunks) | `baseline_old_15s` | `4.2810` | `4.2893` | `233,589` | `233,138` | `1.0959 / 1.0980 ms` | `1.00x` |
| Micro stream `c1` (`51,400` chunks) | **`candidate_new_15s`** | **`0.5068`** | **`0.5082`** | **`1,973,254`** | **`1,967,569`** | **`0.1258 / 0.1260 ms`** | **`8.44x` (`-88.2%` `µs/chunk`)** |
| Micro stream `c16` (`82,240` chunks) | `disabled_0s` | `0.0931` | `0.0934` | `10,736,144` | `10,711,086` | `0.0225 / 0.0226 ms` | — |
| Micro stream `c16` (`82,240` chunks) | `baseline_old_15s` | `3.3162` | `3.3439` | `301,551` | `299,055` | `13.5868 / 13.6999 ms` | `1.00x` |
| Micro stream `c16` (`82,240` chunks) | **`candidate_new_15s`** | **`0.4942`** | **`0.4971`** | **`2,023,442`** | **`2,011,865`** | **`1.1058 / 1.1133 ms`** | **`6.72x` (`-85.1%` `µs/chunk`)** |
| Full ASGI `StreamingResponse` `c1` (`30,840` chunks) | `disabled_0s` | `0.7181` | `0.7081` | `1,392,652` | `1,412,141` | `0.1783 / 0.1757 ms` | — |
| Full ASGI `StreamingResponse` `c1` (`30,840` chunks) | `baseline_old_15s` | `4.9991` | `5.0056` | `200,035` | `199,778` | `1.2791 / 1.2806 ms` | `1.00x` |
| Full ASGI `StreamingResponse` `c1` (`30,840` chunks) | **`candidate_new_15s`** | **`1.5530`** | **`1.5509`** | **`643,928`** | **`644,792`** | **`0.3931 / 0.3925 ms`** | **`3.22x` (`-69.0%` `µs/chunk`)** |
| Full ASGI `StreamingResponse` `c16` (`41,120` chunks) | `disabled_0s` | `0.6081` | `0.6038` | `1,644,391` | `1,656,109` | `2.4287 / 2.4115 ms` | — |
| Full ASGI `StreamingResponse` `c16` (`41,120` chunks) | `baseline_old_15s` | `3.8968` | `3.9265` | `256,622` | `254,679` | `15.9554 / 16.0772 ms` | `1.00x` |
| Full ASGI `StreamingResponse` `c16` (`41,120` chunks) | **`candidate_new_15s`** | **`1.2868`** | **`1.2860`** | **`777,097`** | **`777,585`** | **`5.2178 / 5.2144 ms`** | **`3.04x` (`-67.1%` `µs/chunk`)** |

At normal decode step times (`~16 ms/step`), the net wrapper cost of `candidate_new_15s` over `disabled_0s` in the full ASGI pipeline is **`0.84 µs/chunk` (`0.005%` of a `16 ms` decode step)** at `c1` and **`0.68 µs/chunk`** at `c16`.

### 4.3 Long-Prefill Heartbeat Cadence & Client Disconnect Abort Completion

| Metric | Boot 1 | Boot 2 |
| --- | ---: | ---: |
| Prefill stall (`72.0 ms`, `interval = 20.0 ms`) `: keep-alive\n\n` frames emitted | `3` (`20.22, 20.11, 20.09 ms`) | `3` (`20.53, 20.64, 20.30 ms`) |
| First-token TTFT handoff overhead when token arrives mid-interval | `8.14 µs` (`TTFT = 72.502 ms`) | `27.70 µs` (`TTFT = 72.475 ms`) |
| Concurrent streaming disconnects (`32` requests) completed async aborts (`baseline_old`) | `0 / 32` (`0.0%`) | `0 / 32` (`0.0%`) |
| Concurrent streaming disconnects (`32` requests) completed async aborts (`candidate_new`) | **`32 / 32` (`100.0%`)** | **`32 / 32` (`100.0%`)** |
| Leaked `server_load_metrics` or `OutputProcessor` states after disconnect | `0` | `0` |

## 5. Conclusion

Enabling periodic SSE keep-alive heartbeats (`15.0s` default via `VLLM_SSE_KEEPALIVE_INTERVAL_S`) with an eager-start task + rolling timer and AnyIO-shielded streaming abort cleanup keeps long-context (`200K–1M`) queued/prefill SSE streams alive across proxies, cuts per-token keep-alive wrapper overhead by `67–88%`, and restores `100%` (`32/32`) async engine abort completion when streaming clients disconnect during prefill or chunk transmission.
