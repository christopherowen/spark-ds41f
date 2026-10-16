# r6t: startup collectives wait as long as the startup rendezvous

r6t is r6s ([experiments/2026-10-16-r6s](../2026-10-16-r6s/README.md)) with vLLM 0156 and
sparknet's `startup-timeout` branch (`dd2767b`, timeout-stop plus one adapter method).

## Why

A forum report on r6n (topic 384438, [post 24](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438/24),
huxiaofengtiger) failed three TP3 starts in memory profiling: `cuModuleLoadData` returned
`CUDA_ERROR_LAUNCH_FAILED`, and a diagnostic patch showed rank 0's one-shot collective had
seen nothing from rank 1 for 10 s.

Kernels still JIT-compile inside memory profiling, for as long as each rank's cache needs.
The startup rendezvous (vLLM 0143) lines ranks up before `profile_run`, but `profile_run`
is a distributed forward: a rank whose TileLang cache is warm reaches the next collective
and waits there for a rank that is compiling, and the serving wait (10 s) stops the GPU.
A failed start tears the other ranks down mid-compile, so their caches fall further
behind and the next start fails the same way. Identical nodes starting with equally empty
caches compile in near-lockstep, which is why r6n's empty-cache boot (409 s) passed here.

Reproduced on r6s, 2026-10-16 22:44 UTC: production's recipe with dgx2's TileLang cache
moved aside (dgx1 and dgx3 warm) fails in `determine_available_memory` -> `profile_run`
while dgx2 compiles the attention metadata kernels (8 s each), with an Xid 43 on dgx1 and
the error above, no collective named. `cluster start`'s rollback to the previous
containers failed the same way (dgx2 still partly cold). Production returned once dgx2's
cache was moved back (22:53). Logs: `results/private/failed-starts/20261016T224947Z-dsv41-karmic-kraken`
and `results/private/lab/r6s-coldrank-20261016T224322Z` on dgx1.

## Change

- **0156.** From the communicators' creation, startup's one-shot collectives wait as long
  as the startup rendezvous (`VLLM_STARTUP_RENDEZVOUS_TIMEOUT_MS`, 30 min); the serving
  wait (`SPARKNET_ROCE_TIMEOUT_MS`, 10 s) returns once warm-up is done, before the JIT
  monitor starts. The value is a control word the kernels read at their first clock check,
  so captured graphs follow. A stopped collective still stops the GPU. Memory profiling and
  warm-up now re-raise a CUDA error as the stopped collective's report, as
  `execute_model` and `sample_tokens` do (`profile_run` calls the model runner directly).
  Tests: `tests/v1/worker/test_b12x_roce_health.py`.
- **sparknet `dd2767b`.** `SparknetOneShotAllReduce.set_timeout_ms` passes the wait to the
  runtime (which has had `set_timeout_ms` since timeout-stop) and returns the value it
  replaced.

Serving is unchanged: the same kernels, the same 10 s wait after warm-up.

Patch head `14d3846f` (tree `39c5c4c8`), replayed on `19f2c20e` as `build prepare` does.

## Runs

1. `image.json`: built on dgx2 with serving stopped, copied over the CX7 links.
2. `tests.json`: the worker and RoCE-policy tests in the image.
3. `gate.json`: temperature-0 determinism and distinct streams; outputs must match r6s's.
4. `coldrank.json`: [r6t-tp3-coldrank.json](r6t-tp3-coldrank.json), a TileLang cache seeded
   on dgx1 and dgx3 with copies of production's (r6t's kernels are r6s's) and empty on
   dgx2. It must boot, log the startup and serving waits, and serve.
5. `release.json` (on the owner's word): release-tp3, r6t / r6s / r6t again.
