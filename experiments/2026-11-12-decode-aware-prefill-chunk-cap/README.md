# Decode-aware prefill chunk cap below 256K

On TP4, one 16,288-token prefill chunk step stops every running decode for
1.9 s at 64K context and 2.3 s near 256K. Patch 0751 already steps the chunk
down while decodes are runnable, but only at 256K and beyond. This experiment
adds an opt-in cap for the range below 256K:

```
SPARK_DS41F_PREFILL_MIX_CHUNK=8192
```

With the variable set, `_context_aware_chunk_threshold` returns
`min(long_prefill_token_threshold, cap)` below 256K when decodes are runnable
and `--prefill-yield short` is on. Unset, it returns the full threshold, so the
default scheduling is the 1107 scheduling. Solo prefills, prefills without
prefill yield and the 256K+ tiers (4,096 and 2,048) are untouched. Only
positive multiples of the 2,048-token grid are accepted; any other value logs
a warning and keeps the default. A cap at or above the threshold changes
nothing.

The intended variable is the chunk size of a prefill that shares its steps
with decodes, below 256K context.

## Source

- Deployment base: `7b2d6d45` (Promote spark-ds41f-1107).
- vLLM: the frozen 1107 series plus
  `vllm/0762-sched-opt-in-decode-aware-prefill-chunk-cap-below-256K.patch`
  (`vllm/series`). A clean `git am` replay with `bin/spark build prepare`'s
  committer and dates reproduces the 1107 head `2756478b` and tree `ed1bdb7a`,
  then gives head `39dcbc20` and tree `2eb7063c` with this patch
  (`source.json`, `upstreams.lock.json`). `git diff --check` from the base
  revision is clean.
- The patch changes `vllm/v1/core/sched/scheduler.py` (+38 −4) and adds
  `tests/v1/core/test_prefill_mix_chunk.py`. On promotion the new test file
  needs a `gate` entry in `config/release-tests.json`, beside
  `test_aligned_prefill_chunks.py` and `test_prefill_yield.py`.
- `tp4-switched.json` is the profile for this experiment
  (`spark-ds41f:1107-mix-chunk-cap`, launch disabled until an image built from
  this series is qualified).

## Tests

```
pytest -q -p no:cacheprovider tests/v1/core/test_prefill_mix_chunk.py \
  tests/v1/core/test_aligned_prefill_chunks.py tests/v1/core/test_prefill_yield.py
```

69 passed (CPU container on the 1107 image, with this patch's scheduler and
test mounted). The new tests cover: unset keeps 1107 at every context and
yield setting, 8,192 and 4,096 cap below 256K with decodes, solo prefill and
no yield keep full chunks, the 256K+ tiers stay, a cap at or above the
threshold is a no-op, the parser (unset, empty, 8192, 4096, 2048, 1024, 5000,
−4096, `abc`), and aligned chunks ending on the cap's grid. Importing the
scheduler with the variable unset, `8192` and `5000` gave a cap of 0, 8,192
(with the info log) and 0 (with the warning).

## Measurement

Four DGX Spark, `tp4-switched` shape over a 200GbE RoCE switch, stock 4 KiB
pages, official DeepSeek V4.1 Flash-1017 weights (`1bae0e10`), 1M context, the
frozen 1107 recipe with site-specific memory guard lines, 2026-11-09. The
measured image was the 1107 image with this scheduler copied over its own
(`scheduler.py` at this patch, before the comments, docstring, helper name and
log text were made neutral and the invalid-value warning was added). Both arms
used that image; the default arm left the variable unset, the cap arm set
`8192`. Each arm was booted four times across three windows (start 236–253 s).

The stall workloads stream three ~2K-token prompts (600 output tokens forced
with `min_tokens`; 2,400 for the 256K case) and send one long prompt 2 s later.
The stall is the largest inter-token gap any of the three streams sees.

| Metric | Default (1107) | Cap 8192 | Change |
|---|---:|---:|---:|
| Max decode stall, 64K prompt (3 rounds) | 1.894–1.905 s | 1.020–1.047 s | −45% |
| Max decode stall, 256K prompt (2 rounds) | 2.249–2.268 s | 1.258–1.266 s | −44% |
| TTFT of the 64K prompt itself (mean of 3) | 9.083 s | 10.114 s | +11.4% |
| TTFT of the 256K prompt itself (mean of 2) | 38.605 s | 43.775 s | +13.4% |
| 2K prompt arriving during a 64K prefill, TTFT (median of 4 run medians) | 2.795 s | 2.818 s | +0.8% |
| Isolated 32K prefill (mean of 5 boot runs, 3 prompts each) | 8,435 tok/s | 8,342 tok/s | −1.1% |
| C16 aggregate decode (servebench, 10 batches) | 245.7 tok/s ±1.37% | 244.7 tok/s ±1.36% | −0.4% |

`results.json` has every round and boot run behind these rows. The TTFT cost
is the trade-off this cap buys the stall with: the 69,418-token prompt takes 9
instead of 5 chunk steps, each paying the fixed per-step cost that kept 0751's
step-down at 256K and beyond. It is why the cap is opt-in.

Quality checks on both arms: the 6/6 smoke checks, whose two temperature-0
greedy checks kept 1107's hashes (`9de7f6a0e610`, `0ef91bc1ba4e`), authentication 9/9, and the
gateway acceptance suite 18/18 on the cap arm. These short prompts run solo,
so they exercise the default path. Under mixed load the cap moves chunk
boundaries, so a long prompt's greedy output can depend on whether decodes
were running, as it already can at 256K and beyond under 0751. That case was
not measured.

The cap arm has run in this site's production since 2026-11-09 (same image,
same setting); a single 64K mixed round there gave a 1.02 s max stall and a
9.84 s TTFT.

## Unfavourable runs and errors

- The first window's default 32K prefill run failed before sending (a missing
  local input file); it was rerun in the second window.
- The cap arm's 32K runs spread lower: 8,257.6–8,442.7 tok/s against
  8,421.8–8,440.3 tok/s, with one prompt at 7,895.5 tok/s. Step timings were
  not recorded, so the cause of that low run is unknown.
- The first 2K-arrival round of the cap arm was +9.8% (3.058 s); three more
  rounds gave 2.887, 2.743 and 2.749 s.
- The first boot of the image took 862 s (JIT caches); later boots 236–253 s.

## Memory

Minimum MemAvailable during the first window: default 22.44 / 23.43 / 24.15 /
24.49 GiB, cap 21.98 / 23.12 / 23.63 / 24.24 GiB (spark1..4). Swap stayed at
0.11–0.78 GiB in both arms. MemAvailable stayed above the 16 GiB memory guard
warning line, and the guard logged no action.

## Scope

Measured only on the 4 KiB, switched TP4 shape. The 64 KiB kernels and the
ring shapes may give different stalls and costs; TP3's 6,096-token chunk is
already below 8,192, where the cap changes nothing.

## Conclusion

Retain as an opt-in experiment. At 8,192 the cap cuts the worst decode stall
below 256K by 44–45% for an 11–13% longer TTFT of the long prompt itself, with
isolated prefill and C16 decode within noise. Before promotion, build the image
from this series, qualify it on repeated boots, and run the determinism gate
with mixed load.
