# r6x on the triangle at 1M context

Owner, 2026-10-19: once the triangle is re-cabled, "modify the tp=3 config so it has 1mb
context and suitable kv cache for 4.5x", then validate, bench and promote; and "is there enough
space to set max_num_batched_tokens to the same as tp=4?"

## Change (recipe sizing only; the image is r6x's)

`tp3-triangle` today serves 524,288 tokens with a 3.5 GiB KV pool (2,845,543 tokens). r6x's
expert scale codes free about 2.7 GiB a rank at TP3 (experiments/2026-10-18-scale-codes). The
candidate (`r6x-tp3-1m-1017.json`) serves 1,048,576 tokens with a 4.90625 GiB pool
(5,268,045,824 bytes): about 4,766,000 tokens, 4.55 full 1M contexts.

**First size, 5.875 GiB (2026-10-18 23:47 UTC): refused by dgx1's startup guard.** vLLM reported
5,707,081 tokens (5.44x 1M; more tokens a GiB than r6u's 512K boot implied), and both boots
of it ended at graph capture with dgx1's MemAvailable 4.81 and 4.96 GiB, below the 5 GiB
startup floor (exit 137; the guard, not the kernel's OOM killer). The measured 971,000 tokens a
GiB put 4.5x at 4.86 GiB, so the pool is 4.90625 GiB: 4.55x, about 0.8 GiB of startup margin
on dgx1.

**Prefill budget.** `r6x-tp3-1m-8192-1017.json` adds TP4's `max_num_batched_tokens` 8,192
(threshold 8,144). In September 8,192 stopped dgx1 at its startup guard on TP3 (r2); sequence
parallel prefill and the memory saver have changed the margin since, and the 4.55x pool now
competes for it. A probe boot and a 1M prefill answer it; it is chosen only if it fits with
margin and pays for its longer prefill stall.

## Runs (the triangle, 1017)

1. `gate-1017.json`: determinism (outputs equal to r6u's TP3 path: 62313cb5 / 9742ac2c with
   log probabilities, 4d143977 / 12fd8257 as text), distinct streams at 8 and 4, and each
   node's memory once serving (vLLM's KV line must report at least 4.5x for 1,048,576).
2. `prefill-1m-1017.json`: one cold 1,000,000-token prefill.
3. `batched-8192-1017.json`: the 8,192 probe: memory once serving, then the 1M prefill.
4. `release-tp3-1017.json`: release-tp3 with single-stream and sampled: the candidate, r6v's
   TP3 (production's TP3 build before r6x, 512K), the candidate again.

Gate: outputs identical; at least 4.5x reported; dgx1 above its guard through a 1M prefill; no
release verdict worse than r6v's TP3.

## Results so far (2026-10-19, the triangle, 1017)

**Capacity.** vLLM reports 4,765,898 KV tokens, 4.55x for 1,048,576 (r6v's TP3: 2,845,543,
5.43x for 524,288). Model load 94.26 GiB a rank.

**Gate** (`gate-1017.json`): temperature-0 and greedy outputs identical to r6u's TP3 path
(62313cb5 / 9742ac2c, 4d143977 / 12fd8257), zero token and log-probability differences in
every scenario. Distinct streams: c8 147.6 tok/s, c4 112.5 (r6u's TP3 gate, another session:
141.1 and 107.3). MemAvailable once serving: dgx1 6.70 GiB, dgx2 8.04, dgx3 8.11.

**Cold 1M prefill** (`prefill-1m-1017.json`): 1,000,000 real-text tokens at 3,227 tok/s (310 s),
none failed; dgx1's lowest MemAvailable during it 5.93 GiB (dgx2 7.36, dgx3 7.51), no swap.

**Prefill budget 8,192** (`batched-8192-1017.json`): it boots past the startup guard; serving
it leaves dgx1 5.07 GiB (dgx2 6.12, dgx3 6.25), 1.6-1.9 GiB less than 4,096; the cold 1M
prefill runs at 3,725 tok/s (271 s, about +15%, one sample) with dgx1's lowest MemAvailable
4.49 GiB, 1.5 GiB above its 3 GiB guard. Its cost, a prefill step of about 2.2 s against 1.3 s
during which every other stream waits, is not yet measured (servebench has no mixed step).

**Decision (owner, 2026-10-19):** TP3 promotes at 4,096; 8,192 is revisited after the CED
skip (the prefill follow-ups' items 1 and 5), with the stall measured.
