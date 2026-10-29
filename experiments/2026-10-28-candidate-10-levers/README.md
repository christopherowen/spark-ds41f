# The cluster program's levers on candidate 10

Phase 3 of the cluster program (`../2026-10-26-cluster-program/README.md`): the lean A/Bs of
the features built beside candidate 9, now on candidate 10 (rc10, vLLM 58938a074), judged
with the paired single-stream verdict (`../2026-10-26-paired-verdict/README.md`). One spec a
lever, by the one-at-a-time procedure: the lever's tests on candidate 10's image, then the bit
gates (determinism with and without logprobs, the seeded sampled digest) on the base and the
arm, then a lean suite on the base, the arm and the base again.

Each lever is its branch's commits cherry-picked unchanged onto candidate 10's head; none
conflicted. [vllm/](vllm/) holds them as patches. `make_levers.py` writes the bundles, arms
and specs from the three worktrees:

```bash
python3 experiments/2026-10-28-candidate-10-levers/make_levers.py \
  VLLM_C10_C1_FRONT VLLM_C10_COPY_DRAFTS VLLM_C10_PREFILL_YIELD
```

| Lever | Source commits | Port (tree) | Arm changes | Claim |
| --- | --- | --- | --- | --- |
| front | c1-front 8a70cb9a9, 0060836b4 | 067db2179 (061fa2751) | 2 files | same bits; c1 decode about +1.1% (with prefetch, the program's estimate) |
| copy | copy-drafts 05d5db22d..63082e9b9 (#619) | 4115f102d (0781ad915) | 20 files; `copy_drafts` | same text; agent traffic +4.75% at c1, +2.49% at c8 (#619's estimate) |
| yield | prefill-yield 4316d0fc6 (#620) | f90d1e7eb (b01440c72) | 3 files; `--prefill-yield short` | same bits; short-prompt TTFT behind a long prefill −45% to −95% (#620's model) |

## front (`front.json`, against candidate 10)

At decode rows with cache pages, the attention forward runs one TileLang kernel after the
fused Q-A/KV projection in place of five: the q and kv RMSNorms, the q latent's MXFP8 cast for
Q-B, the kv RoPE and the SWA cache record. Prefill, sequence-parallel rows and memory profiling
keep the five. Bundle `front` (dgx1): every output byte against the five kernels at 1 to 65
rows, and the forward's routing. Read the lean table's `c1: decode tok/s, paired` row.

## copy (`copy.json`, against copy-control)

Copy drafts beside DSpark: a request whose context's tail (16 to 256 tokens) appeared earlier
in its own history drafts the tokens that followed, 8 drafts doubling to 16 after a fully
accepted copy step; DSpark drafts everywhere else. The wider verification needs the copy
recipe's ladder (capture sizes to 107, budget 16395), so the control, `copy-control`, is
candidate 10 at that ladder without copies: the A/B isolates the drafter, and the ladder's
own cost is the control against candidate 10's other lean runs.

- Bundles: `copy` (dgx2): the copy scheduler, kernel and proposer, adaptive verification and
  aligned-chunk tests. `copy-wide` (dgx3): TileLang verification at 16 steps, the ratio-2
  compressor ring and kernel preparation.
- The gate block also runs `copy_probe.py --prompts 12` on both arms: tokens a step on
  copied text, which the lean suite's prompts barely exercise. The lean c1 row bounds the cost
  on text without copies. The agent gain needs AgentX (#619's `agentx-tp4.json`), which runs
  only if this lands level or better.

## yield (`yield.json`, against candidate 10)

Prefill yield: a long prompt's aligned chunk gives its step to waiting prompts that finish
within one chunk and to a held first token's step; chunk boundaries do not move. The suite,
`yield-suite.json`, is the lean suite with the concurrency-stalls suite's 131K stall probe
before the prefill step (500-token probes 2-14 s into a cold 131K prefill beside 11 decoding
streams, five repeats). Bundle `yield` (dgx4): the yield and aligned-chunk scheduler tests.
Read the probe's TTFT, and the c8, c1 and prefill rows for cost (claim: level).

### Depth (`yield-depth.json`, against candidate 10)

A site triangle (TP3, `max_num_seqs` 8, our recipe) measured on #650 that yield cuts a
probe's TTFT behind a 132K prefill 4.52 s to 1.68 s, but at 6 decoding streams takes the
median decode gap during the prefill from 9.7 to about 18 times the baseline, candidate 8's
level. Candidate 10 alone was 9.2 times at 7 streams and 1.2 times at 8. At 8 the long
prompt waits for a slot rather than prefilling beside the streams, so that row is queueing,
not interleaving. Yield acts only while the scheduler can admit (`len(running) <
max_num_seqs`), so its effect depends on depth; `yield.json` measures one depth, 11 of 16.
By design it never removes decode rows from a step, so the doubling is unexplained.

`yield-depth-suite.json` runs only the stall step, 131K, four repeats, at 3, 7 and 11
streams with the probes and at 15 without (15 and the long prompt fill the sixteen slots).
Read the decode p50/p99 and the probe TTFT at each depth. Yield stays in candidate 11 only
if no depth shows a decode cost; otherwise it leaves the series before candidate 11's
qualification and is reworked.

## Results

(Filled in as each spec lands.)

- **copy, first run** (`c10-copy`, 2026-10-28): copy-control booted and passed its gates and
  `copy_probe.py`; the copy arm never became ready. Every worker failed in warmup at
  `assert self.adaptive_verification is not None` in `sample_tokens`: `warmup_kernels` clears
  the manager so pre-capture steps use fixed draft counts, and the copy path required it. The
  fix (vLLM 657f0df54) takes no verified drafts when the manager is absent, as `propose()`
  already reads None; `make_levers.py` rewrote only that file in `overlay/copy` and the two
  copy bundles. Requeued ahead of candidate 11.
