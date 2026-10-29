# Copy drafts on candidate 12, fixed

Copy drafts (#619) left the release at candidate 12: enabled, they failed to start, and the
release gate had 40 failures in their test stand-ins. A release ships no feature that does
not work and none switched off but IKET's debug mode, so copy drafts return only as a
candidate that starts, passes the full release gate with their own tests in, keeps the bits,
and gains.

The branch `copy-drafts-c12` (vLLM, head acd5d49b6, tree 520080a98) is candidate 12's head
(d984b2e66) with candidate 11's six copy commits cherry-picked unchanged (no conflicts) and two
fixes; [vllm/](vllm/) holds the eight as patches.

- **The start failure** (`c10-copy`, `c10-copy-2`): the warmup's eager step at 214 tokens
  (adaptive verification's tail sizes past the 107-row ladder) lays out 16 requests of 13-14
  tokens. Copy drafts raise the reorder threshold to 1 + 16 rows, so every one is a decode
  request, and the decode metadata arena, sized for DSpark's rows plus one copying request
  (16 x 11 + 11 = 187), cut the step's SWA indices to 187 rows; the sparse-MLA kernel refused
  them (`SwaIndices shape[0]`, 214 expected). Any step of 16 short prompts could do the same
  in serving. The arena now holds every request at the threshold (16 x 17 = 272 rows, a
  few kilobytes of int32 metadata); without copy drafts it is unchanged (16 x 11).
- **The 40 gate failures**: the stand-ins of seven test files (sp-prefill, compressor,
  attention, rejection-sampler utils and chunking, sink, verified logprob token ids) lacked
  the fields copy drafts added: `max_decode_rows`, `num_verification_tokens`,
  `num_draft_steps`, `num_draft_tokens_per_req`. They carry them now, at their values without
  copy drafts. The attention layer's test also checks the arena at both thresholds.

`make_copy.py` writes the bundles, arms and spec from the worktree:

```bash
python3 experiments/2026-10-29-copy-drafts-c12/make_copy.py VLLM_COPY_DRAFTS_C12
```

## The run (`copy.json`, `c12-copy`)

- **Bundles:** `copy` (dgx2): the copy scheduler, kernels and proposer, adaptive
  verification, aligned chunks and the attention layer. `copy-wide` (dgx3): TileLang
  verification at the 16-draft lookahead, the ratio-2 compressor ring, kernel preparation.
  `release-tests-copy` (dgx4): candidate 12's release gate (114 files) with copy drafts' two
  test files back in (`release-tests-copy.json`; 116 files, 0 classification problems), on
  the copy overlay. It must pass with no failures.
- **Bits:** copy-control (candidate 12 at the copy recipe's ladder, capture sizes to 107,
  budget 16395, DSpark alone) and copy through the determinism gates with and without
  logprobs and the seeded sampled digest; then `copy_probe.py --prompts 12`, tokens a step on
  copied text.
- **Speed:** the lean suite on copy-control, copy and copy-control again. Read the
  `c1: decode tok/s, paired` and c8 rows: the lean prompts barely copy, so these bound the
  cost on ordinary traffic and should be level.

Copy drafts join candidate 13 only if every bundle passes, the bits match, and the lean rows
are level; the agent gain (#619's estimate +4.75% at c1, +2.49% at c8) is then measured with
AgentX (#574) before the release claims it.

## Results

Pending.
