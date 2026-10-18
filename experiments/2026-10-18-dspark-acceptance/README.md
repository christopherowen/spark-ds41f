# DSpark acceptance: where the ceiling is

Owner, 2026-10-18: "if we were to think outside of the box, what if we were to design the
dspark relationship to serving differently? would we be able to achieve better? [...]
return a tree of probabilities? 'fuse' drafting and verification?" and "when we take on
the task to improve acceptance, we should rework this part too" (the boot-timed cost
model; see experiments/2026-10-18-dspark-profile-replays).

## Measured first (no build)

From the r6w one-stream profile (experiments/2026-10-18-r6w-swiglu, `profile.json`, rank 0,
26 steps of 1 + 5 drafts, under the torch profiler):

- **The GPU is busy 97.6% of the window** (any stream); idle time is 2.4%, nearly all in
  gaps under 10 us. Fusing drafting and verification into one graph removes gaps, so its
  ceiling is about 2%.
- **Kernel time a step:** routed MoE 19.0 ms (about half), the L2 prefetch 10.7 ms (a side
  stream, overlapped), MXFP8 GEMMs 5.1 ms, one-shot all-reduces 3.8 ms (about 80 a step),
  vocabulary heads 2.1 ms, mHC 1.7 ms.
- **Tokens a step at one stream: 2.24** of a possible 6 (accepted 1.27 a draft; release
  suite, r6w and r6v alike). The step's cost barely depends on how many drafts are accepted,
  so acceptance is where the ceiling is: 3.0 tokens a step would be about +34% at one stream.

## Candidates, in order

1. **Probabilistic drafting** at the recipe's temperature 1 (`draft_sample_method:
   probabilistic`): a configuration change, measured in experiments/2026-10-18-dspark-probabilistic.
2. **Copy-aware drafting for agent traffic:** when the context holds a long exact
   continuation (tool output, code), a suffix/n-gram drafter proposes 8-16 tokens at almost
   no cost (vLLM has a `suffix` method); a per-request hybrid with DSpark. Ceiling: the share
   of agent output tokens that continue text already in the context, to be measured on
   private traffic statistics first.
3. **Draft trees:** DSpark drafts a block in one pass and has every position's distribution,
   so branches cost little to draft; verification needs ancestor masks through sparse MLA, the
   sliding window, the compressor, the indexer, Engram and CED. Largest ceiling at low
   concurrency, largest change in the stack; only after 1-2 bound what is left.
4. **The draft budget's cost model** (owner: no boot timings): replace it with a rule of the
   inputs. A survival-threshold version is parked on vLLM `dspark-heuristic-budget`
   (c8abdb156) for this task, to be measured against the current budget and against
   adaptive verification off.
