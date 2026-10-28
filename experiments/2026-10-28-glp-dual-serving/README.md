# GLP dual serving: the stock model and its steered variant under two names

One engine serves `deepseek-v4.1-flash` stock beside `deepseek-v4.1-flash-weightless`,
the same weights steered with the published GLP file. A request picks a variant by its
`model`. The goal is a recipe that serves both by default, with no special setup beyond
downloading the GLP file.

- **Base:** spark-ds41f-1025 (vLLM patch head 11171760, tree 6f8e062f; image
  `spark-ds41f:19f2c20ed4d6-next-6`), spark-ds41f main c1cf4990.
- **vLLM:** branch `glp-dual-serving` (local, not pushed), one commit on the 1025 head:
  [vllm/0538](vllm/0538-DeepSeek-V4.1-GLP-steering-per-request-under-a-secon.patch), head
  ee78356d, tree ebb7aaa8. [vllm/series](vllm/series) is 1025's series plus 0538, and
  [source.json](source.json) and [upstreams.lock.json](upstreams.lock.json) are its lock.
  0538 is provisional: 0537 is taken by the copy-drafter branch, and the series is
  renumbered where it joins a release.
- **The variable:** steering selected per request by model name (0538), with the recipe
  serving both names.
- **Status:** built and CPU-tested; GPU qualification is prepared, not run (see
  [Runs](#runs)).

## What changes

**In vLLM (0538).** Before 0538, `WEIGHTLESS_STEER_PATH` steered every request, and each
covered layer folded every row explicitly. That cost 2.4% on the decode grid in the
next-release qualification ([2026-10-23-next-release](../2026-10-23-next-release/README.md)).
0538 makes steering a property of each row:

- `WEIGHTLESS_STEER_MODEL` (new) names the served model whose requests are steered. When it
  is unset, every request is steered, as before. Setting it without
  `WEIGHTLESS_STEER_PATH` is refused at startup.
- The OpenAI server marks a request to that name in its sampling parameters. The response
  reports the name the request asked for. Chat completions (and messages), completions,
  responses and tokenization serve the name. Every other server, and beam search, refuses
  it with HTTP 400 instead of serving it unsteered.
- The scheduler keys a marked request's first prefix block apart, as a cache salt does.
  Steered rows write different KV, so neither name reuses the other's cached blocks.
- Before each step, the model state writes each row's mark (0 or 1) into an int32 buffer:
  a TileLang kernel reads the step's query bounds on the GPU and each request's mark from a
  UVA array. CUDA graphs replay that buffer in place. The mark follows each row through
  sequence-parallel sharding and CED's compaction.
- Each covered layer (1-39) runs one TileLang launch over the step's rows:
  - **An unmarked row:** its block returns after reading its mark, so its residual, `post`
    and `comb` keep their bits. Stock traffic sharing a step with steered traffic is
    therefore bit-identical to an engine with steering off.
  - **A marked row:** the kernel adds the layer's write from each rank's unreduced parts,
    in rank order. It then folds the write as mHC's post kernel does, projects each of the
    four streams off the layer's direction (h ← h − α(h·d)d, α 0.75), and leaves the result
    as the row's residual, with `comb` set to the identity and `post` to zero.
  - **The next reader:** the next layer's fused post/pre (or an Engram layer's post, the
    drafter's auxiliary states, the final collapse) folds a marked row back exactly. A
    negative zero may come back positive.

**In the recipe ([cluster.json](cluster.json); a promotion would make it
`config/cluster.json`).** The recipe changes in three places:

- `--served-model-name deepseek-v4.1-flash deepseek-v4.1-flash-weightless`. The stock
  name stays first, so `bin/spark` and every probe that takes the first entry of
  `/v1/models` keep using stock.
- `WEIGHTLESS_STEER_MODEL=deepseek-v4.1-flash-weightless` and
  `WEIGHTLESS_STEER_PATH=/glp/vector.gguf`.
- A read-only mount of the GLP file from the hub cache at `/glp/vector.gguf`. The file is
  `msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-39-L1-39-a0.75`, revision
  `2715ffd51f59899115853e1118db080536966c44`, file
  `glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf` (803,392 bytes, MIT). Because it
  is a mount, `cluster start` reports a node without the file ("ro mount source is
  unavailable") before anything boots. The repository's `main` has moved on (c275c3ad, which
  adds an `a1.0-dealign-attn` file); the pin keeps the audited vector.

The one setup step a site needs, beside the model's download (docs/setup.md section 5, at
promotion):

```sh
hf download msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-39-L1-39-a0.75 \
  glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf \
  --revision 2715ffd51f59899115853e1118db080536966c44
```

The repository is gated: the downloading account must first accept the author's
responsible-use agreement. **On 2026-10-28 the owner's account was not on its authorized
list** (the API answered "you are not in the authorized list"), so no node has the file
yet. Until it does, the qualification runs on the synthetic file below.

**What the steered name serves.** According to its card, the published vector removes
safety refusals (1/32 to 23/32 compliance on the card's refusal suite at α 0.75) and puts
stray CJK characters in about a third of replies. Any client that can reach the endpoint
can select it by name. The stock name is unchanged.

## Tests

**CPU (done, 2026-10-28).** These ran on a Python 3.12 container with torch 2.13.0 (CPU)
and tilelang 0.1.15 (the image's versions) and vLLM's common requirements, on the
owner's laptop with no network; the vLLM tree was mounted read-only.

- `tests/models/test_deepseek_v4_1_glp.py` and `tests/test_weightless.py`: 60 passed,
  1 skipped. The skip is the published-file test, which needs the file.
- `tests/test_envs.py`: 57 passed. `tests/entrypoints/serve/lora/test_serving_models.py`:
  9 passed.
- `tests/v1/core/test_kv_cache_utils.py` and `tests/entrypoints/openai/models/test_models.py`:
  134 passed, 39 failed, 6 errors, the same counts on the 1025 head. The failures and errors
  need a model download (`Qwen/Qwen3-0.6B`).
- Both new TileLang kernels lower to CUDA source for SM120: `glp_steer` at world sizes 1
  and 4, and `mark_rows` with int32 and int64 indices. The generated source is as designed:
  - an unmarked block returns before any barrier;
  - a marked row adds the parts in rank order, rounds to BF16, then folds in the post
    kernel's order (the first stream's product, `post`·x, the other streams);
  - a warp-then-block dot product per stream;
  - one BF16 rounding of the projected stream.
- `dual_probe.py` passes against a stub server, a check of the probe itself.

**GPU (prepared).** `tests/kernels/test_deepseek_v41_tilelang_glp.py` (SM12x) checks the
following. It runs with the release tests ([bundles/release-tests](bundles/release-tests),
98 gate files: 1025's manifest plus `tests/test_weightless.py`).

- a marked row is within one BF16 rounding of the FP64 projection of mHC's fold, at world
  sizes 1-4;
- the next layer's fold gives the marked row back exactly;
- unmarked rows are untouched, bit for bit;
- batch and mark independence, and repeatability;
- α 0 equals the fold;
- a NaN stays in its row;
- zero tokens, and the kernels' declarations;
- the row marks.

## Runs

| Spec | What | Gate |
| --- | --- | --- |
| [qualify-tp4.json](qualify-tp4.json) | build `spark-ds41f:19f2c20ed4d6-glp-1` on dgx2 and copy it to the ring; the release tests on dgx2; boot [synthetic-tp4.json](synthetic-tp4.json); the determinism gate, `dual_probe.py --speed`, 16 distinct streams | release tests pass or skip; determinism hashes equal 1025's (`3eab5112` / `db25fde3` with log probabilities, `c9cf9dbe` / `c6bb4f4e` as text): stock with steering configured is 1025; every `dual_probe.py` check passes |
| [release-tp4.json](release-tp4.json) | `release` and `single-stream` suites: the candidate, 1025, the candidate again | no metric worse than 1025 beyond the repeat's interval: the cost of steering configured and unused |

- **The synthetic arm.** [synthetic-tp4.json](synthetic-tp4.json) mounts
  `cache/glp/synthetic-0.gguf`, a random unit direction per layer
  ([make_glp.py](../2026-10-23-next-release/glp/make_glp.py), seed 0; sha256 `2a49af60…`, on
  all four nodes since 2026-10-23), in place of the published file. It has the same layers,
  α, kernels and row marks, so it qualifies the mechanism and its cost. It does not show the
  published vector's behaviour.
- **What `dual_probe.py` checks** (stock and steered, greedy, compared on token IDs and
  log probabilities):
  - both names are listed and echoed;
  - the steered replies repeat, and they differ from stock;
  - each name gives its alone replies in a batch shared with the other;
  - neither name reads the other's prefix-cache blocks;
  - completions and tokenize serve the steered name, and beam search refuses it.
- **What `dual_probe.py --speed` measures:** 1 and 16 concurrent streams on each name. The
  time per engine step comes from vLLM's draft counters, which removes the confound of
  steered text changing draft acceptance.

**Not run yet.** The cluster has been held since 2026-10-27 21:47 UTC by another lab queue
(window `dgx1-1793137674`, running the `one-at-a-time` experiment's benches), so nothing
here has touched a GPU. Both specs pass `scripts/lab.py run --dry-run`. They need this
experiment on `origin/main`, because a window and its `sync` job deploy only that.

## Promotion (owner's acceptance after the runs)

Following AGENTS.md, one commit would do all of the following:

- copy [cluster.json](cluster.json) to `config/cluster.json` with the new image and its
  labels;
- point `upstreams.lock.json` at this series and source manifest;
- add the GLP download step to docs/setup.md section 5;
- describe the second name in README.md and docs/current-state.md;
- classify `tests/test_weightless.py` (gate) in `config/release-tests.json`, and update the
  two GLP entries;
- record a new baseline under `manifests/baselines/` from `release-tp4.json`'s run.

That waits for:

- the published file on every node, and one more boot of [rc-tp4.json](rc-tp4.json)
  through `dual_probe.py` on it (the behaviour change on refusal prompts is the card's,
  not ours);
- a TP3 run (`tp3-triangle` shares the recipe) before the triangle serves it.
