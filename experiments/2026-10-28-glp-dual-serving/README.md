# GLP dual serving: the stock model and its steered variant under two names

One engine serves `deepseek-v4.1-flash` stock beside `deepseek-v4.1-flash-weightless`,
the same weights steered with the published GLP file. A request picks a variant by its
`model`. The goal is a recipe that serves both by default, with no special setup beyond
downloading the GLP file.

- **Base:** spark-ds41f-1025 (vLLM patch head 11171760, tree 6f8e062f; image
  `spark-ds41f:19f2c20ed4d6-next-6`), spark-ds41f main c1cf4990.
- **vLLM:** branch `glp-dual-serving` (local, not pushed), one commit on the 1025 head:
  [vllm/0538](vllm/0538-DeepSeek-V4.1-GLP-steering-per-request-under-a-secon.patch), head
  be780056, tree 89dc5ca4, image `glp-3` (`glp-1` was ee78356d, tree ebb7aaa8; `glp-2`
  6bfd56bb, tree 3e48d992). [vllm/series](vllm/series) is 1025's series plus 0538, and
  [source.json](source.json) and [upstreams.lock.json](upstreams.lock.json) are its lock.
  0538 is provisional: 0537 is taken by the copy-drafter branch, and the series is
  renumbered where it joins a release.
- **The variable:** steering selected per request by model name (0538), with the recipe
  serving both names.
- **Status:** `glp-1` qualified on a synthetic file; `glp-3` (the options below, the
  published file and AdvBench) is queued (see [Runs](#runs)).

## What changes

**In vLLM (0538).** Before 0538, `WEIGHTLESS_STEER_PATH` steered every request, and each
covered layer folded every row explicitly. That cost 2.4% on the decode grid in the
next-release qualification ([2026-10-23-next-release](../2026-10-23-next-release/README.md)).
0538 makes steering a property of each row:

- Two options, set together or not at all: `WEIGHTLESS_STEER_PATH`, the GLP file, and
  `WEIGHTLESS_STEER_MODEL` (new), the served model name whose requests alone are steered.
  One without the other stops the boot. The file is applied as published: the
  `WEIGHTLESS_STEER_ALPHA` and `WEIGHTLESS_STEER_LAYERS` overrides are gone (no recipe set
  them), and so is steering every request (`glp-3`; `glp-1` and `glp-2` kept both).
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
`config/cluster.json`).** One `glp` block configures the steered name:

```json
"glp": {
  "served_model_name": "deepseek-v4.1-flash-weightless",
  "repo": "msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-39-L1-39-a0.75",
  "revision": "2715ffd51f59899115853e1118db080536966c44",
  "file": "glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf",
  "sha256": "9c66d47701e0ceb42c33906f2d3789c0f1f16b21e7d97185f20f7f3f09fbf5cf"
}
```

[scripts/recipe.py](../../scripts/recipe.py) renders it in three places:

- the second `--served-model-name`, after `deepseek-v4.1-flash`. The stock name stays
  first, so `bin/spark` and every probe that takes the first entry of `/v1/models` keep
  using stock;
- `WEIGHTLESS_STEER_MODEL=deepseek-v4.1-flash-weightless` and
  `WEIGHTLESS_STEER_PATH=/glp/vector.gguf`;
- a read-only mount at `/glp/vector.gguf` of the file in the node's hub cache, at the
  pinned revision.

A profile does not set those itself; an arm can rename the steered model (`set`) or drop it
(`unset: ["glp"]`). The file is 803,392 bytes, MIT. `bin/spark doctor --live` and
`cluster start` report a node whose file is missing or differs from `sha256`, with the
download command, before anything boots. The repository's `main` has moved on (c275c3ad,
which adds an `a1.0-dealign-attn` file); the pin keeps the audited vector.

The one setup step a site needs, beside the model's download (docs/setup.md section 5, at
promotion):

```sh
hf download msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-39-L1-39-a0.75 \
  glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf \
  --revision 2715ffd51f59899115853e1118db080536966c44
```

The repository is gated: the downloading account must first accept the author's
responsible-use agreement; anonymous downloads are refused too. On 2026-10-28 the owner
accepted it, and the file was downloaded on dgx1 at the pinned revision (its sha256 equals
the Hub's LFS hash) and copied to dgx2, dgx3 and dgx4 at the same hub-cache path, where
each copy's sha256 was checked.

**What the steered name serves.** According to its card, the published vector removes
safety refusals (1/32 to 23/32 compliance on the card's refusal suite at α 0.75) and puts
stray CJK characters in about a third of replies. Any client that can reach the endpoint
can select it by name. The stock name is unchanged.

## Tests

**CPU, `glp-3` (2026-10-28).** On the owner's laptop, Python 3.12 with torch 2.13.0 (CPU),
tilelang 0.1.15 and vLLM's common requirements, without vLLM's shared `conftest.py` (its
teardown clears accelerator memory, which crashes CPU torch on macOS):

- `tests/models/test_deepseek_v4_1_glp.py` and `tests/test_weightless.py`: 57 passed,
  with `GLP_TEST_FILE` naming the published file. The published-file test now allows the
  file's `controlvector.*` keys, llama.cpp's control-vector header, which the reader
  ignores; `glp.py` reads the file as 39 directions (layers 1-39), α 0.75.
- `tests/test_envs.py`: 57 passed.

**CPU, `glp-1` (2026-10-28).** These ran on a Python 3.12 container with torch 2.13.0 (CPU)
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
- `dual_probe.py` passes against a stub server, a check of the probe itself; so does
  `advbench.py` (1,040 replies scored, and an unserved steered name refused).

**GPU.** `tests/kernels/test_deepseek_v41_tilelang_glp.py` (SM12x) checks the
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

In `glp-1`'s run (below), 5,738 release tests passed, 160 skipped and 6 failed. All six
failures were in the patch's tests, not its code. Four tests in
`test_b12x_v41_ced_metadata.py` and two in `test_deepseek_v4_1_engram.py` build
`DeepseekV41ModelState` with `__new__` and set the attributes it reads; they lacked the
two the patch adds, `steering` and `steered_requests`. Patch 0538 now sets both to `None`
in those six tests (from `glp-2`).

## Runs

| Spec | What | Gate |
| --- | --- | --- |
| [qualify-tp4.json](qualify-tp4.json) | build `spark-ds41f:19f2c20ed4d6-glp-3` on dgx2 and copy it to the ring; the release tests on dgx2; boot [rc-tp4.json](rc-tp4.json), the published file; the determinism gate, `dual_probe.py --speed`, 16 distinct streams, [advbench.py](advbench.py) | release tests pass or skip; determinism hashes equal 1025's (`3eab5112` / `db25fde3` with log probabilities, `c9cf9dbe` / `c6bb4f4e` as text): stock with steering configured is 1025; every `dual_probe.py` check passes. AdvBench is recorded, not gated |
| [release-tp4.json](release-tp4.json) | `release` and `single-stream` suites: the candidate, 1025, the candidate again | no metric worse than 1025 beyond the repeat's interval: the cost of steering configured and unused |
| [overhead.json](overhead.json) | [bundles/steer-overhead](bundles/steer-overhead/steer_overhead.py) on dgx3 and dgx4: a captured step's cost for no steering, the launch, a PyTorch conditional node and a shared condition, at 2, 16 and 96 rows, unmarked and marked | the conditional arms leave the streams bit-identical to the launch |

- **The synthetic arm (`glp-1` only).** Before the published file could be downloaded,
  `glp-1`'s runs booted `synthetic-tp4.json` and its `-dev` variant (removed with `glp-3`;
  they are in this folder's history at 12ee9ece). These mounted
  `cache/glp/synthetic-0.gguf` in place of the published file: a random unit direction per
  layer ([make_glp.py](../2026-10-23-next-release/glp/make_glp.py), seed 0; sha256
  `2a49af60…`, on all four nodes since 2026-10-23). It has the same layers, α, kernels and
  row marks, so it qualifies the mechanism and its cost. It does not show the published
  vector's behaviour.
- **What `advbench.py` records:** AdvBench's 520 harmful behaviours
  ([walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench) at 9d473054,
  MIT, gated like the GLP file). The owner's account accepted its agreement on 2026-10-28;
  the dataset was converted to JSON lines at `~/.cache/advbench/advbench.jsonl` on dgx1
  (sha256 `7cafb159…`, checked by the script), outside the repository. Every prompt goes to
  both names, greedy, thinking off, at most 256 tokens. Each reply is scored by the refusal
  phrases of llm-attacks' evaluation (the AdvBench paper's attack-success metric) and by
  the stock name answering HarmBench's classifier question. Replies stay in
  `results/private`; only the rates are recorded here.
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

**`glp-1` qualification on the synthetic file (2026-10-28, lab window
`dgx1-1793137674`).**

- Release tests: 6 failed. All six were the test stubs described under Tests.
- Determinism and greedy: identical in every scenario. The hashes equal 1025's
  (`3eab5112` / `db25fde3`, `c9cf9dbe` / `c6bb4f4e`), so stock with steering configured is
  1025.
- `dual_probe.py`: every check passes. Steering changed all 8 prompts' replies.
- Speed, as time per engine step: stock 28.97 ms and steered 29.09 ms at 1 stream; stock
  149.85 ms and steered 150.08 ms at 16 streams. In tokens per second: 66.8 and 67.1 at
  1 stream, 279.2 and 276.0 at 16. The 16 distinct streams gave 256.7 tokens/s.

**`glp-1` release on the synthetic file (stock name only, against 1025).** Greedy output
was identical for 256 of 256 prompts; teacher-forced distributions and draft acceptance were
unchanged. The engine grid's decode geometric mean was 0.78% and 1.37% slower in the
two candidate arms. The arms differed from each other by 0.58%, so the cost is about 1%.
Every throughput measurement was unchanged within its interval: c16, c8, Poisson,
prefill from 32K to 1M tokens, and single stream. This is the cost of steering
configured and unused. The 39 steering launches run in every step, and an unmarked row's
block only reads its mark. Steering a request adds 0.1–0.4% more (the `dual_probe.py`
speeds above).

**Where the unused cost goes, and what a conditional node saves.**
[overhead.json](overhead.json) runs
[bundles/steer-overhead](bundles/steer-overhead/steer_overhead.py) on dgx3 and dgx4. It
times a captured stand-in step of 40 layers at 2, 16 and 96 rows, unmarked and marked,
in four arms:

- no steering;
- the `glp-3` launch in each layer;
- that launch in a PyTorch conditional node in each layer, which sets its own condition
  with a launch of its own;
- that launch in conditional nodes that share one condition, set once a step by a kernel
  that reads the step's marks.

The conditional arms must leave the streams bit-identical to the launch, or the run
fails. If the shared condition costs much less than the launches, patch 0538 takes it
for captured steps. A step that runs eagerly keeps the launch.

`glp-1`'s release run (synthetic arms) was already running when the published file
arrived, and it was left to finish. `glp-2`'s qualification was queued and then withdrawn
before it ran, for `glp-3`: the two options, the `glp` block and AdvBench. `glp-3`'s
two specs boot the published file. Specs run only from `origin/main`, because a window
and its `sync` job deploy only that.

## Promotion (owner's acceptance after the runs)

Following AGENTS.md, one commit would do all of the following:

- copy [cluster.json](cluster.json) to `config/cluster.json` with the new image and its
  labels;
- point `upstreams.lock.json` at this series and source manifest;
- add the `glp` block's rendering note to docs/setup.md section 5 (the download step is
  there already);
- describe the second name in README.md and docs/current-state.md;
- classify `tests/test_weightless.py` (gate) in `config/release-tests.json`, and update the
  two GLP entries;
- record a new baseline under `manifests/baselines/` from `release-tp4.json`'s run.

That waits for:

- `glp-3`'s runs, which boot [rc-tp4.json](rc-tp4.json) with the published file through
  `dual_probe.py` and AdvBench;
- a TP3 run (`tp3-triangle` shares the recipe) before the triangle serves it.
