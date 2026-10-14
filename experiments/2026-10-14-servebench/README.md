# servebench: steady-state load beside the burst measurements

Owner, 2026-10-14: adopt the better benchmarking techniques (MLPerf Inference's server
scenario, InferenceX's throughput-against-interactivity curves, GuideLLM and AIPerf's
goodput), built so the benchmark can later become its own project. Phase 1 is
[servebench](../../servebench/README.md): closed-loop and Poisson load with a warm-up and
drain outside the measurement window, 256 prompts, token-exact windows, batch-mean
intervals, and output tok/s split into engine steps/s x tokens per step from the server's
counters.

## Validation ([validate.json](validate.json))

On production r6h, in a lab window: closed loop at sixteen and eight requests in flight
(60 s warm-up, 300 s window, ten sub-windows), Poisson at 0.6 requests/s (about 70% of
sixteen streams' capacity) with goodput limits of 2 s to first token and 100 ms per
token, the sixteen-stream run again with the same seed (does its interval hold run to
run?), then the burst distinct-stream measurements at sixteen and eight for comparison.

## Engine cost grid ([engine-validate.json](engine-validate.json))

`servebench engine` times the cost profile's fixed-shape dummy steps inside the server, the
same work on every build (vLLM `engine-grid` `2e5bc3de3`: `GPUModelRunner.engine_grid` and
`Worker.engine_grid` for the dev-mode collective RPC, mounted from [overlay](overlay); both
files are unchanged across r6h, r6j and r6k).

The run: r6k ([engine-r6k.json](engine-r6k.json)), r6h ([engine-r6h.json](engine-r6h.json)),
then r6k again on a fresh boot, each with `VLLM_SERVER_DEV_MODE=1` and two grids per boot.
That gives:
- r6k against r6h on identical work;
- the within-boot repeatability (grid 1 against grid 2);
- the boot-to-boot spread (r6k against r6k again), the smallest difference worth believing.

After its two grids, each boot also runs `servebench model`:
- r6k generates greedily and scores its own text;
- r6h and the second r6k boot score r6k's text (`--corpus`).

That gives:
- r6k's self-agreement: prefill against decode arithmetic;
- r6h against r6k on identical tokens: KL divergence, top-1 agreement, perplexity;
- acceptance by draft position for each build;
- the model-behaviour A/A. With batch-invariant greedy decoding the second boot should
  reproduce the first exactly.

## Results (2026-10-13 23:18 to 2026-10-14 00:18 UTC)

### Steady state on r6h ([validate.json](validate.json))

| run | output tok/s | ±95% | steps/s | output tokens/step |
| --- | ---: | ---: | ---: | ---: |
| closed, 16 in flight | 226.07 | 0.62% | 7.260 | 31.17 |
| closed, 16 again | 226.04 | 0.61% | 7.263 | 31.14 |
| closed, 8 in flight | 181.12 | 0.53% | 11.26 | 16.12 |
| Poisson 0.6/s | 155.11 | 16.9% | 14.84 | 12.48 |
| burst, 16 distinct (4 samples) | 224.91 | 0.76% | | |
| burst, 8 distinct (4 samples) | 178.93 | 0.23% | | |

- **A/A within a boot:** -0.02% between the two sixteen-stream runs, inside ±0.6%. The
  batch-mean interval is honest. A run is 60 s warm-up, 300 s window and the drain.
- **Steps/s and tokens per step move against each other** (±1.8% and ±2.1%, product
  ±0.6%). Adaptive verification verifies more drafts when more are accepted, which makes
  steps bigger. Under adaptive verification steps/s is not text-free, so kernel changes
  are judged on the engine grid.
- **Serving runs about 2.1 rows a request per step** at 8 and 16 streams (33.6 and 17.4
  scheduled tokens a step). Grid shapes at five drafts (96 and 48 rows) are not what
  serving runs, so the default grid now has depths 1 and 2.
- **Poisson throughput is the offered load** (±17%), not speed. Goodput at 2 s to first
  token and 100 ms per token was 91.7%; every miss was time to first token, in bursts of
  arrivals. Use it for release confirmation and scheduler tuning.

### Engine grid ([engine-validate.json](engine-validate.json); 30 replays, no settle runs)

- **Per shape:** ±0.6-1.0% pointwise (±1.3-2.0% for all shapes at once). An 18-shape grid
  took about 95 s.
- **Within a boot:** the second grid ran 0.2-0.6% slower (geometric mean) in all three
  boots. That is thermal drift.
- **Boot to boot (r6k against r6k again):** +0.5-0.7%, the later boot slower. So the
  grid's floor today is systematic drift of about half a percent, not sampling noise.
- **r6h against r6k at decode:** level (geometric mean -0.0% to +0.5%), as the release
  table's distinct-prompt measures found.
- **Dummy prefill disagrees with real prefill.** It called r6k 1% slower at every depth,
  where real cold prefill was level or faster. It also ran 1.7x faster than a real 16K
  prefill's kernels and did not slow with context. It is out of the default grid.
- **Order artifact.** A step after a jump between 1M and no context cost 11 ms more, in
  either direction: the 1M shape alternated 21 / 32 ms. The dummy metadata is rebuilt on
  the host inside the timed span. Fixed by settle runs (vLLM `engine-grid` 326935990).

### Model behaviour ([engine-validate.json](engine-validate.json))

- **Self-agreement (r6k):** every greedy token ranks first when the same build scores it
  teacher-forced (65,536 positions). Prefill and decode arithmetic agree on every choice.
  Perplexity of its own text 1.262, acceptance length 2.327.
- **A/A across boots (r6k against r6k again):** bit-identical. 256/256 responses, every
  distribution equal (KL 0), acceptance equal to the last digit. Text and acceptance add
  no noise.
- **r6h on r6k's text:**
  - greedy responses all differ, diverging at token 29 (median; p10 3);
  - top-1 agreement 95.13%;
  - KL 0.0207 nats (±0.0009 over sequences);
  - perplexity 1.262 -> 1.289, though the text is r6k's own, which favours r6k;
  - acceptance level (2.317 against 2.327).
  For scale, llama.cpp reports about 0.001 nats for 8-bit and 0.01-0.03 for 4-bit
  quantizations against FP16. The difference is mostly B12X's FP8 quantization of WO-A's
  input, which r6k removed to match the reference.
- **Cost:** generation 283 s, scoring 40 s, result 15 MB compressed.
