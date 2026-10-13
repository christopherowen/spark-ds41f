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
