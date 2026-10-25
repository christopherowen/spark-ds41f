# servebench

Steady-state load measurements for OpenAI-compatible LLM servers. Standard library only;
nothing here imports the cluster tooling, so it can move to its own repository unchanged.

```sh
python3 -m servebench run http://127.0.0.1:8000 --mode closed --concurrency 16 \
    --warmup 60 --duration 300 --output c16.json
python3 -m servebench run http://127.0.0.1:8000 --mode poisson --rate 0.6 \
    --slo-ttft 2 --slo-tpot 100 --output poisson.json
python3 -m servebench compare base.json candidate.json
```

## Why

The distinct-stream measurements it complements start N requests at once and time from
the first token to the last. That window holds a ramp (all prefills together) and a
tail (fast streams finish first, so the batch drains toward one), so a "16-stream"
number is partly an 8-, 4- and 1-stream number. Sixteen prompts also leave speculative
decoding's acceptance tied to a handful of texts, and a change in the text (a precision
change) then reads as a change in speed. The methods here follow MLPerf Inference's
server scenario, GuideLLM and AIPerf.

## Method

- **Load.** `closed` keeps N requests in flight, starting the next as each ends, so the
  server sees a steady N with new prefills mixed into decode steps. `poisson` sends
  requests at a fixed rate whatever the server does (open loop), so queueing shows as
  latency. `burst` reproduces the old shape for comparison.
- **Context.** With `--context TOKENS` (closed loop) each stream owns a long document,
  Python standard-library source sized by the server's tokenizer, and asks every question
  after it under one cache salt. Each document is prefilled once before the clock starts,
  so the window measures decode at that depth plus each question's short prefill, as with
  a long conversation or a coding agent's repository context. The window's prefix-cache
  hit rate shows whether the documents stayed cached. Streams x context must fit the KV
  cache.
- **Window.** A warm-up (default 60 s) runs before the measurement window (default
  300 s); when the window closes no new requests start, and the in-flight ones drain
  outside it.
- **Tokens.** Every streamed chunk is timed with the request's cumulative token count
  (vLLM's per-chunk usage), so the window counts exactly the tokens produced inside it.
- **Prompts.** 256 distinct prompts (sixteen task styles by 32 topics), each request with
  a fresh cache salt so repeats never hit the prefix cache, and by default exactly
  `--max-tokens` output tokens each (`ignore_eos`), so output length is not a variable.
- **Intervals.** The window is cut into equal sub-windows (default 10); each is one
  sample of throughput and of every server rate (batch means), with Student t intervals.
- **Speed and text apart.** With the vLLM adapter the server's counters are read at
  every sub-window boundary: output tok/s = engine steps/s x output tokens per step.
  The step rate is the engine's speed; tokens per step moves with draft acceptance, that
  is with the text. A precision change that alters the text shows in the second, a
  kernel change in the first.
- **Latency.** Time to first token, time per output token and per-user tok/s
  (percentiles over requests started in the window); with `--slo-ttft` / `--slo-tpot`,
  SLO attainment and goodput (requests per second meeting both limits).
- **Compare.** Batch-mean metrics get a Welch interval; percentile metrics are judged
  against a tolerance (default 3%) and marked "no interval". The comparison warns when
  the two runs' workloads differ.

## Suites

```sh
python3 -m servebench suite http://127.0.0.1:8000 --suite screen --out runs/candidate --label candidate
python3 -m servebench suite http://127.0.0.1:8000 --suite screen --out runs/base --label base \
    --corpus runs/candidate/model.json.gz
python3 -m servebench compare runs/base runs/candidate
```

A suite is a fixed, named set of measurements, kept as data in
[suites/](suites/): steps, each a servebench command (`model`, `engine`, `run`,
`prefill`, `stall`) with its arguments. Every build measured with a suite is measured the same way:
- a run writes each step's result and `suite.json`, which records the suite and its
  SHA-256, plus whatever `--metadata FILE` holds (a JSON object, stored as given: the
  cluster tooling records the deployment's identity there);
- `compare` on two run directories compares step by step, and skips a step whose output
  a run did not keep (a reference kept without the model step's text);
- it refuses runs of different suites, so a changed suite is a new suite.

Built in:
- `screen`, about 18 minutes a build: model behaviour on 128 prompts (first, so it also
  heat-soaks the GPU), the engine grid, sixteen streams in a closed loop, and prefill at
  32K and 256K.
- `release` (version 2), about 51 minutes: all 256 prompts, eight streams as well,
  Poisson arrivals at 0.7 requests/s (about 80% of sixteen streams' rate) with goodput
  under a 2 s time to first token and 100 ms per output token, and prefill from 32K to
  1M. Version 1 had no Poisson step.
- `screen-tp3` and `release-tp3`: the same for a server of eight streams and 1M of
  context. They run eight (and four) streams, the grid up to eight requests
  (`--max-requests`), Poisson at 0.42 requests/s and prefill at the ring's sizes, to 1M.
  `release-tp3` version 1 stopped at 500K, from when the triangle served 512K.

Every built-in suite times the engine grid, so it needs a development server (below).

`--corpus` makes model steps score another run's text, so builds compare on identical tokens.

## Prefill

`python3 -m servebench prefill URL --sizes 32768,262144 --repeats 3` times cold prompts of
fixed sizes. Each prompt is Python standard-library source from a fixed seed, so every
build reads the same text; it is sized by the server's tokenizer and sent with a fresh
cache salt. The rate is prompt tokens over time to first token. Repeats go round the
sizes in turn, and comparisons use Welch intervals.

## Stalls during a long prefill

`python3 -m servebench stall URL --sizes 131072 --streams 8 --repeats 3` measures what
other streams wait while one long cold prompt prefills. The streams decode first (fixed
length, fresh salts, the default prompts); after `--warmup` seconds the prefill step's
document is sent for one output token, and the streams stop `--tail` seconds after its
first token. A stream's gaps between output chunks are its step times: those ending
before the document was sent are the decode baseline, those overlapping its prefill the
stall. It reports the worst overlapping gap (the longest step that carried a prefill
chunk, which the long-prefill budget sets), the overlapping gaps' median and p99, the
baseline median and the prefill rate under decode. A repeat fails unless every stream
decoded across the whole prefill (`--max-tokens` must outlast warm-up and prefill).

## Engine cost grid

```sh
python3 -m servebench engine http://127.0.0.1:8000 --output grid.json
python3 -m servebench compare base-grid.json candidate-grid.json
```

The server times its own fixed-shape dummy steps, the speculative-decoding cost profile's
steps, at the shapes asked for. The default grid is verification rows (requests x (1 +
depth)):
- 1, 4, 8 and 16 requests at depths 1 and 2, at 4K context. This is what adaptive
  verification runs under load: about two rows a request at 8-16 streams.
- 1-16 requests at the full five drafts, at 4K.
- Depth 2 at 64K and 256K and at the server's longest context.

Every build does identical work with no text, acceptance or scheduling in it, so the step
times are the engine's speed. This is the measurement that judges kernel and engine changes
at decode; tok/s judges what users get. It needs vLLM's dev-mode collective RPC
(`VLLM_SERVER_DEV_MODE=1`) and a worker exposing `engine_grid`; a production server should
not expose that RPC, so suites with a grid step run against a development server. It
refuses a busy server and resets the prefix cache afterwards, since dummy steps write KV
slots. `--max-requests` (the server's `--max-num-seqs`, default 16) leaves out shapes of
more requests.

- **Timing.** Each shape is warmed, then timed in rounds of alternating order (default
  100). Each timed step follows an untimed one of the same shape (`--settle`): switching
  shapes rebuilds the dummy metadata, which once cost 11 ms after a jump between 1M and
  no context. That is the harness's cost, not the engine's.
- **Comparison.**
  - Slow outliers (above the median by five scaled MADs) are dropped first.
  - The Welch intervals hold for all shapes at once (Bonferroni), so a grid of twenty
    shapes does not flag one by chance.
  - The decode shapes' geometric mean gives one overall verdict.
- **Prefill is not in the default grid.** Dummy prefill ran 1.7x faster than a real 16K
  prefill's kernels, and called a build 1% slower that real cold prefill found level or
  faster, presumably because identical dummy tokens route the MoE unlike text. `--prefill`
  adds 8K chunks for completeness; judge prefill on real prompts.

### Boot-to-boot spread

`python3 -m servebench spread GRID... [--against GRID...]` takes engine grids of one build,
each from its own boot. It gives every boot's decode geometric mean against the set's mean,
and their standard deviation. With `--against`, it compares two sets with boots as the
samples. Within a boot, grids repeat to about ±0.05%; between boots they moved up to 0.5%
(2026-10-14), so changes this small need several boots per build.

## Model behaviour

```sh
python3 -m servebench model http://127.0.0.1:8000 --label base --output base.json.gz
python3 -m servebench model http://127.0.0.1:8000 --label candidate --corpus base.json.gz \
    --output candidate.json.gz
python3 -m servebench compare base.json.gz candidate.json.gz
```

What a build computes, with no timing in it, so a repeat gives the same answer:
- **Greedy generation.** Every prompt once at temperature 0, at a fixed concurrency (default
  16), with each response's token ids. The speculative-decoding counters are read before
  and after: mean acceptance length, draft tokens verified per step, and per draft
  position the acceptance rate and how often that position was verified. On a
  batch-invariant server the text, and with it the acceptance, belongs to the build.
  Compared runs report how many responses are token-identical and where the rest diverge.
- **Teacher-forced scoring.** Token sequences (by default this run's own prompts and
  responses; with `--corpus`, an earlier run's) go back as prompts with
  `prompt_logprobs`, keeping per response token its log-probability and rank and the top K
  (`--top`, vLLM's `--max-logprobs` limit, default 20). Two builds that scored the same
  tokens compare position by position:
  - identical distributions;
  - top-1 agreement;
  - KL divergence, estimated from the top K with the rest of the vocabulary as one bucket;
  - the given token's log-probability change;
  - perplexity.
- **Self-agreement.** A build scoring its own greedy text should rank every given token
  first. Anything less means its prefill arithmetic and the decode arithmetic that wrote the
  text pick different tokens.

The base run is a property of the base build, so it can be kept and reused for every
candidate measured against that build.

## Output

`--output` writes `servebench/1` JSON: tool, server, workload (with the prompt set's
SHA-256), and summary (`output_tps`, `server`, `server_batches`, `latency`,
`inflight_mean`, `requests_per_s`, `requests`). `--save-requests` adds every request's
timeline. `engine` writes `servebench-engine/1`, `model` `servebench-model/1`; a path
ending in `.gz` is compressed. The last stdout line is a one-line summary
(`{"summary": ...}`) for tables.

## Planned

- Concurrency and rate sweeps: throughput against per-user speed.
- A context sweep profile: `--context` at about x4 steps to the configuration's longest
  context, with fewer streams where the KV cache requires it.
- Projected tok/s from the engine grid and the measured acceptance.
- Adaptive window lengths that stop once the interval is tight enough.
