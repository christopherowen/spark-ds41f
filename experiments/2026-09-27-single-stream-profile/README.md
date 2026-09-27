# Single-stream decode and prefill profile of r5e

Base: the promoted configuration (`2026-09-27-karmic-kraken-r5e`) with vLLM's
torch profiler enabled (`cluster-profile.json`, from `make_arms.py`).

## Question

Where does a single-stream decode step (~47 ms) and a real-text prefill
chunk spend their time on r5e? The answer ranks the work to close the gap on
single-stream decode and prompt processing.

## Method

One boot of `cluster-profile.json`, then `capture.py`:

- one warm request, then a profiled request: the `explain` case, 256 tokens,
  reasoning off (the first 24 engine steps are recorded);
- a profiled cold prefill of about 16K tokens of real text (`source_text`).

Traces land in `cache/kkref/profiles/r5e/` on each node; rank 0 is analyzed.
The promoted service is restored afterwards.

## Results

Pending.
