# Parallel prefills for simultaneous arrivals

Base: the promoted configuration (`2026-09-27-karmic-kraken-r5c`).

## Hypothesis

At eight streams the bench's eight identical 75-token requests start 0.3-0.5 s
apart (the spread of their first-token times), about one admission per step
under `--max-parallel-prefills 1`, and the start stagger explains most of the
spread of eight-stream throughput on reasoning prose (correlation -0.89 over
50 samples). Admitting up to eight prefills per step removes the stagger,
lowers first-token time for the later requests, and raises aggregate
throughput at two to eight streams. The 4,096-token step budget still bounds
each step, so long prompts should not need more memory.

## Arms

| Arm | Change |
|---|---|
| `control` | none (promoted configuration) |
| `mpp8` | `--max-parallel-prefills 8` |

## Workload and gates

One boot each, same protocol: LRU gate, reasoning and answer cases at one
and eight streams (three samples), and four concurrent 64K contexts
(admission, peak KV). Memory guards unchanged; dgx1's lowest MemAvailable
recorded.

## Results

`mpp-s-control`, `mpp-s-mpp8`: LRU 5/5 on both; dgx1's lowest MemAvailable
6.15 and 6.35 GiB.

- The start stagger does not change: the spread of first-token times of the
  eight requests is 0.28-0.39 s with eight parallel prefills against
  0.30-0.44 s without. The requests reach the engine about 45 ms apart, so the
  stagger comes from the front end handling them one at a time, not from
  admitting one prefill per step.
- Eight-stream throughput is unchanged (-2.5 to +2.0%, within noise).
- Four simultaneous 64K prompts: without parallel prefills they get their
  first token at 16, 33, 49 and 65 s (mean 41 s, run 74 s); with eight they
  all wait 62-68 s (mean 67 s, run 79 s). Per-stream decode after the first
  token rises from 12.5 to 23.1 tok/s only because the streams no longer
  decode beside later prefills. First-come-first-served prefill serves users
  better.
- Not adopted: `--max-parallel-prefills 1` stays.
