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

Pending.
