# The FFN prefetch window's pace

Date: 2026-11-18. Base commit: `c0e4124f7` (main). Image: `spark-ds41f:candidate-61` as rebuilt on 18 November (vLLM
tree `42332c66ca46`). Nothing is built: four vLLM files are mounted over the image.

## Question

Does the FFN window of the L2 prefetch end too late for the shared expert's up-projection, and does a faster pace
for that window alone shorten the step?

## Why

[window_timing.py](window_timing.py) on rank 0's trace of candidate-61's single-stream step
([the floor experiment's](../2026-11-17-c1-floor-now/README.md#a-traced-step-of-candidate-61), 310 target forwards,
12,400 layers; [receipts/window-timing-dgx1.txt](receipts/window-timing-dgx1.txt)), in microseconds:

| | p10 | p50 | p90 |
| --- | ---: | ---: | ---: |
| WO fill ends after WO-B ends | -18.5 | -3.1 | 6.2 |
| FFN fill starts after WO-B ends | 0.5 | 0.5 | 6.7 |
| FFN fill, duration (12 MB) | 54.6 | 58.0 | 63.0 |
| FFN fill ends after the shared up-projection starts | -3.6 | 9.5 | 25.3 |
| NEXT fill ends after the MoE all-reduce ends | 9.0 | 24.2 | 34.5 |

- **The WO and NEXT windows are on time.** The WO fill ends with WO-B. The NEXT fill ends 23 to 37 µs before the
  next layer's Q-B starts, and the next layer's fused Q-A/KV ends at the same time whether the fill is early or late
  (32.7 µs after the all-reduce in both halves of a median split).
- **The FFN fill ends after its first large consumer has started,** in more than half of the layers. Splitting each
  layer's forwards at the median: where the fill ends 1.1 µs after the shared up-projection starts, that projection
  takes 22.7 µs and ends 41.2 µs after the attention all-reduce; where it ends 18.1 µs after, 28.4 µs and 44.9 µs.
- **The fill is paced, not bandwidth-bound.** 12 MB in 58 µs is 207 GB/s; the routed experts stream at 266.6 GB/s.
  The kernel waits 200 ns after each 64 KiB request. That wait protects the kernels a fill runs beside: the WO fill
  runs beside WO-A and WO-B, and candidate-61's faster fill for every window stretched WO-A from 29.8 to 65.6 µs
  ([c61-r14](../2026-11-16-c61-decode/README.md)). The FFN fill runs beside the attention all-reduce and the mHC
  FFN mix only. A pace for the FFN window alone has not been measured.

This is a correlation inside one boot, not an A/B: a late fill and a slow projection can share a cause.

## One variable

`VLLM_DS41_L2_PREFETCH_FFN_PACE_NS`: the FFN window's wait between requests (default 200, the other windows' pace,
which do not change). The overlay gives the fill kernel its pace as a compile-time argument, one compiled kernel a
pace, and the FFN plan its own:

- `vllm/models/glm5next/nvidia/tilelang/l2_prefetch.py`: `l2_prefetch_kernel(grid, block, chunk, delay_ns)`;
- `vllm/models/glm5next/nvidia/l2_prefetch.py`: a launcher a pace, `L2Prefetcher.issue(..., delay_ns)`;
- `vllm/models/deepseek_v4_1/l2_prefetch.py`: `PACE_FFN_NS`, `make_ffn_plan`, the second kernel's declaration;
- `vllm/models/deepseek_v4_1/nvidia/model.py`: the FFN plan through `make_ffn_plan` (one line).

At the default pace the FFN plan, the kernel's call and the launcher are the image's own. The layer's log line
names the pace (`FFN ... [pace 100 ns]`).

The arms name the experiment's own copy of candidate-61's source lock ([upstreams.lock.json](upstreams.lock.json),
byte for byte `experiments/2026-11-16-c61-decode/upstreams.lock.json`): under candidate-61's own lock the lab reads
an arm's variable as drift from candidate-61's profile and refuses the arm.

## Prediction, before the runs

- If every layer's forward had the early half's timing, the shared up-projection would end 1.9 µs earlier a layer on
  average: 60 to 75 µs of the 27.5 ms single-stream step, **0.2 to 0.3%**.
- The engine grid's padding rows skip the routed experts, so the same microseconds are a larger share there: about
  0.3 to 0.4% of the grid's geometric mean (19.8 ms), 0.5 to 0.7% of a 1x3 forward. One boot shows 0.42%, so the
  sweep is at the edge of what it can see.
- 12 MB cannot arrive in less than 45 µs at 266.6 GB/s. 100 ns issues the window in about 40 µs, 0 ns in about 21 µs:
  below 100 ns the requests only queue, and the kernel's comment says an unpaced fill can overflow the prefetch FIFO.

## Arms

One boot an arm at `tp4-ring`, the engine grid step alone (`candidate-48/grid-suite.json`), the reference first and
last ([sweep-tp4.json](sweep-tp4.json)):

| Arm | Profile | FFN pace |
| --- | --- | ---: |
| `pace-ref`, `pace-ref-again` | [tp4-pace-ref-dev.json](tp4-pace-ref-dev.json) | 200 ns (default) |
| `pace-100` | [tp4-pace100-dev.json](tp4-pace100-dev.json) | 100 ns |
| `pace-50` | [tp4-pace50-dev.json](tp4-pace50-dev.json) | 50 ns |
| `pace-0` | [tp4-pace0-dev.json](tp4-pace0-dev.json) | 0 ns |

Every arm mounts the same four files; the arms differ in the one variable.

## Marks, declared before the runs

The sweep:

1. An arm is picked when its grid geometric mean is lower than both reference boots' by more than 0.42% (twice the
   0.15% boot SD measured over candidate-55's identical boots, for two single boots). Of several, the lowest.
2. An arm whose grid is lower than both reference boots' by less than that is a lead, not a pick: it goes to the
   confirmation only for its trace (mark 5), which can show the gain where the grid cannot.
3. No arm lower than both reference boots: the FFN window's pace has nothing to give on candidate-61, and nothing
   more runs.

The confirmation, two boots of the picked or leading arm against the sweep's two reference boots:

1. **Grid:** the geometric mean faster beyond 0.30%.
2. **No shape worse:** no shape slower than both reference boots on both boots.
3. **Outputs:** the temperature-0 gate identical to candidate-61's on both boots.
4. **Serving:** single stream and c16 not slower than candidate-61's recorded screen beyond the tool's margin.
5. **The mechanism, in a traced boot:** `window_timing.py` on the arm's trace has the FFN fill ending before the
   shared up-projection starts at the median, and that projection ending at least 2 µs earlier after the attention
   all-reduce than in a traced boot of the reference arm (the trace above has 43.0 µs, the mean of its two halves).

A pick that meets marks 1 to 4 is for the owner to adopt; the pace then becomes the FFN window's only pace in
candidate-61's stack, with no variable. A lead that meets marks 2 to 5 but not mark 1 is recorded as a gain under
the grid's floor, with its predicted share.

## Commands

```bash
bin/spark candidate queue tp4-ring experiments/2026-11-18-ffn-prefetch-pace/sweep-tp4.json --apply
```

## Limits

- `tp4-ring` only. The switched shapes and TP3 have other all-reduce times, so another window length.
- The trace behind the prediction is three prose prompts on one boot of candidate-60 with candidate-61's files
  mounted; the screen runs on candidate-61's built image.
- The overlay's files have no unit tests of their own; the reference arm boots the same files at the default pace.

## Results

Queue entry `20261118T052142-sweep-tp4.json`, five boots, 05:47 to 06:04 UTC on 18 November
([receipts/sweep-tp4-table.txt](receipts/sweep-tp4-table.txt)). The grid's geometric mean over its 20 shapes, in
milliseconds a step, and the six shapes of one request (1x2, 1x3 at four context lengths, 1x6):

| Arm | FFN pace | Grid geometric mean | against `pace-ref` | against `pace-ref-again` | One-request shapes, against the two reference boots' mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pace-ref` | 200 ns | 19.468 | | -0.05% | |
| `pace-100` | 100 ns | 19.571 | +0.53% | +0.48% | +1.93% |
| `pace-0` | 0 ns | 19.592 | +0.64% | +0.59% | +1.92% |
| `pace-50` | 50 ns | 19.572 | +0.53% | +0.48% | +1.56% |
| `pace-ref-again` | 200 ns | 19.478 | +0.05% | | |

- **Every faster pace is slower than both reference boots,** by 0.48 to 0.64% of the grid's geometric mean. The two
  reference boots, first and last, agree to 0.05%, and the table's interval on the geometric mean is 0.11%.
- **The loss is in the shapes the prediction named.** 1x2 rows: 13.56 and 13.51 ms at 200 ns, 13.93, 13.81 and
  13.90 ms at 100, 50 and 0 ns. Each arm is slower than both reference boots in 9 to 11 of the 20 shapes and faster
  than both in 6 (8x16, 4x12, 2x12 and 4x24 in every arm), by less.
- **The arms acted.** Each boot mounted the four files, and the paced arms' containers held the variable
  ([receipts/sweep-tp4-boots.txt](receipts/sweep-tp4-boots.txt), [receipts/sweep-tp4-overlays.txt](receipts/sweep-tp4-overlays.txt)).
  The kernel cache gained three fill kernels during the sweep, `l2_prefetch_segments_paced<65536, 2, 100>`, `<..., 50>`
  and `<..., 0>`. The last reference boot's log has the image's own kernel line and no pace on its FFN plan. The paced
  arms' own log lines were not kept (the watcher cut them before the pace suffix).

## Conclusion

Sweep mark 3: no arm is lower than both reference boots, so nothing more runs. **The FFN window keeps the 200 ns
pace** of the other windows; the variable and the overlay stay in this directory and enter no stack.

The prediction was wrong in sign. Inside one boot a late FFN fill went with a slow shared up-projection, but ending
the fill earlier made the step of one request about 2% slower, not 0.2 to 0.3% faster. The sweep does not say why:
no arm was traced. Two readings the trace could separate, neither measured here: the faster fill takes memory
bandwidth from the attention all-reduce and the mHC mix it runs beside, or lines that arrive earlier are evicted
before the shared expert reads them (the NEXT window follows on the same stream).

The switched shapes and TP3 were not run. A faster pace that loses at `tp4-ring` could not have become one constant
for every shape, which was the only form it would have been adopted in.

### Entries that did not boot

Five queue entries of the sweep (`20261118T005001`, `T005410`, `T005840`, `T022031`, `T041728`) stopped at the
run's validation, before any boot: `invalid spec: ... qualified config drift for tp4-ring:
environment['VLLM_DS41_L2_PREFETCH_FFN_PACE_NS']='100', expected None` (and the same for 0 and 50). The arms then
named candidate-61's lock, whose directory holds the profile the lab compares an arm with. Nothing was measured.
Candidate-61's image was rebuilt meanwhile (tree `8aa03deba93a` to `42332c66ca46`); the overlay's `nvidia/model.py`
is the rebuilt image's file with the one line changed, and the three prefetch files' bases are unchanged in it.
