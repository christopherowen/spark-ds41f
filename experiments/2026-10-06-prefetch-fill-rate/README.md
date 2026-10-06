# L2 prefetch fill rate on TP4

The DeepSeek V4.1 L2 prefetch (`vllm/models/deepseek_v4_1/l2_prefetch.py` on the
GLM-5.3 launcher) pulls the next projections' weights into L2 on a side stream.
Two of its three windows (FFN and NEXT) fire right before the attention and MoE
all-reduces, so the fill and the collective share memory bandwidth. The fill
budget per window is already an "up to" amount (`VLLM_DS41_L2_PREFETCH_*_MB`);
its speed is the number of issuing CTAs, `VLLM_L2_PREFETCH_GRID`, which defaulted
to 4 on GB10. This experiment measures that speed on the TP4 ring at 1 to 16
streams and writes the result into each recipe's environment, where it was
implicit before.

## Result

`VLLM_L2_PREFETCH_GRID=2` is the TP4 value: about 1 to 1.5% faster decode than
the old default of 4 at every concurrency measured, in three separate windows,
with quality 5/5, prefill unchanged and GPU energy per token flat. The best
value did not move with concurrency (1, 4, 8 and 16 streams), so one value per
recipe is enough on TP4. The other values (1, 3, 6) are within about 1% of the
old default.

[config/cluster-tp4.json](../../config/cluster-tp4.json) sets 2. The TP3
recipes ([config/cluster.json](../../config/cluster.json), `cluster-4k`,
`cluster-64k`) set 4, the value they ran with, until a TP3 sweep on the triangle
fabric measures them: the triangle's direct one-shot collective is shorter than
the ring's relay, which changes the window the fill has to fit.

## Runs

All runs used the r6 image and `config/cluster-tp4.json` arms that differ only
in the prefetch environment, on the four-node ring. Changes are against the
mean of the baseline (grid 4) measured first and again last in the same window.

**Screen**, 2026-10-05 23:49 to 2026-10-06 00:14 UTC ([runs/screen](runs/screen)):
lean profile, distinct-prompt 8 and 16 streams.

| Arm | Step, prose / JSON, 1 stream | 8 streams | 16 streams |
| --- | ---: | ---: | ---: |
| prefetch off | +5.0% / +5.2% | -2.8% | -3.1% |
| grid 2 | -1.5% / -1.6% | +0.4% | +0.3% |
| FFN 8 MB, NEXT 10 MB | +0.9% / +0.8% | -1.2% | -0.2% |

**Decode profiles**, 2026-10-06 00:18 to 00:25 UTC
([runs/profiles/summary.json](runs/profiles/summary.json), from
[profile_summary.py](profile_summary.py)): a 128-token single-stream decode
under the torch profiler on every rank.

| Per rank | grid 4 | grid 2 | off |
| --- | ---: | ---: | ---: |
| Profiled span | 1,605 ms | 1,567 ms | 1,684 ms |
| MoE all-reduce, median | 45 to 48 us | 39 to 41 us | 26 to 31 us |
| One-shot collectives, total | 227 to 252 ms | 202 to 221 ms | 143 to 174 ms |
| Dense GEMM, total | 79 to 81 ms | 89 to 93 ms | 173 to 175 ms |

Turning the prefetch off makes every hot all-reduce about 20 us faster (61 to
105 ms less collective time per rank) but costs the compute kernels 266 to 279
ms (the dense GEMMs double and the TileLang projections slow down); the
prefetch more than pays for its collision. Half the fill rate recovers about
40% of the collective time the collision costs (20 to 44 ms per rank) and gives
up none of the compute gain.

**Gate**, 2026-10-06 05:51 to 06:34 UTC ([runs/gate](runs/gate)): the
recipe-limit benchmark for grid 2 (quality, decode on prose and code at 1 to 16
streams with three samples, prefill on source text to 1M with two repeats),
decode-only baselines before and after, and grid 1 as a screen. Concurrent
points here send copies of one prompt.

| Point | grid 4, tok/s | grid 2 | grid 1 |
| --- | ---: | ---: | ---: |
| prose, 1 | 60.9 | +1.5% | -0.4% |
| prose, 8 | 216.0 | +1.4% | -0.6% |
| prose, 16 | 312.5 | +1.7% | +0.6% |
| code, 1 | 73.9 | +1.8% | +0.1% |
| code, 16 | 341.4 | +1.2% | +0.5% |

The two baselines differed by 0.3 to 1.4%. Grid 2's prefill (32K 5,689, 256K
5,382, 500K 4,990 and 1M 4,300 tok/s) is within the intervals of the r6
receipts; the prefetch never fires on prefill chunks, which exceed its
256-token limit. GPU energy over the identical decode phase was 45.7 to 46.0 kJ
in every arm, about 147 W summed over the four GPUs
([energy.txt](runs/gate/energy.txt)). Both baselines ran 2 to 4% below the r6
receipts of 2026-10-05 (the bench's regression lines); the cause is not known
and it affects every arm alike.

**Sweep**, 2026-10-06 07:09 to 07:39 UTC ([runs/sweep](runs/sweep)): lean
profile, distinct-prompt 4 and 16 streams.

| Grid | Step, prose / JSON, 1 stream | 4 streams | 16 streams | Tokens per joule, 16 streams |
| --- | ---: | ---: | ---: | ---: |
| 1 | -0.4% / -0.4% | -0.1% | 0.0% | +0.0% |
| 2 | -1.2% / -1.0% | +1.1% | +1.2% | +0.7% |
| 3 | +0.5% / +1.0% | +0.7% | -0.1% | +0.1% |
| 6 | +0.5% / +0.6% | -1.0% | -0.5% | -0.3% |

Baseline drift was -0.4% / -0.5% in step time, +1.0% at 4 streams and +1.8% at
16 streams. GPU power is flat across arms, so tokens per joule follow
throughput ([energy.txt](runs/sweep/energy.txt), from [energy.py](energy.py)
and the power samples kept for each measured window).

## Rerunning

[scripts/lab_sweep.py](../../scripts/lab_sweep.py) wrote this directory's arms
and run spec from the recipe as it now stands (grid 2 as the base, then 1, 3, 4
and 6):

```sh
scripts/lab_sweep.py --base config/cluster-tp4.json --variable VLLM_L2_PREFETCH_GRID \
  --values 1,2,3,4,6 --experiment experiments/2026-10-06-prefetch-fill-rate \
  --run prefetch-sweep-tp4 --streams 1,4,16
scripts/lab.py run experiments/2026-10-06-prefetch-fill-rate/prefetch-sweep-tp4.json \
  --production-config config/cluster-tp4.json
```

A TP3 sweep is the same command with `--base config/cluster.json` once the
triangle is cabled. TP2 is not a recipe for this model: at TP4 each node holds
73.6 GiB of weights and at TP3 about 97.5 GiB, so about 287 GiB are sharded and
under 2 GiB replicated, and two nodes would need about 145 GiB each against
114 GiB free.
