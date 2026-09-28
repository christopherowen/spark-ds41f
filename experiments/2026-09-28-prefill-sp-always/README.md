# Sequence-parallel prefill on every prompt chunk

Base: the promoted configuration (`2026-09-27-karmic-kraken-r5g`).

## Question

r5g runs prefill sequence-parallel only for forwards of at least 2048
tokens (`SPARK3_DS41_PREFILL_SP_MIN_ROWS`). That figure came from another
stack, not from a measurement here. The only structural limit is decode:
decode forwards (at most 48 tokens) replay captured CUDA graphs and never
use SP. Does SP pay on every other forward, down to small prompt chunks and
prompt chunks mixed into decode batches? If it does, the setting goes away.

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `current` | none (SP from 2048 tokens) |
| `always` | `SPARK3_DS41_PREFILL_SP_MIN_ROWS=1`, which the code raises to 49: every forward above the decode graph sizes |
| `always2` | vLLM patch 0018 mounted over the r5g image, no setting. Round two: SP above the decode sizes, small reduce-scatters through the RoCE all-reduce. Round three (reworked 0018): SP where the reduce-scatter exceeds the RoCE all-reduce limit |
| `fd` | vLLM patch 0019 (`vllm/env_override.py`) mounted over the r5g image: custom-op `fill_defaults` reads the schema once |
| `both` | the reworked 0018 and 0019 |
| `*-profile`, `current-stack` | the torch profiler on `current` and `always2` (and with Python stacks), for `profile.sh` |

## Workload and gates

`sequence.sh`: current, always, current, always (ABAB). Each runs the LRU
gate, decode at one and eight streams (eight-stream runs mix new prompts
into decode batches), and cold prefill of real text at about 64, 128, 256,
512, 1024, 1536, 2048 and 4096 tokens, four repeats each.

## Results

First round (2026-09-27 23:10-23:36 UTC, ABAB), prefill TTFT in ms:

| Tokens | 73 | 154 | 251 | 478 | 1060 | 1431 | 1776 | 3772 |
|---|---|---|---|---|---|---|---|---|
| current | 227.0 | 249.0 | 287.5 | 345.0 | 466.5 | 552.5 | 639.5 | 976.0 |
| always | 236.0 | 250.0 | 289.5 | 341.0 | 450.5 | 519.0 | 609.0 | 980.0 |
| change | +4.0% | +0.4% | +0.7% | -1.2% | -3.4% | -6.1% | -4.8% | +0.4% |

SP gains from about 500 tokens: the 2048-token cut was leaving 3-6% on
1-2K prompts. Below that it lost up to 9 ms: each encoder layer's two
reduce-scatters went through NCCL at 100-200 us per call on small
messages, against about 16 us for the RoCE all-reduce. Eight-stream decode,
which mixes short prompts into decode batches, read 1-3% lower. Patch 0018
routes those small reduce-scatters through the RoCE all-reduce (`always2`).

Second round (2026-09-27 23:43-00:10 UTC, ABAB, the first 0018), prefill
TTFT in ms:

| Tokens | 73 | 154 | 251 | 478 | 1060 | 1431 | 1776 | 3772 |
|---|---|---|---|---|---|---|---|---|
| current | 225.5 | 258.5 | 289.0 | 342.5 | 466.0 | 553.0 | 639.0 | 982.5 |
| always2 | 237.5 | 256.0 | 285.0 | 338.0 | 449.0 | 520.0 | 607.0 | 981.0 |
| change | +5.3% | -1.0% | -1.4% | -1.3% | -3.6% | -6.0% | -5.0% | -0.2% |

Routing small reduce-scatters through the RoCE all-reduce did not remove the
loss on the smallest prompt. Decode stayed within noise.

### Tiny-prompt profile

`profile.sh` boots `current-profile` and `always2-profile` once each;
`capture_tiny.py` records six prefills of 54-74 tokens. Rank 0, median of
the prefill steps:

| | current | always2 (first 0018) |
|---|---|---|
| step wall | 236 ms | 243 ms |
| GPU busy | 157 ms | 159 ms |
| routed MoE | 119 ms | 119 ms |
| collectives | 9.2 ms (81 RoCE all-reduces) | 16.2 ms (+84 RoCE all-gathers) |
| mHC and norms | 7.5 ms | 6.4 ms |

At this size SP cannot pay. Its reduce-scatter is an all-reduce plus a slice,
the same cost as the full-row path, and 60-row kernels are latency-bound, so
splitting the rows saved 1 ms while the two all-gathers per layer cost 6 ms.

The profile also shows the forward is host-bound: the GPU idles 74 ms of the
236 ms, 59 ms of it inside `b12x::tp_moe_dynamic_launch`, which holds the
host for 1.39 ms per MoE layer without a CUDA call. Repeated `py-spy dump`
samples of the rank-0 worker put 70% of that in
`torch._library.utils.fill_defaults`, called by the ADInplaceOrView kernel
that `torch.library.custom_op` registers for ops with `mutates_args`. It
reads `schema.arguments`, which builds a new list of every argument, once per
argument. Most of the rest is CuTe DSL argument rectification, whose
`missing in args` check compares every tensor argument. A CPU microbenchmark
at the launch op's shape (43 tensors, 21 scalars, 18 mutated):

| | per call |
|---|---|
| torch `fill_defaults` | 873 us |
| reading `schema.arguments` once | 48 us |
| `missing in args` over 20 tensors | 199 us |
| identity check over the same list | 1.3 us |

A torch-profiler capture with Python stacks (`current-stack`) is not usable
on dgx1: building the trace at `stop_profile` took MemAvailable to 1.5 GiB and
the memory guard stopped the service.

### Changes

- Patch 0018, reworked: a forward runs sequence-parallel when it is prompt
  processing and its padded hidden-state message exceeds what the one-shot
  RoCE all-reduce takes (2 MiB, so 205 tokens at TP 3 with 5120 bf16 hidden
  rows); below that SP only adds the all-gathers. Reduce-scatters are plain
  NCCL reduce-scatters again. No setting (`always2`).
- Patch 0019: `vllm/env_override.py` installs a `fill_defaults` that reads
  the schema once and returns the same arguments (`fd`).

Third round (2026-09-28 00:52-01:35 UTC; current, both, fd, always2,
current, both), prefill TTFT in ms (current and both are means of two runs):

| Tokens | 73 | 154 | 251 | 478 | 1060 | 1431 | 1776 | 3772 |
|---|---|---|---|---|---|---|---|---|
| current | 223.5 | 255.5 | 292.5 | 346.0 | 468.0 | 554.5 | 646.5 | 978.0 |
| fd | -11.4% | +1.4% | -1.9% | -1.2% | -1.3% | -0.5% | -1.8% | -0.3% |
| always2 | +0.2% | +1.4% | -2.2% | -1.2% | -4.7% | -6.4% | -6.6% | -0.2% |
| both | -11.2% | -2.0% | -1.4% | -1.6% | -4.0% | -6.3% | -5.6% | -0.3% |

Decode (tok/s, current / both): code answers 78.0 / 79.1 at one stream and
228 / 234 at eight, prose answers 53.4 / 54.6 and 169 / 167, within noise.
LRU 5/5 on every run; dgx1 minimum MemAvailable 6.59-6.64 GiB.

The reworked 0018 removes the small-prompt loss and keeps the 4-7% on 1-2K
prompts; 0019 takes 25 ms off the smallest prompts, where the forward is
host-bound, and is neutral elsewhere. They compose.
